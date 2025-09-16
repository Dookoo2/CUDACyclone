#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}

__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) {
        f = load_found_flag_relaxed(d_found_flag);
    }
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__device__ uint64_t g_pGx[MAX_BATCH_SIZE * 4];
__device__ uint64_t g_pGy[MAX_BATCH_SIZE * 4];

__device__ uint64_t g_Jx[4];
__device__ uint64_t g_Jy[4];

__device__ __forceinline__ uint64_t to_u64_clamped(const uint64_t a[4]) {
     return a[0];
}

// Last try or i will burn this shit
__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check_sliced(
    const uint64_t* __restrict__ Px,          
    const uint64_t* __restrict__ Py,          
    uint64_t* __restrict__ Rx,                
    uint64_t* __restrict__ Ry,                
    uint64_t* __restrict__ start_scalars,     
    uint64_t* __restrict__ counts256,         
    uint64_t threadsTotal,
    uint32_t batch_size,                     
    uint32_t max_batches_per_launch,          
    int do_initial_anchor_check,              
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
    const int batch = (int)batch_size;
    if (batch <= 0 || (batch & 1)) return;
    if (batch > MAX_BATCH_SIZE) return;
    const int half = batch >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;

    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 16384u
    #define WARP_FLUSH_HASHES()                                                              \
        do {                                                                                 \
            unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes);    \
            if (lane == 0 && v) atomicAdd(hashes_accum, v);                                  \
            local_hashes = 0;                                                                \
        } while (0)
    #define MAYBE_WARP_FLUSH()                                                               \
        do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], base_scalar[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        base_scalar[i] = start_scalars[idx];
    }

    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

     if ((rem[0] | rem[1] | rem[2] | rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            Rx[gid*4+i] = x1[i];
            Ry[gid*4+i] = y1[i];
        }
        WARP_FLUSH_HASHES();
        return;
    }

    // First anchor try
    if (do_initial_anchor_check) {
        uint8_t tmp_hash[20];
        uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
        getHash160_33_from_limbs(prefix, x1, tmp_hash);
        ++local_hashes; MAYBE_WARP_FLUSH();

        bool local_pref = hash160_prefix_equals(tmp_hash, target_prefix);
        if (__any_sync(full_mask, local_pref)) {
            if (local_pref && hash160_matches_prefix_then_full(tmp_hash, c_target_hash160, target_prefix)) {
                if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                    d_found_result->threadId = (int)gid;
                    d_found_result->iter     = 0;
#pragma unroll
                    for (int k = 0; k < 4; ++k) d_found_result->scalar[k] = base_scalar[k];
#pragma unroll
                    for (int k = 0; k < 4; ++k) d_found_result->Rx[k] = x1[k];
#pragma unroll
                    for (int k = 0; k < 4; ++k) d_found_result->Ry[k] = y1[k];
                    __threadfence_system();
                    atomicExch(d_found_flag, FOUND_READY);
                }
            }
            __syncwarp(full_mask);
            WARP_FLUSH_HASHES();
            return;
        }

        sub256_u64_inplace(rem, 1ull);
        if ((rem[0] | rem[1] | rem[2] | rem[3]) == 0ull) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                Rx[gid*4+i] = x1[i];
                Ry[gid*4+i] = y1[i];
                counts256[gid*4+i] = rem[i];
                start_scalars[gid*4+i] = base_scalar[i];
            }
            WARP_FLUSH_HASHES();
            return;
        }
    }

    uint32_t batches_done = 0;

    // Main cycle
    while (batches_done < max_batches_per_launch && ((rem[0]|rem[1]|rem[2]|rem[3]) != 0ull)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        bool rem_ge_batch = ge256_u64(rem, (uint64_t)batch);

        if (rem_ge_batch) {
            uint64_t subp[MAX_BATCH_SIZE/2][4];
            uint64_t acc[4], tmp[4];

#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] = g_Jx[j];
            ModSub256(acc, acc, x1);
#pragma unroll
            for (int j = 0; j < 4; ++j) subp[half - 1][j] = acc[j];

            for (int i = half - 1; i > 0; --i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) tmp[j] = g_pGx[(size_t)i * 4 + j];
                ModSub256(tmp, tmp, x1);
                _ModMult(acc, acc, tmp);
#pragma unroll
                for (int j = 0; j < 4; ++j) subp[i - 1][j] = acc[j];
            }

            uint64_t d0[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) d0[j] = g_pGx[0 * 4 + j];
            ModSub256(d0, d0, x1);

            uint64_t inverse[5];
#pragma unroll
            for (int j = 0; j < 4; ++j) inverse[j] = d0[j];
            _ModMult(inverse, subp[0]); 
            inverse[4] = 0ULL;
            _ModInv(inverse);

            for (int i = 0; i < half - 1; ++i) {
                uint64_t dx[4];
                _ModMult(dx, subp[i], inverse); // 1/(pGx[i]-x1)

                // +Pi
                {
                    uint64_t px_i[4], py_i[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) { px_i[j] = g_pGx[(size_t)i*4 + j]; py_i[j] = g_pGy[(size_t)i*4 + j]; }

                    uint64_t lam[4], x3[4], s[4];
                    ModSub256(s, py_i, y1);
                    _ModMult(lam, s, dx);

                    _ModSqr(x3, lam);
                    ModSub256(x3, x3, x1);
                    ModSub256(x3, x3, px_i);

                    ModSub256(s, x1, x3);
                    _ModMult(s, s, lam);
                    uint8_t parityY;
                    ModSub256isOdd(s, y1, &parityY);

                    uint8_t h20[20];
                    getHash160_33_from_limbs(parityY ? 0x03 : 0x02, x3, h20);
                    ++local_hashes; MAYBE_WARP_FLUSH();

                    bool pref = hash160_prefix_equals(h20, target_prefix);
                    if (__any_sync(full_mask, pref)) {
                        if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                            if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                                d_found_result->threadId = (int)gid;
                                d_found_result->iter     = 0;

                                uint64_t fs[4];
#pragma unroll
                                for (int k=0;k<4;++k) fs[k]=base_scalar[k];
                                uint64_t carry=(uint64_t)(i+1);
#pragma unroll
                                for (int k=0;k<4 && carry;++k){ uint64_t old=fs[k]; fs[k]+=carry; carry=(fs[k]<old)?1:0; }
#pragma unroll
                                for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];

#pragma unroll
                                for (int k=0;k<4;++k) d_found_result->Rx[k]=x3[k];

                                ModSub256(s, x1, x3);
                                _ModMult(s, s, lam);
                                uint64_t y3_full[4]; ModSub256(y3_full, s, y1);
#pragma unroll
                                for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];

                                __threadfence_system();
                                atomicExch(d_found_flag, FOUND_READY);
                            }
                        }
                        __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                    }
                }

                // -Pi
                {
                    uint64_t pxn[4], pyn[4];
#pragma unroll
                    for (int j=0;j<4;++j){ pxn[j]=g_pGx[(size_t)i*4 + j]; pyn[j]=g_pGy[(size_t)i*4 + j]; }
                    ModNeg256(pyn, pyn);

                    uint64_t lam[4], x3[4], s[4];
                    ModSub256(s, pyn, y1);
                    _ModMult(lam, s, dx);
                    _ModSqr(x3, lam);
                    ModSub256(x3, x3, x1);
                    ModSub256(x3, x3, pxn);
                    ModSub256(s, x1, x3);
                    _ModMult(s, s, lam);
                    uint8_t parityY;
                    ModSub256isOdd(s, y1, &parityY);

                    uint8_t h20[20];
                    getHash160_33_from_limbs(parityY ? 0x03 : 0x02, x3, h20);
                    ++local_hashes; MAYBE_WARP_FLUSH();

                    bool pref = hash160_prefix_equals(h20, target_prefix);
                    if (__any_sync(full_mask, pref)) {
                        if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                            if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                                d_found_result->threadId = (int)gid;
                                d_found_result->iter     = 0;

                                uint64_t fs[4];
#pragma unroll
                                for (int k=0;k<4;++k) fs[k]=base_scalar[k];
                                uint64_t borrow=(uint64_t)(i+1);
#pragma unroll
                                for (int k=0;k<4 && borrow;++k){ uint64_t old=fs[k]; fs[k]=old-borrow; borrow=(old<borrow)?1:0; }
#pragma unroll
                                for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];

#pragma unroll
                                for (int k=0;k<4;++k) d_found_result->Rx[k]=x3[k];

                                ModSub256(s, x1, x3);
                                _ModMult(s, s, lam);
                                uint64_t y3_full[4]; ModSub256(y3_full, s, y1);
#pragma unroll
                                for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];

                                __threadfence_system();
                                atomicExch(d_found_flag, FOUND_READY);
                            }
                        }
                        __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                    }
                }

#pragma unroll
                for (int j = 0; j < 4; ++j) tmp[j] = g_pGx[(size_t)i*4 + j];
                ModSub256(tmp, tmp, x1);
                _ModMult(inverse, tmp);
            }

            // Last-P_{half-1}
            {
                uint64_t dx_last[4];
                _ModMult(dx_last, subp[half - 1], inverse); // 1/(pGx[half-1]-x1)

                uint64_t pxn[4], pyn[4];
#pragma unroll
                for (int j=0;j<4;++j){ pxn[j]=g_pGx[(size_t)(half-1)*4 + j]; pyn[j]=g_pGy[(size_t)(half-1)*4 + j]; }
                ModNeg256(pyn, pyn);

                uint64_t lam[4], x3[4], s[4];
                ModSub256(s, pyn, y1);
                _ModMult(lam, s, dx_last);
                _ModSqr(x3, lam);
                ModSub256(x3, x3, x1);
                ModSub256(x3, x3, pxn);
                ModSub256(s, x1, x3);
                _ModMult(s, s, lam);
                uint8_t parityY;
                ModSub256isOdd(s, y1, &parityY);

                uint8_t h20[20];
                getHash160_33_from_limbs(parityY ? 0x03 : 0x02, x3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = hash160_prefix_equals(h20, target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;

                            uint64_t fs[4];
#pragma unroll
                            for (int k=0;k<4;++k) fs[k]=base_scalar[k];
                            uint64_t borrow=(uint64_t)(half);
#pragma unroll
                            for (int k=0;k<4 && borrow;++k){ uint64_t old=fs[k]; fs[k]=old-borrow; borrow=(old<borrow)?1:0; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];

#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=x3[k];

                            ModSub256(s, x1, x3);
                            _ModMult(s, s, lam);
                            uint64_t y3_full[4]; ModSub256(y3_full, s, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];

                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            uint64_t tmp2[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp2[j] = g_pGx[(size_t)(half-1)*4 + j];
            ModSub256(tmp2, tmp2, x1);
            _ModMult(inverse, tmp2);

            uint64_t dyJ[4], lamJ[4], xJ[4], sJ[4];
            ModSub256(dyJ, g_Jy, y1);
            _ModMult(lamJ, dyJ, inverse);

            _ModSqr(xJ, lamJ);
            ModSub256(xJ, xJ, x1);
            ModSub256(xJ, xJ, g_Jx);

            ModSub256(sJ, x1, xJ);
            _ModMult(sJ, sJ, lamJ);
            ModSub256(sJ, sJ, y1);

#pragma unroll
            for (int j=0;j<4;++j){ x1[j]=xJ[j]; y1[j]=sJ[j]; }

            {
                uint64_t carry = (uint64_t)batch;
#pragma unroll
                for (int k=0;k<4 && carry;++k){ uint64_t old=base_scalar[k]; base_scalar[k]+=carry; carry=(base_scalar[k]<old)?1:0; }
            }
            sub256_u64_inplace(rem, (uint64_t)batch);
            ++batches_done;
            continue;
        }

        //Tail batc
        uint64_t rem64 = to_u64_clamped(rem);
        uint32_t active = (rem[3] | rem[2] | rem[1]) ? (uint32_t)batch
                                                     : (uint32_t)((rem64 < (uint64_t)batch) ? rem64 : (uint64_t)batch);
        if (active == 0u) break;


        const uint32_t last_idx = active; 
        uint64_t dx[MAX_BATCH_SIZE + 1][4];
        uint64_t px[MAX_BATCH_SIZE]; 
        uint64_t py[MAX_BATCH_SIZE];

        uint64_t prod[MAX_BATCH_SIZE + 1][4];

        for (uint32_t k = 0; k < active; ++k) {
#pragma unroll
            for (int j=0;j<4;++j) dx[k][j] = g_pGx[(size_t)k*4 + j];
            ModSub256(dx[k], dx[k], x1);
        }

        uint64_t Jx[4], Jy[4];
#pragma unroll
        for (int j=0;j<4;++j) { Jx[j] = g_pGx[(size_t)(active-1)*4 + j]; Jy[j] = g_pGy[(size_t)(active-1)*4 + j]; }
#pragma unroll
        for (int j=0;j<4;++j) dx[last_idx][j] = Jx[j];
        ModSub256(dx[last_idx], dx[last_idx], x1);


#pragma unroll
        for (int j=0;j<4;++j) prod[0][j] = dx[0][j];
        for (uint32_t i = 1; i <= last_idx; ++i) {
            _ModMult(prod[i], prod[i-1], dx[i]);
        }

        uint64_t inv_total[5];
#pragma unroll
        for (int j=0;j<4;++j) inv_total[j] = prod[last_idx][j];
        inv_total[4] = 0ULL;
        _ModInv(inv_total);

        uint64_t inv_dx_k[4];
        uint64_t suffix[4] = { 1ull, 0ull, 0ull, 0ull }; 

        uint64_t inv_dx_last[4];
        if (last_idx > 0) {
            uint64_t t[4];
#pragma unroll
            for (int j=0;j<4;++j) t[j] = prod[last_idx - 1][j];
            _ModMult(inv_dx_last, inv_total, t);
        } else {
            // last_idx == 0 => inv_dx_last = inv_total
#pragma unroll
            for (int j=0;j<4;++j) inv_dx_last[j] = inv_total[j];
        }


        uint64_t acc_suffix[4] = { 1ull, 0ull, 0ull, 0ull };

        _ModMult(acc_suffix, acc_suffix, dx[last_idx]);

        for (int k = (int)active - 1; k >= 0; --k) {
            uint64_t left_prod[4];
            if (k > 0) {
#pragma unroll
                for (int j=0;j<4;++j) left_prod[j] = prod[k - 1][j];
            } else {
                // 1
                left_prod[0]=1ull; left_prod[1]=0ull; left_prod[2]=0ull; left_prod[3]=0ull;
            }
            uint64_t inv_dx_k_tmp[4];
            _ModMult(inv_dx_k_tmp, inv_total, left_prod);
            _ModMult(inv_dx_k, inv_dx_k_tmp, acc_suffix);

            {
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) { px_i[j] = g_pGx[(size_t)k*4 + j]; py_i[j] = g_pGy[(size_t)k*4 + j]; }

                uint64_t lam[4], x3[4], s[4];
                ModSub256(s, py_i, y1);
                _ModMult(lam, s, inv_dx_k);

                _ModSqr(x3, lam);
                ModSub256(x3, x3, x1);
                ModSub256(x3, x3, px_i);

                ModSub256(s, x1, x3);
                _ModMult(s, s, lam);
                uint8_t parityY;
                ModSub256isOdd(s, y1, &parityY);

                uint8_t h20[20];
                getHash160_33_from_limbs(parityY ? 0x03 : 0x02, x3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = hash160_prefix_equals(h20, target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;

                            uint64_t fs[4];
#pragma unroll
                            for (int t=0;t<4;++t) fs[t]=base_scalar[t];
                            uint64_t addv = (uint64_t)(k + 1);
#pragma unroll
                            for (int t=0;t<4 && addv;++t){ uint64_t old=fs[t]; uint64_t sum=old+addv; fs[t]=sum; addv=(sum<old)?1ull:0ull; }
#pragma unroll
                            for (int t=0;t<4;++t) d_found_result->scalar[t]=fs[t];

#pragma unroll
                            for (int t=0;t<4;++t) d_found_result->Rx[t]=x3[t];
                            uint64_t y3_full[4]; ModSub256(y3_full, s, y1);
#pragma unroll
                            for (int t=0;t<4;++t) d_found_result->Ry[t]=y3_full[t];

                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            _ModMult(acc_suffix, acc_suffix, dx[k]);
        }

        {
            uint64_t dyJ[4], lamJ[4], xJ2[4], sJ2[4];
            ModSub256(dyJ, Jy, y1);
            _ModMult(lamJ, dyJ, inv_dx_last);

            _ModSqr(xJ2, lamJ);
            ModSub256(xJ2, xJ2, x1);
            ModSub256(xJ2, xJ2, Jx);

            ModSub256(sJ2, x1, xJ2);
            _ModMult(sJ2, sJ2, lamJ);
            ModSub256(sJ2, sJ2, y1);

#pragma unroll
            for (int j=0;j<4;++j){ x1[j]=xJ2[j]; y1[j]=sJ2[j]; }
        }

        {
            uint64_t carry = (uint64_t)active;
#pragma unroll
            for (int k=0;k<4 && carry;++k){ uint64_t old=base_scalar[k]; base_scalar[k]+=carry; carry=(base_scalar[k]<old)?1:0; }
        }
        sub256_u64_inplace(rem, (uint64_t)active);
        ++batches_done;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = base_scalar[i];
    }

    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();

    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

static inline bool is_zero_256_host(const uint64_t a[4]) {
    return (a[0]|a[1]|a[2]|a[3]) == 0ull;
}

static void divmod_256_by_u64_safe(const uint64_t a[4], uint64_t d,
                                   uint64_t q[4], uint64_t &r)
{
    unsigned __int128 rem = 0;
    uint64_t t[4] = { a[3], a[2], a[1], a[0] }; 

    uint64_t q_be[4];
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 cur = (rem << 64) | t[i];
        uint64_t qword = (uint64_t)(cur / d);
        rem = (cur % d);
        q_be[i] = qword;
    }
    q[0] = q_be[3];
    q[1] = q_be[2];
    q[2] = q_be[1];
    q[3] = q_be[0];
    r = (uint64_t)rem;
}

int main(int argc, char** argv) {
    std::string target_hash_hex, range_hex;
    std::string address_b58;
    bool grid_provided = false;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm    = 8;   
    uint32_t slices_per_launch         = 64; 

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        char* endp = nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10);
        if (*endp != '\0') return false;
        endp = nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10);
        if (*endp != '\0') return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out = (uint32_t)aa;
        b_out = (uint32_t)bb;
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--target-hash160" && i + 1 < argc) target_hash_hex = argv[++i];
        else if (arg == "--address"        && i + 1 < argc) address_b58     = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
            grid_provided = true;
        }
        else if (arg == "--slices" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<20)) {
                std::cerr << "Error: --slices must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [--grid A,B] [--slices N]\n";
        return EXIT_FAILURE;
    }
    if (!target_hash_hex.empty() && !address_b58.empty()) {
        std::cerr << "Error: provide either --address or --target-hash160, not both.\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58, target_hash160)) {
            std::cerr << "Error: invalid P2PKH address (Base58Check failed or wrong version)\n";
            return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: Invalid target hash160 hex\n"; return EXIT_FAILURE;
        }
    }

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << ".\n";
        return EXIT_FAILURE;
    }
    if (slices_per_launch == 0) {
        std::cerr << "Error: slices_per_launch must be > 0\n";
        return EXIT_FAILURE;
    }

    // range_len = end - start + 1
    uint64_t range_len[4];
    sub256(range_end, range_start, range_len);
    add256_u64(range_len, 1ull, range_len);

    auto is_zero_256 = [](const uint64_t a[4])->bool {
        return (a[0]|a[1]|a[2]|a[3]) == 0ull;
    };
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if (is_zero_256(a)) return false;
        uint64_t am1[4];
        uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = a[i] - borrow;
            borrow = (a[i] < borrow) ? 1ull : 0ull;
            am1[i] = v;
            if (borrow == 0ull && i+1<4) { for (int k=i+1;k<4;++k) am1[k] = a[k]; break; }
        }
        uint64_t andv0 = a[0] & am1[0];
        uint64_t andv1 = a[1] & am1[1];
        uint64_t andv2 = a[2] & am1[2];
        uint64_t andv3 = a[3] & am1[3];
        return (andv0|andv1|andv2|andv3) == 0ull;
    };
    if (!is_power_of_two_256(range_len)) {
        std::cerr << "Error: range length (end - start + 1) must be a power of two.\n";
        return EXIT_FAILURE;
    }
    uint64_t len_minus1[4];
    {
        uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = range_len[i] - borrow;
            borrow = (range_len[i] < borrow) ? 1ull : 0ull;
            len_minus1[i] = v;
            if (borrow == 0ull && i+1<4) { for (int k=i+1;k<4;++k) len_minus1[k] = range_len[k]; break; }
        }
    }
    {
        uint64_t and0 = range_start[0] & len_minus1[0];
        uint64_t and1 = range_start[1] & len_minus1[1];
        uint64_t and2 = range_start[2] & len_minus1[2];
        uint64_t and3 = range_start[3] & len_minus1[3];
        if ((and0|and1|and2|and3) != 0ull) {
            std::cerr << "Error: start must be aligned to the range length (power-of-two aligned).\n";
            return EXIT_FAILURE;
        }
    }

    int device = 0;
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) { std::cerr << "cudaGetDevice error\n"; return EXIT_FAILURE; }
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) { std::cerr << "cudaGetDeviceProperties error\n"; return EXIT_FAILURE; }

    size_t stackSize = 64 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);

    int threadsPerBlock = 256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    const uint64_t bytesPerThread = 2 * 4 * sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    uint64_t q_div_batch[4], r_div_batch = 0;
    divmod_256_by_u64_safe(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull) {
        std::cerr << "Error: range length must be divisible by batch size (" << runtime_points_batch_size << ").\n";
        return EXIT_FAILURE;
    }
    bool q_fits_u64 = (q_div_batch[3] | q_div_batch[2] | q_div_batch[1]) == 0ull;
    uint64_t q_u64  = q_fits_u64 ? q_div_batch[0] : UINT64_MAX;

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    auto pick_threads_total = [&](uint64_t upper)->uint64_t {
        if (upper < (uint64_t)threadsPerBlock) return 0ull;
        uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
        if (!q_fits_u64) return t;
        uint64_t q = q_u64;
        while (t >= (uint64_t)threadsPerBlock) {
            if ((q % t) == 0ull) return t;
            t -= (uint64_t)threadsPerBlock;
        }
        return 0ull;
    };

    uint64_t upper = maxThreadsByMem;
    if (q_fits_u64 && q_u64 < upper) upper = q_u64;
    if (userUpper   < upper)         upper = userUpper;

    uint64_t threadsTotal = pick_threads_total(upper);
    if (threadsTotal == 0ull) {
        std::cerr << "Error: failed to pick threadsTotal satisfying divisibility.\n";
        return EXIT_FAILURE;
    }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    uint64_t q256[4]; uint64_t r_u64 = 0;
    divmod_256_by_u64_safe(range_len, threadsTotal, q256, r_u64);
    if (r_u64 != 0ull) {
        std::cerr << "Internal error: range_len not divisible by threadsTotal after alignment.\n";
        return EXIT_FAILURE;
    }
    {
        uint64_t qq[4], rr = 0;
        divmod_256_by_u64_safe(q256, (uint64_t)runtime_points_batch_size, qq, rr);
        if (rr != 0ull) {
            std::cerr << "Internal error: per-thread count is not a multiple of batch size.\n";
            return EXIT_FAILURE;
        }
    }

    uint64_t* h_counts256     = new uint64_t[threadsTotal * 4];
    uint64_t* h_start_scalars = new uint64_t[threadsTotal * 4];

    for (uint64_t i = 0; i < threadsTotal; ++i) {
        h_counts256[i*4+0] = q256[0];
        h_counts256[i*4+1] = q256[1];
        h_counts256[i*4+2] = q256[2];
        h_counts256[i*4+3] = q256[3];
    }
    {
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            h_start_scalars[i*4+0] = cur[0];
            h_start_scalars[i*4+1] = cur[1];
            h_start_scalars[i*4+2] = cur[2];
            h_start_scalars[i*4+3] = cur[3];
            uint64_t next[4];
            add256(cur, &h_counts256[i*4], next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    }

    {
        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    }

    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr, *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    int *d_found_flag=nullptr;
    FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr;
    unsigned int *d_any_left=nullptr;

    cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Px, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Py, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Rx, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ry, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_counts256, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_found_result, sizeof(FoundResult));
    cudaMalloc(&d_hashes_accum, sizeof(unsigned long long));
    cudaMalloc(&d_any_left, sizeof(unsigned int));

    cudaMemcpy(d_start_scalars, h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts256,     h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    {
        int zero = FOUND_NONE;
        cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
        unsigned long long zero64 = 0ull;
        cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }

    {
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        cudaDeviceSynchronize();
    }

    {
        const uint32_t B = runtime_points_batch_size;

        uint64_t *d_pGx=nullptr, *d_pGy=nullptr;
        cudaMalloc(&d_pGx, (size_t)B * 4 * sizeof(uint64_t));
        cudaMalloc(&d_pGy, (size_t)B * 4 * sizeof(uint64_t));

        uint64_t* h_scal = new uint64_t[(size_t)B * 4];
        std::memset(h_scal, 0, (size_t)B * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < B; ++k) h_scal[(size_t)k*4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_pG_scalars=nullptr;
        cudaMalloc(&d_pG_scalars, (size_t)B * 4 * sizeof(uint64_t));
        cudaMemcpy(d_pG_scalars, h_scal, (size_t)B * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

        int blocks_scal = (int)((B + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_pG_scalars, d_pGx, d_pGy, (int)B);
        cudaDeviceSynchronize();

        cudaMemcpyToSymbol(g_pGx, d_pGx, (size_t)B * 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(g_pGy, d_pGy, (size_t)B * 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);

        // J = B*G
        cudaMemcpyToSymbol(g_Jx, d_pGx + (size_t)(B - 1) * 4, 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(g_Jy, d_pGy + (size_t)(B - 1) * 4, 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);

        cudaFree(d_pG_scalars);
        delete[] h_scal;
        cudaFree(d_pGx);
        cudaFree(d_pGy);
    }

    size_t freeB=0,totalB=0;
    cudaMemGetInfo(&freeB,&totalB);
    size_t usedB = totalB - freeB;
    double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "SM"                << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(20) << "ThreadsPerBlock"   << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(20) << "Blocks"            << " : " << blocks << "\n";
    std::cout << std::left << std::setw(20) << "Points batch size" << " : " << runtime_points_batch_size << "\n";
    std::cout << std::left << std::setw(20) << "Batches/SM"        << " : " << runtime_batches_per_sm << "\n";
    std::cout << std::left << std::setw(20) << "Batches/launch"    << " : " << slices_per_launch << " (per thread)\n";
    std::cout << std::left << std::setw(20) << "Memory utilization"<< " : "
              << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)usedB) << " / " << human_bytes((double)totalB) << ")\n";
    std::cout << "------------------------------------------------------- \n";
    std::cout << std::left << std::setw(20) << "Total threads"     << " : " << threadsTotal << "\n\n";

    std::cout << "======== Phase-1: BruteForce (sliced) =================\n";

    cudaStream_t streamKernel;
    cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    size_t sharedBytes = 0;

    bool first_launch = true;
    bool stop_all = false;

    while (!stop_all) {
        unsigned int zeroU = 0u;
        cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, streamKernel);

        kernel_point_add_and_check_sliced<<<blocks, threadsPerBlock, sharedBytes, streamKernel>>>(
            d_Px, d_Py, d_Rx, d_Ry,
            d_start_scalars,
            d_counts256,
            threadsTotal,
            runtime_points_batch_size,
            slices_per_launch,
            first_launch ? 1 : 0,
            d_found_flag, d_found_result,
            d_hashes_accum,
            d_any_left
        );
        cudaGetLastError();

        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 1.0) {
                unsigned long long h_hashes = 0ull;
                cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                long double total_keys_ld = ld_from_u256(range_len);
                long double prog = total_keys_ld > 0.0L ? ((long double)h_hashes / total_keys_ld) * 100.0L : 0.0L;
                if (prog > 100.0L) prog = 100.0L;
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
                          << " Mkeys/s | Count: " << h_hashes
                          << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %";
                std::cout.flush();
                lastHashes = h_hashes;
                tLast = now;
            }

            int host_found = 0;
            cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_found == FOUND_READY) { stop_all = true; break; }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) break;
            else if (qs != cudaErrorNotReady) { cudaGetLastError(); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        cudaStreamSynchronize(streamKernel);
        std::cout.flush();

        if (stop_all) break;

        unsigned int h_any = 0u;
        cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        std::swap(d_Px, d_Rx);
        std::swap(d_Py, d_Ry);

        first_launch = false;

        if (h_any == 0u) {
            break;
        }
    }

    cudaDeviceSynchronize();
    std::cout << "\n";

    int h_found_flag = 0;
    cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
        std::cout << "\n";
        std::cout << "======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
    }

    cudaFree(d_start_scalars);
    cudaFree(d_Px);
    cudaFree(d_Py);
    cudaFree(d_Rx);
    cudaFree(d_Ry);
    cudaFree(d_counts256);
    cudaFree(d_found_flag);
    cudaFree(d_found_result);
    cudaFree(d_hashes_accum);
    cudaFree(d_any_left);
    cudaStreamDestroy(streamKernel);

    delete[] h_start_scalars;
    delete[] h_counts256;

    return 0;
}
