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
#define MAX_BATCH_SIZE 512 
#endif 
#ifndef WARP_SIZE 
#define WARP_SIZE 32 
#endif 

__constant__ uint64_t c_pGx[MAX_BATCH_SIZE * 4]; 
__constant__ uint64_t c_pGy[MAX_BATCH_SIZE * 4];

__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    const uint64_t* __restrict__ start_scalars,
    const uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,                   
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum
)
{
    const int batch = (int)batch_size;
    if (batch <= 0 || (batch & 1)) return;    
    if (batch > MAX_BATCH_SIZE) return;        
    const int half  = batch >> 1;

    extern __shared__ uint64_t s_mem[];
    uint64_t* s_pGx = s_mem;
    uint64_t* s_pGy = s_pGx + (size_t)batch * 4;

    const int total_limbs = batch * 4;
    for (int idx = threadIdx.x; idx < total_limbs; idx += blockDim.x) {
        s_pGx[idx] = c_pGx[idx];
        s_pGy[idx] = c_pGy[idx];
    }
    __syncthreads(); 

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
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES();
        return;
    }

    {
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
    }

    sub256_u64_inplace(rem, 1ull);

    while (ge256_u64(rem, (uint64_t)batch)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        uint64_t subp[MAX_BATCH_SIZE/2][4]; 
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = s_pGx[(size_t)(batch - 1) * 4 + j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j = 0; j < 4; ++j) subp[half - 1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = s_pGx[(size_t)(i + 1) * 4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j = 0; j < 4; ++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) d0[j] = s_pGx[0 * 4 + j];
        ModSub256(d0, d0, x1);

        uint64_t inverse[5];
#pragma unroll
        for (int j = 0; j < 4; ++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ULL;
        _ModInv(inverse);

        for (int i = 0; i < half; ++i) {
            uint64_t dx[4];
            _ModMult(dx, subp[i], inverse);

            {
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) { px_i[j] = s_pGx[(size_t)i*4 + j]; py_i[j] = s_pGy[(size_t)i*4 + j]; }

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

            {
                uint64_t pxn[4], pyn[4];
#pragma unroll
                for (int j=0;j<4;++j){ pxn[j]=s_pGx[(size_t)i*4 + j]; pyn[j]=s_pGy[(size_t)i*4 + j]; }
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
            for (int j = 0; j < 4; ++j) tmp[j] = s_pGx[(size_t)i*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(inverse, tmp);
        }

        {
            uint64_t px_last[4], py_last[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { px_last[j]=s_pGx[(size_t)(batch-1)*4 + j]; py_last[j]=s_pGy[(size_t)(batch-1)*4 + j]; }

            uint64_t lam[4], x3[4], s[4];
            ModSub256(s, py_last, y1);
            _ModMult(lam, s, inverse);

            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            ModSub256(x3, x3, px_last);

            ModSub256(s, x1, x3);
            _ModMult(s, s, lam);
            ModSub256(s, s, y1);

#pragma unroll
            for (int j=0;j<4;++j){ x1[j]=x3[j]; y1[j]=s[j]; }
        }

        {
            uint64_t carry = (uint64_t)batch;
#pragma unroll
            for (int k=0;k<4 && carry;++k){ uint64_t old=base_scalar[k]; base_scalar[k]+=carry; carry=(base_scalar[k]<old)?1:0; }
        }

        sub256_u64_inplace(rem, (uint64_t)batch);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) { Rx[gid*4+i]=x1[i]; Ry[gid*4+i]=y1[i]; }

    WARP_FLUSH_HASHES();

    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

int main(int argc, char** argv) {
    std::string target_hash_hex, range_hex;
    std::string address_b58;
    bool grid_provided = false;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm    = 8;

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
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [--grid A,B]\n";
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
    if (runtime_points_batch_size > 512u) {
        std::cerr << "Error: batch size must be <= 512 (kernel limit).\n";
        return EXIT_FAILURE;
    }

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

    int threadsPerBlock = 256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    const uint64_t bytesPerThread = 2 * 4 * sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    uint64_t q_div_batch[4], r_div_batch = 0;
    divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
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
    divmod_256_by_u64(range_len, threadsTotal, q256, r_u64);
    if (r_u64 != 0ull) {
        std::cerr << "Internal error: range_len not divisible by threadsTotal after alignment.\n";
        return EXIT_FAILURE;
    }
    {
        uint64_t qq[4], rr = 0;
        divmod_256_by_u64(q256, (uint64_t)runtime_points_batch_size, qq, rr);
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

    cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Px, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Py, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Rx, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ry, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_counts256, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_found_result, sizeof(FoundResult));
    cudaMalloc(&d_hashes_accum, sizeof(unsigned long long));

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

        cudaMemcpyToSymbol(c_pGx, d_pGx, (size_t)B * 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(c_pGy, d_pGy, (size_t)B * 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);

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
    std::cout << std::left << std::setw(20) << "Memory utilization"<< " : "
              << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)usedB) << " / " << human_bytes((double)totalB) << ")\n";
    std::cout << "------------------------------------------------------- \n";
    std::cout << std::left << std::setw(20) << "Total threads"     << " : " << threadsTotal << "\n\n";

    std::cout << "======== Phase-1: Brooteforce =========================\n";

    cudaStream_t streamKernel;
    cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking);

    cudaFuncSetCacheConfig(kernel_point_add_and_check, cudaFuncCachePreferShared);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    size_t sharedBytes = (size_t)runtime_points_batch_size * 4 * sizeof(uint64_t) * 2; // pGx + pGy

    kernel_point_add_and_check<<<blocks, threadsPerBlock, sharedBytes, streamKernel>>>(
        d_Px, d_Py, d_Rx, d_Ry,
        d_start_scalars,
        d_counts256,
        threadsTotal,
        runtime_points_batch_size,
        d_found_flag, d_found_result,
        d_hashes_accum
    );
    cudaGetLastError();

    long double total_keys_ld = ld_from_u256(range_len);
    bool kernel_done = false;
    while (!kernel_done) {
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            unsigned long long h_hashes = 0ull;
            cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            double delta = (double)(h_hashes - lastHashes);
            double mkeys = delta / (dt * 1e6);
            double elapsed = std::chrono::duration<double>(now - t0).count();
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
        if (host_found == FOUND_READY) break;

        cudaError_t qs = cudaStreamQuery(streamKernel);
        if (qs == cudaSuccess) kernel_done = true;
        else if (qs != cudaErrorNotReady) { cudaGetLastError(); break; }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
    cudaStreamDestroy(streamKernel);

    delete[] h_start_scalars;
    delete[] h_counts256;

    return 0;
}
