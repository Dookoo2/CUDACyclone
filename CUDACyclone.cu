
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
#include <csignal>
#include <atomic>
#include <mutex>
#include <vector>
#include <array>
#include <random>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1536
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check_oneinv(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];
    }
    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            bool pref = hash160_prefix_equals(h20, target_prefix);
            if (__any_sync(full_mask, pref)) {
                if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=S[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=x1[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }
        }

        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
#pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
#pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        uint64_t sy_neg[4], sx_neg[4];
        ModNeg256(sy_neg, y1);
        ModNeg256(sx_neg, x1);

        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = hash160_prefix_equals(h20, target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t addv=(uint64_t)(i+1);
                            for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];

                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i, py_i);

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = hash160_prefix_equals(h20, target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t sub=(uint64_t)(i+1);
                            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            uint64_t gxmi[4];
#pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4];
            uint64_t px_i[4], py_i[4];
#pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
            ModNeg256(py_i, py_i);

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, px_i);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, y1, &odd);

            uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            bool pref = hash160_prefix_equals(h20, target_prefix);
            if (__any_sync(full_mask, pref)) {
                if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t sub=(uint64_t)half;
                        for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                        uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }

            uint64_t last_dx[4];
#pragma unroll
            for (int j=0;j<4;++j) last_dx[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(last_dx, last_dx, x1);
            _ModMult(inverse, inverse, last_dx);
        }

        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        {
            uint64_t addv=(uint64_t)B;
            for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old+addv; addv=(S[k]<old)?1ull:0ull; }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        ++batches_done;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern bool decode_p2pkh_address(const std::string& addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

// ── Shared state between GPU threads ──────────────────────────────────────────
struct GpuShared {
    std::atomic<int>                any_found{0};
    std::mutex                      result_mtx;
    FoundResult                     best_result{};
    bool                            has_result{false};
    std::atomic<unsigned long long> total_hashes{0};
    std::atomic<unsigned long long> chunks_tried{0};
    std::atomic<int>                gpus_exhausted{0};
    std::atomic<int>                init_done{0};
};

static std::mutex g_print_mutex;

// ── Per-GPU worker ────────────────────────────────────────────────────────────
static void run_on_gpu(
    int            gpu_id,
    const uint64_t range_start[4],
    const uint64_t range_end[4],
    const uint8_t  target_hash160[20],
    uint32_t       runtime_points_batch_size,
    uint32_t       runtime_batches_per_sm,
    uint32_t       slices_per_launch,
    bool           random_mode,
    GpuShared&     shared
) {
    cudaSetDevice(gpu_id);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    auto ck = [&](cudaError_t e, const char* msg) {
        if (e != cudaSuccess) {
            std::lock_guard<std::mutex> lk(g_print_mutex);
            fprintf(stderr, "\n[GPU %d] %s: %s\n", gpu_id, msg, cudaGetErrorString(e));
            std::exit(EXIT_FAILURE);
        }
    };

    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, gpu_id), "cudaGetDeviceProperties");

    int threadsPerBlock = 256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    uint64_t gpu_range_len[4];
    sub256(range_end, range_start, gpu_range_len);
    add256_u64(gpu_range_len, 1ull, gpu_range_len);

    const uint64_t bytesPerThread = 2ull * 4ull * sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    uint64_t q_div_batch[4]; uint64_t r_batch = 0ull;
    divmod_256_by_u64(gpu_range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_batch);
    if (r_batch != 0ull) {
        uint64_t adjust = (uint64_t)runtime_points_batch_size - r_batch;
        add256_u64(gpu_range_len, adjust, gpu_range_len);
        divmod_256_by_u64(gpu_range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_batch);
    }
    if ((q_div_batch[3] | q_div_batch[2] | q_div_batch[1]) != 0ull) {
        std::lock_guard<std::mutex> lk(g_print_mutex);
        fprintf(stderr, "[GPU %d] Error: range too large.\n", gpu_id);
        std::exit(EXIT_FAILURE);
    }
    uint64_t total_batches_u64 = q_div_batch[0];

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    uint64_t desired_upper = maxThreadsByMem;
    if (userUpper < desired_upper) desired_upper = userUpper;
    uint64_t threadsTotal = (desired_upper / (uint64_t)threadsPerBlock) * (uint64_t)threadsPerBlock;
    if (threadsTotal < (uint64_t)threadsPerBlock) threadsTotal = (uint64_t)threadsPerBlock;
    if (total_batches_u64 < threadsTotal) {
        threadsTotal = (total_batches_u64 / (uint64_t)threadsPerBlock) * (uint64_t)threadsPerBlock;
        if (threadsTotal < (uint64_t)threadsPerBlock) threadsTotal = (uint64_t)threadsPerBlock;
    }
    if ((total_batches_u64 % threadsTotal) != 0ull) {
        uint64_t rem = total_batches_u64 % threadsTotal;
        total_batches_u64 += threadsTotal - rem;
        add256_u64(gpu_range_len, (threadsTotal - rem) * (uint64_t)runtime_points_batch_size, gpu_range_len);
    }

    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    uint64_t per_thread_cnt[4]; uint64_t r_u64 = 0ull;
    if (random_mode) {
        // Each kernel launch covers exactly one fixed-size chunk per thread
        per_thread_cnt[0] = (uint64_t)runtime_points_batch_size * slices_per_launch;
        per_thread_cnt[1] = per_thread_cnt[2] = per_thread_cnt[3] = 0ull;
    } else {
        divmod_256_by_u64(gpu_range_len, threadsTotal, per_thread_cnt, r_u64);
    }

    const uint32_t B    = runtime_points_batch_size;
    const uint32_t half = B >> 1;

    // Target constants (per-device constant memory)
    {
        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        ck(cudaMemcpyToSymbol(c_target_prefix,  &prefix_le,    sizeof(prefix_le)), "ToSymbol c_target_prefix");
        ck(cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20),               "ToSymbol c_target_hash160");
    }

    // Host buffers (plain malloc — no cudaHostAlloc needed for one-time upload)
    std::vector<uint64_t> h_counts256(threadsTotal * 4);
    std::vector<uint64_t> h_start_scalars(threadsTotal * 4);

    for (uint64_t i = 0; i < threadsTotal; ++i) {
        h_counts256[i*4+0] = per_thread_cnt[0];
        h_counts256[i*4+1] = per_thread_cnt[1];
        h_counts256[i*4+2] = per_thread_cnt[2];
        h_counts256[i*4+3] = per_thread_cnt[3];
    }
    {
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t Sc[4]; add256_u64(cur, (uint64_t)half, Sc);
            h_start_scalars[i*4+0] = Sc[0];
            h_start_scalars[i*4+1] = Sc[1];
            h_start_scalars[i*4+2] = Sc[2];
            h_start_scalars[i*4+3] = Sc[3];
            uint64_t next[4]; add256(cur, per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    }

    // Device buffers
    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr,
             *d_Rx=nullptr,           *d_Ry=nullptr, *d_counts256=nullptr;
    int            *d_found_flag   = nullptr;
    FoundResult    *d_found_result = nullptr;
    unsigned long long *d_hashes_accum = nullptr;
    unsigned int       *d_any_left     = nullptr;

    ck(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_start_scalars)");
    ck(cudaMalloc(&d_Px,            threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Px)");
    ck(cudaMalloc(&d_Py,            threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Py)");
    ck(cudaMalloc(&d_Rx,            threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Rx)");
    ck(cudaMalloc(&d_Ry,            threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Ry)");
    ck(cudaMalloc(&d_counts256,     threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_counts256)");
    ck(cudaMalloc(&d_found_flag,    sizeof(int)),                         "cudaMalloc(d_found_flag)");
    ck(cudaMalloc(&d_found_result,  sizeof(FoundResult)),                 "cudaMalloc(d_found_result)");
    ck(cudaMalloc(&d_hashes_accum,  sizeof(unsigned long long)),          "cudaMalloc(d_hashes_accum)");
    ck(cudaMalloc(&d_any_left,      sizeof(unsigned int)),                "cudaMalloc(d_any_left)");

    ck(cudaMemcpy(d_start_scalars, h_start_scalars.data(), threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start_scalars");
    ck(cudaMemcpy(d_counts256,     h_counts256.data(),     threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts256");
    { int z = FOUND_NONE; unsigned long long z64 = 0ull;
      ck(cudaMemcpy(d_found_flag,   &z,   sizeof(int),                cudaMemcpyHostToDevice), "init found_flag");
      ck(cudaMemcpy(d_hashes_accum, &z64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes_accum"); }

    // Precompute initial EC points
    {
        int bs = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<bs, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
        ck(cudaGetLastError(),      "scalarMulKernelBase launch");
    }

    // Precompute G*1..G*half → constant memory c_Gx / c_Gy
    {
        std::vector<uint64_t> h_scalars_half(half * 4, 0);
        for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4] = (uint64_t)(k + 1);

        uint64_t *d_sh=nullptr, *d_Gxh=nullptr, *d_Gyh=nullptr;
        ck(cudaMalloc(&d_sh,  (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_sh)");
        ck(cudaMalloc(&d_Gxh, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gxh)");
        ck(cudaMalloc(&d_Gyh, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gyh)");
        ck(cudaMemcpy(d_sh, h_scalars_half.data(), (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy half scalars");

        int bs = (int)((half + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<bs, threadsPerBlock>>>(d_sh, d_Gxh, d_Gyh, (int)half);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(half) sync");
        ck(cudaGetLastError(),      "scalarMulKernelBase(half) launch");

        std::vector<uint64_t> h_Gxh(half * 4), h_Gyh(half * 4);
        ck(cudaMemcpy(h_Gxh.data(), d_Gxh, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx_half");
        ck(cudaMemcpy(h_Gyh.data(), d_Gyh, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy_half");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gxh.data(), (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gyh.data(), (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

        cudaFree(d_sh); cudaFree(d_Gxh); cudaFree(d_Gyh);
    }

    // Precompute jump point J = G*B → constant memory c_Jx / c_Jy
    {
        std::vector<uint64_t> h_scB(4, 0); h_scB[0] = (uint64_t)B;
        uint64_t *d_scB=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
        ck(cudaMalloc(&d_scB, 4 * sizeof(uint64_t)), "cudaMalloc(d_scB)");
        ck(cudaMalloc(&d_Jx,  4 * sizeof(uint64_t)), "cudaMalloc(d_Jx)");
        ck(cudaMalloc(&d_Jy,  4 * sizeof(uint64_t)), "cudaMalloc(d_Jy)");
        ck(cudaMemcpy(d_scB, h_scB.data(), 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scB");

        scalarMulKernelBase<<<1, 1>>>(d_scB, d_Jx, d_Jy, 1);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(B) sync");
        ck(cudaGetLastError(),      "scalarMulKernelBase(B) launch");

        uint64_t hJx[4], hJy[4];
        ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
        ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
        ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");

        cudaFree(d_scB); cudaFree(d_Jx); cudaFree(d_Jy);
    }

    // Print GPU info block
    {
        size_t freeB=0, totalB=0; cudaMemGetInfo(&freeB, &totalB);
        double util = totalB ? (double)(totalB - freeB) * 100.0 / (double)totalB : 0.0;

        std::lock_guard<std::mutex> lk(g_print_mutex);
        std::cout << "======== GPU " << gpu_id << " : " << prop.name
                  << " (compute " << prop.major << "." << prop.minor << ") ========\n";
        std::cout << std::left << std::setw(20) << "SM"                 << " : " << prop.multiProcessorCount << "\n";
        std::cout << std::left << std::setw(20) << "ThreadsPerBlock"    << " : " << threadsPerBlock << "\n";
        std::cout << std::left << std::setw(20) << "Blocks"             << " : " << blocks << "\n";
        std::cout << std::left << std::setw(20) << "Total threads"      << " : " << threadsTotal << "\n";
        std::cout << std::left << std::setw(20) << "Points batch size"  << " : " << B << "\n";
        std::cout << std::left << std::setw(20) << "Batches/SM"         << " : " << runtime_batches_per_sm << "\n";
        std::cout << std::left << std::setw(20) << "Batches/launch"     << " : " << slices_per_launch << " (per thread)\n";
        std::cout << std::left << std::setw(20) << "Memory utilization" << " : "
                  << std::fixed << std::setprecision(1) << util << "% ("
                  << human_bytes((double)(totalB - freeB)) << " / " << human_bytes((double)totalB) << ")\n";
        std::cout << "------------------------------------------------------- \n";
        std::cout.flush();
    }

    // Signal init complete
    shared.init_done.fetch_add(1, std::memory_order_release);

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");
    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv, cudaFuncCachePreferL1);

    unsigned long long last_hashes_gpu = 0ull;
    bool stop_all    = false;
    bool completed_all = false;

    // Random mode: range for chunk selection and RNG
    uint64_t full_range_len[4];
    sub256(range_end, range_start, full_range_len);
    add256_u64(full_range_len, 1ull, full_range_len);
    // chunk_span = how many keys each random chunk covers (threadsTotal * per_thread_cnt[0])
    // per_thread_cnt[0] = B * slices in random mode, fits comfortably in uint64_t
    uint64_t chunk_span = (uint64_t)threadsTotal * per_thread_cnt[0];

    std::mt19937_64 rng_state(
        (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count()
        ^ ((uint64_t)gpu_id * 0x9e3779b97f4a7c15ULL)
    );

    // Fills chunk_start with a random position in [range_start, range_end - chunk_span]
    auto pick_random_start = [&](uint64_t chunk_start[4]) {
        // Effective range for chunk selection: range_len - chunk_span
        // Use __uint128_t; safe for any range up to 2^128
        __uint128_t rl = ((__uint128_t)full_range_len[1] << 64) | full_range_len[0];
        if (rl > (__uint128_t)chunk_span) rl -= (__uint128_t)chunk_span;
        else rl = 1;
        __uint128_t r = ((__uint128_t)rng_state() << 64) | rng_state();
        __uint128_t off = r % rl;
        uint64_t offset[4] = {(uint64_t)off, (uint64_t)(off >> 64), 0, 0};
        add256(range_start, offset, chunk_start);
    };

    // Refills h_start_scalars / h_counts256 and reinits EC points for a new chunk
    auto reinit_chunk = [&](const uint64_t chunk_start[4]) {
        uint64_t cur[4] = {chunk_start[0], chunk_start[1], chunk_start[2], chunk_start[3]};
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t Sc[4]; add256_u64(cur, (uint64_t)half, Sc);
            h_start_scalars[i*4+0] = Sc[0]; h_start_scalars[i*4+1] = Sc[1];
            h_start_scalars[i*4+2] = Sc[2]; h_start_scalars[i*4+3] = Sc[3];
            uint64_t next[4]; add256(cur, per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            h_counts256[i*4+0] = per_thread_cnt[0];
            h_counts256[i*4+1] = per_thread_cnt[1];
            h_counts256[i*4+2] = per_thread_cnt[2];
            h_counts256[i*4+3] = per_thread_cnt[3];
        }
        cudaMemcpy(d_start_scalars, h_start_scalars.data(),
                   threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_counts256,     h_counts256.data(),
                   threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        int bs = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<bs, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        cudaDeviceSynchronize();
    };

    while (!stop_all) {
        if (shared.any_found.load(std::memory_order_relaxed)) break;
        if (g_sigint) break;

        // Random mode: always pick a new random position before each kernel launch
        if (random_mode) {
            uint64_t chunk_start[4];
            pick_random_start(chunk_start);
            reinit_chunk(chunk_start);
            shared.chunks_tried.fetch_add(1, std::memory_order_relaxed);
            if (shared.any_found.load(std::memory_order_relaxed)) break;
            if (g_sigint) break;
        }

        unsigned int zeroU = 0u;
        ck(cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, streamKernel), "zero d_any_left");

        kernel_point_add_and_check_oneinv<<<blocks, threadsPerBlock, 0, streamKernel>>>(
            d_Px, d_Py, d_Rx, d_Ry,
            d_start_scalars, d_counts256,
            threadsTotal, B, slices_per_launch,
            d_found_flag, d_found_result,
            d_hashes_accum, d_any_left
        );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::lock_guard<std::mutex> lk(g_print_mutex);
            fprintf(stderr, "\n[GPU %d] Kernel launch error: %s\n", gpu_id, cudaGetErrorString(launchErr));
            stop_all = true;
            break;
        }

        // Poll until kernel finishes
        while (!stop_all) {
            if (shared.any_found.load(std::memory_order_relaxed)) {
                // Another GPU found the key — signal our kernel to exit early
                int ready = FOUND_READY;
                cudaMemcpy(d_found_flag, &ready, sizeof(int), cudaMemcpyHostToDevice);
                stop_all = true;
                break;
            }
            if (g_sigint) { stop_all = true; break; }

            // Accumulate hash count from this GPU into shared counter
            unsigned long long h_hashes = 0ull;
            cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            if (h_hashes > last_hashes_gpu) {
                shared.total_hashes.fetch_add(h_hashes - last_hashes_gpu, std::memory_order_relaxed);
                last_hashes_gpu = h_hashes;
            }

            int host_found = 0;
            cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_found == FOUND_READY) {
                FoundResult res{};
                cudaMemcpy(&res, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
                {
                    std::lock_guard<std::mutex> lk(shared.result_mtx);
                    if (!shared.has_result) {
                        shared.best_result = res;
                        shared.has_result  = true;
                    }
                }
                shared.any_found.store(1, std::memory_order_release);
                stop_all = true;
                break;
            }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess)           break;
            if (qs != cudaErrorNotReady) { cudaGetLastError(); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        cudaStreamSynchronize(streamKernel);

        // Final hash flush after sync (memory now fully visible)
        {
            unsigned long long h_hashes = 0ull;
            cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            if (h_hashes > last_hashes_gpu) {
                shared.total_hashes.fetch_add(h_hashes - last_hashes_gpu, std::memory_order_relaxed);
                last_hashes_gpu = h_hashes;
            }

            // Re-check found flag after sync — catches the race where cudaStreamQuery
            // returned success before the polling loop read FOUND_READY
            if (!stop_all && !shared.any_found.load(std::memory_order_relaxed)) {
                int host_found = 0;
                cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
                if (host_found == FOUND_READY) {
                    FoundResult res{};
                    cudaMemcpy(&res, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
                    {
                        std::lock_guard<std::mutex> lk(shared.result_mtx);
                        if (!shared.has_result) {
                            shared.best_result = res;
                            shared.has_result  = true;
                        }
                    }
                    shared.any_found.store(1, std::memory_order_release);
                    stop_all = true;
                }
            }
        }

        if (stop_all || g_sigint) break;

        unsigned int h_any = 0u;
        cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (random_mode) {
            // Chunk done — loop back to pick a new random position (no swap)
        } else {
            std::swap(d_Px, d_Rx);
            std::swap(d_Py, d_Ry);
            if (h_any == 0u) { completed_all = true; break; }
        }
    }

    cudaDeviceSynchronize();

    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py);
    cudaFree(d_Rx); cudaFree(d_Ry); cudaFree(d_counts256);
    cudaFree(d_found_flag); cudaFree(d_found_result);
    cudaFree(d_hashes_accum); cudaFree(d_any_left);
    cudaStreamDestroy(streamKernel);

    if (completed_all)
        shared.gpus_exhausted.fetch_add(1, std::memory_order_relaxed);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string target_hash_hex, range_hex, address_b58;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;
    bool     random_mode               = false;

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
        char* endp=nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp) return false;
        endp=nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp) return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
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
        else if (arg == "--random") {
            random_mode = true;
        }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [--grid A,B] [--slices N] [--random]\n";
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
            std::cerr << "Error: invalid P2PKH address\n"; return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n"; return EXIT_FAILURE;
        }
    }

    if (runtime_points_batch_size < 2 || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be at least 2 and even.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (constant memory limit).\n";
        return EXIT_FAILURE;
    }

    // Detect GPUs
    int num_gpus = 0;
    if (cudaGetDeviceCount(&num_gpus) != cudaSuccess || num_gpus == 0) {
        std::cerr << "No CUDA-capable GPUs found.\n";
        return EXIT_FAILURE;
    }

    // Full range length (for progress display)
    uint64_t range_len[4];
    sub256(range_end, range_start, range_len);
    add256_u64(range_len, 1ull, range_len);

    // In random mode every GPU searches the full range independently.
    // In sequential mode split evenly across GPUs.
    std::vector<std::array<uint64_t,4>> gpu_starts(num_gpus), gpu_ends(num_gpus);
    if (random_mode) {
        for (int g = 0; g < num_gpus; ++g) {
            gpu_starts[g] = { range_start[0], range_start[1], range_start[2], range_start[3] };
            gpu_ends[g]   = { range_end[0],   range_end[1],   range_end[2],   range_end[3]   };
        }
    } else {
    uint64_t per_gpu_len[4]; uint64_t r_gpu = 0ull;
    divmod_256_by_u64(range_len, (uint64_t)num_gpus, per_gpu_len, r_gpu);

    {
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (int g = 0; g < num_gpus; ++g) {
            gpu_starts[g] = { cur[0], cur[1], cur[2], cur[3] };
            if (g == num_gpus - 1) {
                gpu_ends[g] = { range_end[0], range_end[1], range_end[2], range_end[3] };
            } else {
                uint64_t next[4]; add256(cur, per_gpu_len, next);
                uint64_t one[4] = {1,0,0,0};
                uint64_t end[4]; sub256(next, one, end);
                gpu_ends[g] = { end[0], end[1], end[2], end[3] };
                cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
            }
        }
    }
    } // end else (sequential range split)

    std::cout << "======== PrePhase: GPU Information (" << num_gpus
              << " GPU" << (num_gpus > 1 ? "s" : "") << ") ===\n";
    for (int g = 0; g < num_gpus; ++g) {
        cudaDeviceProp p{}; cudaGetDeviceProperties(&p, g);
        std::cout << "  GPU " << g << " : " << p.name
                  << "  |  " << p.multiProcessorCount << " SMs"
                  << "  |  " << human_bytes((double)p.totalGlobalMem) << "\n";
    }
    std::cout << "======================================================= \n\n";
    std::cout.flush();

    GpuShared shared;
    std::atomic<int> gpus_running{num_gpus};

    std::vector<std::thread> gpu_threads;
    gpu_threads.reserve(num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
        gpu_threads.emplace_back([&, g]() {
            run_on_gpu(g,
                       gpu_starts[g].data(), gpu_ends[g].data(),
                       target_hash160,
                       runtime_points_batch_size, runtime_batches_per_sm, slices_per_launch,
                       random_mode,
                       shared);
            gpus_running.fetch_sub(1, std::memory_order_relaxed);
        });
    }

    // Wait for all GPUs to finish init before starting display
    while (shared.init_done.load(std::memory_order_acquire) < num_gpus) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (g_sigint) break;
    }

    std::cout << "\n======== Phase-1: " << (random_mode ? "Lottery / Random Jump" : "BruteForce") << " ("
              << num_gpus << " GPU" << (num_gpus > 1 ? "s" : "") << ") =====\n";
    if (random_mode) {
        uint64_t ck = (uint64_t)runtime_points_batch_size * slices_per_launch;
        std::string ck_s;
        if      (ck >= 1000000000ULL) ck_s = std::to_string(ck/1000000000ULL) + "G";
        else if (ck >= 1000000ULL)    ck_s = std::to_string(ck/1000000ULL)    + "M";
        else if (ck >= 1000ULL)       ck_s = std::to_string(ck/1000ULL)       + "K";
        else                          ck_s = std::to_string(ck);
        std::cout << "(random mode: ~" << ck_s
                  << " keys/thread per chunk; lower --slices = more frequent jumps)\n";
    }
    std::cout.flush();

    auto t0    = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;
    long double total_keys_ld = ld_from_u256(range_len);

    while (gpus_running.load(std::memory_order_relaxed) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            unsigned long long h_hashes = shared.total_hashes.load(std::memory_order_relaxed);
            double delta  = (double)(h_hashes - lastHashes);
            double mkeys  = delta / (dt * 1e6);
            double elapsed = std::chrono::duration<double>(now - t0).count();
            long double prog = total_keys_ld > 0.0L
                               ? ((long double)h_hashes / total_keys_ld) * 100.0L : 0.0L;
            if (prog > 100.0L) prog = 100.0L;

            double speed_val = mkeys;
            const char* speed_unit = "Mkeys/s";
            if (speed_val >= 1000000.0) { speed_val /= 1000000.0; speed_unit = "Tkeys/s"; }
            else if (speed_val >= 1000.0) { speed_val /= 1000.0;  speed_unit = "Gkeys/s"; }

            if (random_mode) {
                unsigned long long chunks = shared.chunks_tried.load(std::memory_order_relaxed);
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << std::setw(6) << elapsed
                          << " s | Speed: " << std::fixed << std::setprecision(2) << std::setw(7) << speed_val
                          << " " << speed_unit << " | Count: " << std::setw(14) << h_hashes
                          << " | Chunks: " << std::setw(6) << chunks << "   ";
            } else {
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << std::setw(6) << elapsed
                          << " s | Speed: " << std::fixed << std::setprecision(2) << std::setw(7) << speed_val
                          << " " << speed_unit << " | Count: " << std::setw(14) << h_hashes
                          << " | Progress: " << std::fixed << std::setprecision(2) << std::setw(6) << (double)prog << " %   ";
            }
            std::cout.flush();
            lastHashes = h_hashes; tLast = now;
        }

        if (g_sigint) break;
    }

    for (auto& t : gpu_threads) t.join();

    std::cout << "\n";

    int exit_code = EXIT_SUCCESS;

    if (shared.has_result) {
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(shared.best_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(shared.best_result.Rx, shared.best_result.Ry) << "\n";
    } else if (g_sigint) {
        std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
        std::cout << "Search was interrupted by user. Partial progress above.\n";
        exit_code = 130;
    } else if (shared.gpus_exhausted.load() >= num_gpus) {
        std::cout << "======== KEY NOT FOUND (exhaustive) ===================\n";
        std::cout << "Target hash160 was not found within the specified range.\n";
    } else {
        std::cout << "======== TERMINATED ===================================\n";
    }

    return exit_code;
}
