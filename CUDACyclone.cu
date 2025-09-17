// CUDACyclone.cu — фикс первого батча: anchor + (B-1), затем всегда по B ключей.
// Предрасчёт 1..B·G выполняется один раз и хранится в девайс-памяти.
// Полный батч: ровно B проверок (± вокруг якоря, включая +P(half-1)).
// Хвост: только +Pk, k=0..active-1, ровно active проверок.
// Перенос якоря: на B·G (для полного) или на active·G (для хвоста).

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

#include "CUDAMath.h"         // ModSub256/_ModMult/_ModSqr/_ModInv/...
#include "CUDAHash.cuh"       // getHash160_33_from_limbs, c_target_prefix/c_target_hash160
#include "CUDAUtils.h"        // eq/ge/sub etc. (не дублируем)
#include "sha256.h"        // eq/ge/sub etc. (не дублируем)
#include "CUDAStructures.h"   // FoundResult, FOUND_*

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ===== DEBUG: проверка количества хешей в батче/хвосте (gid==0) =====
#define DEBUG_BATCH_CHECK 0

// ===== Глобальные device-символы для предрасчитанных точек =====
__device__ uint64_t g_pGx[MAX_BATCH_SIZE * 4];
__device__ uint64_t g_pGy[MAX_BATCH_SIZE * 4];

// Утилиты
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

// Объявление вашего ядра умножения скалярного (уже реализовано в проекте)
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

// Деление 256 на u64 (безопасно для нашего представления)
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

// ===== Ядро: 1 запуск = 1 батч (± для полного, +Pk для хвоста) =====
// ВАЖНО: если do_initial_anchor_check != 0, то:
//  - проверяем якорь (1 ключ),
//  - затем считаем актив = min(B-1, rem) в хвостовой ветке (+Pk),
//  - итого за первый запуск ровно B ключей.
// Далее запуск всегда считает ровно B ключей (полный батч).
__launch_bounds__(256, 2)
__global__ void kernel_one_batch_pm_both(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    int do_initial_anchor_check,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum
)
{
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;

    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    // состояние потока
    uint64_t X[4], Y[4], S[4], rem[4];
#pragma unroll
    for (int i=0;i<4;++i) {
        uint64_t idx = gid*4 + i;
        X[i]   = Px[idx];
        Y[i]   = Py[idx];
        S[i]   = start_scalars[idx];
        rem[i] = counts256[idx];
    }

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
        for (int i=0;i<4;++i) {
            Rx[gid*4+i] = X[i];
            Ry[gid*4+i] = Y[i];
        }
        return;
    }

    unsigned int local_hashes = 0;

#if DEBUG_BATCH_CHECK
    unsigned tested = 0;
#endif

    // 1) ЯКОРЬ (только при первом запуске): съедаем 1 ключ
    bool initial = (do_initial_anchor_check != 0);
    if (initial) {
        // hash(anchor)
        uint8_t h20[20];
        uint8_t odd = (uint8_t)(Y[0] & 1ULL) ? 0x03 : 0x02;
        getHash160_33_from_limbs(odd, X, h20);
        ++local_hashes;
#if DEBUG_BATCH_CHECK
        ++tested;
#endif
        bool pref = (*(const uint32_t*)h20 == c_target_prefix);
        if (__any_sync(full_mask, pref)) {
            if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, c_target_prefix)) {
                if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
#pragma unroll
                    for (int i=0;i<4;++i) { d_found_result->scalar[i]=S[i]; d_found_result->Rx[i]=X[i]; d_found_result->Ry[i]=Y[i]; }
                    d_found_result->threadId = (int)gid;
                    d_found_result->iter = 0;
                    __threadfence_system();
                    atomicExch(d_found_flag, FOUND_READY);
                }
            }
            atomicAdd(hashes_accum, (unsigned long long)local_hashes);
            return;
        }
        // rem -= 1; S += 1
        sub256_u64_inplace(rem, 1ull);
        if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
            for (int i=0;i<4;++i){ Rx[gid*4+i]=X[i]; Ry[gid*4+i]=Y[i]; counts256[gid*4+i]=rem[i]; start_scalars[gid*4+i]=S[i]; }
            atomicAdd(hashes_accum, (unsigned long long)local_hashes);
            return;
        }
        inc256_device(S, 1ull);
    }

    // 2) Определим, сколько ключей надо проверить в ЭТОМ запуске
    //    - если initial: осталось вместить только (B-1)
    //    - иначе: целый батч B
    uint32_t effective_B = initial ? (uint32_t)(B - 1) : (uint32_t)B;

    // Для корректного покрытия делаем: если effective_B == B и rem >= B → полный батч (±),
    // иначе — хвост (active = min(effective_B, rem)).
    bool can_full_batch = (!initial) && ge256_u64(rem, (uint64_t)B);

    if (can_full_batch) {
        // ===== ПОЛНЫЙ БАТЧ: ровно B точек (± вокруг якоря, включая +P(half-1)) =====
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

        // subp построение: acc = (pGx[half-1] - X)
#pragma unroll
        for (int j=0;j<4;++j) acc[j] = g_pGx[(size_t)(half-1)*4 + j];
        ModSub256(acc, acc, X);
#pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half-1; i > 0; --i) {
#pragma unroll
            for (int j=0;j<4;++j) tmp[j] = g_pGx[(size_t)i*4 + j];
            ModSub256(tmp, tmp, X);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j=0;j<4;++j) subp[i-1][j] = acc[j];
        }

        // inverse = 1 / ∏_{k=0..half-1}(pGx[k]-X)
        uint64_t inverse[5];
#pragma unroll
        for (int j=0;j<4;++j) inverse[j] = g_pGx[0*4 + j];
        ModSub256(inverse, inverse, X);
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull; _ModInv(inverse);

        // i=0..half-2 → +Pi и -Pi (2*(half-1) точек)
        for (int i=0; i<half-1; ++i) {
            uint64_t dx[4]; _ModMult(dx, subp[i], inverse);

            // +Pi
            {
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=g_pGx[(size_t)i*4 + j]; py_i[j]=g_pGy[(size_t)i*4 + j]; }
                uint64_t dy[4], lam[4], x3[4], s[4];
                ModSub256(dy, py_i, Y);
                _ModMult(lam, dy, dx);
                _ModSqr(x3, lam);
                ModSub256(x3, x3, X);
                ModSub256(x3, x3, px_i);
                ModSub256(s, X, x3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, Y, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, x3, h20);
                ++local_hashes;
#if DEBUG_BATCH_CHECK
                ++tested;
#endif
                bool pref = (*(const uint32_t*)h20 == c_target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, c_target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4];
#pragma unroll
                            for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t addv=(uint64_t)(i+1);
#pragma unroll
                            for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k){ d_found_result->scalar[k]=fs[k]; d_found_result->Rx[k]=x3[k]; }
                            uint64_t y3_full[4]; ModSub256(y3_full, s, Y);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];
                            d_found_result->threadId=(int)gid; d_found_result->iter=0;
                            __threadfence_system(); atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    atomicAdd(hashes_accum, (unsigned long long)local_hashes);
                    return;
                }
            }

            // -Pi
            {
                uint64_t pxn[4], pyn[4];
#pragma unroll
                for (int j=0;j<4;++j) { pxn[j]=g_pGx[(size_t)i*4 + j]; pyn[j]=g_pGy[(size_t)i*4 + j]; }
                ModNeg256(pyn, pyn);

                uint64_t dy[4], lam[4], x3[4], s[4];
                ModSub256(dy, pyn, Y);
                _ModMult(lam, dy, dx);
                _ModSqr(x3, lam);
                ModSub256(x3, x3, X);
                ModSub256(x3, x3, pxn);
                ModSub256(s, X, x3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, Y, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, x3, h20);
                ++local_hashes;
#if DEBUG_BATCH_CHECK
                ++tested;
#endif
                bool pref = (*(const uint32_t*)h20 == c_target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, c_target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4];
#pragma unroll
                            for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t sub=(uint64_t)(i+1);
#pragma unroll
                            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k){ d_found_result->scalar[k]=fs[k]; d_found_result->Rx[k]=x3[k]; }
                            uint64_t y3_full[4]; ModSub256(y3_full, s, Y);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];
                            d_found_result->threadId=(int)gid; d_found_result->iter=0;
                            __threadfence_system(); atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    atomicAdd(hashes_accum, (unsigned long long)local_hashes);
                    return;
                }
            }

            uint64_t tmpdx[4];
#pragma unroll
            for (int j=0;j<4;++j) tmpdx[j]=g_pGx[(size_t)i*4 + j];
            ModSub256(tmpdx, tmpdx, X);
            _ModMult(inverse, tmpdx);
        }

        // i=half-1: сначала -P(half-1)
        {
            int i = half-1;
            uint64_t dx[4]; _ModMult(dx, subp[i], inverse);

            uint64_t pxn[4], pyn[4];
#pragma unroll
            for (int j=0;j<4;++j) { pxn[j]=g_pGx[(size_t)i*4 + j]; pyn[j]=g_pGy[(size_t)i*4 + j]; }
            ModNeg256(pyn, pyn);

            uint64_t dy[4], lam[4], x3[4], s[4];
            ModSub256(dy, pyn, Y);
            _ModMult(lam, dy, dx);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, X);
            ModSub256(x3, x3, pxn);
            ModSub256(s, X, x3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, Y, &odd);

            uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, x3, h20);
            ++local_hashes;
#if DEBUG_BATCH_CHECK
            ++tested;
#endif
            bool pref = (*(const uint32_t*)h20 == c_target_prefix);
            if (__any_sync(full_mask, pref)) {
                if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, c_target_prefix)) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4];
#pragma unroll
                        for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t sub=(uint64_t)(half);
#pragma unroll
                        for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                        for (int k=0;k<4;++k){ d_found_result->scalar[k]=fs[k]; d_found_result->Rx[k]=x3[k]; }
                        uint64_t y3_full[4]; ModSub256(y3_full, s, Y);
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];
                        d_found_result->threadId=(int)gid; d_found_result->iter=0;
                        __threadfence_system(); atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                atomicAdd(hashes_accum, (unsigned long long)local_hashes);
                return;
            }
        }

        // затем +P(half-1) — ОБЯЗАТЕЛЬНО, чтобы получить ровно B проверок
        {
            int i = half-1;
            uint64_t dx[4]; _ModMult(dx, subp[i], inverse);

            uint64_t px_i[4], py_i[4];
#pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=g_pGx[(size_t)i*4 + j]; py_i[j]=g_pGy[(size_t)i*4 + j]; }

            uint64_t dy[4], lam[4], x3[4], s[4];
            ModSub256(dy, py_i, Y);
            _ModMult(lam, dy, dx);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, X);
            ModSub256(x3, x3, px_i);
            ModSub256(s, X, x3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, Y, &odd);

            uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, x3, h20);
            ++local_hashes;
#if DEBUG_BATCH_CHECK
            ++tested;
#endif
            bool pref = (*(const uint32_t*)h20 == c_target_prefix);
            if (__any_sync(full_mask, pref)) {
                if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, c_target_prefix)) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4];
#pragma unroll
                        for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t addv=(uint64_t)(half);
#pragma unroll
                        for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
#pragma unroll
                        for (int k=0;k<4;++k){ d_found_result->scalar[k]=fs[k]; d_found_result->Rx[k]=x3[k]; }
                        uint64_t y3_full[4]; ModSub256(y3_full, s, Y);
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];
                        d_found_result->threadId=(int)gid; d_found_result->iter=0;
                        __threadfence_system(); atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                atomicAdd(hashes_accum, (unsigned long long)local_hashes);
                return;
            }
        }

#if DEBUG_BATCH_CHECK
        if (gid == 0) printf("DEBUG full-batch tested=%u expected=%d\n", tested, B);
        tested = 0;
#endif

        // перенос якоря на B*G
        {
            uint64_t Jx[4], Jy[4];
#pragma unroll
            for (int j=0;j<4;++j) { Jx[j]=g_pGx[(size_t)(B-1)*4 + j]; Jy[j]=g_pGy[(size_t)(B-1)*4 + j]; }
            uint64_t dxJ[4], dyJ[4], lamJ[4], xJ[4], sJ[4];
            ModSub256(dxJ, Jx, X);
            uint64_t invJ[5]; for (int j=0;j<4;++j) invJ[j]=dxJ[j]; invJ[4]=0ull; _ModInv(invJ);
            ModSub256(dyJ, Jy, Y);
            _ModMult(lamJ, dyJ, invJ);
            _ModSqr(xJ, lamJ);
            ModSub256(xJ, xJ, X);
            ModSub256(xJ, xJ, Jx);
            ModSub256(sJ, X, xJ);
            _ModMult(sJ, sJ, lamJ);
            ModSub256(sJ, sJ, Y);
#pragma unroll
            for (int j=0;j<4;++j){ X[j]=xJ[j]; Y[j]=sJ[j]; }
        }

        // S += B; rem -= B
        {
            uint64_t addv=(uint64_t)B;
#pragma unroll
            for (int i=0;i<4 && addv;++i) { uint64_t old=S[i]; S[i]=old+addv; addv=(S[i]<old)?1ull:0ull; }
        }
        sub256_u64_inplace(rem, (uint64_t)B);

    } else {
        // ===== ХВОСТ или «первый запуск после anchor»: ровно effective_B точек (+Pk) =====
        // active = min(effective_B, rem)
        uint64_t rem64 = rem[0];
        uint32_t active = (rem[3]|rem[2]|rem[1]) ? effective_B
                                                 : (uint32_t)((rem64 < (uint64_t)effective_B) ? rem64 : (uint64_t)effective_B);
        if (active == 0u) {
#pragma unroll
            for (int i=0;i<4;++i){ Rx[gid*4+i]=X[i]; Ry[gid*4+i]=Y[i]; counts256[gid*4+i]=rem[i]; start_scalars[gid*4+i]=S[i]; }
            atomicAdd(hashes_accum, (unsigned long long)local_hashes);
            return;
        }

        // dx[0..active-1]=(pGx[k]-X), dx[last_idx]= (B*G).x - X
        uint64_t last_idx = active;
        uint64_t dx[MAX_BATCH_SIZE + 1][4];
        uint64_t prod[MAX_BATCH_SIZE + 1][4];

        for (uint32_t k=0;k<active;++k) {
#pragma unroll
            for (int j=0;j<4;++j) dx[k][j] = g_pGx[(size_t)k*4 + j];
            ModSub256(dx[k], dx[k], X);
        }
        uint64_t Jx[4], Jy[4];
#pragma unroll
        for (int j=0;j<4;++j) { Jx[j]=g_pGx[(size_t)(B-1)*4 + j]; Jy[j]=g_pGy[(size_t)(B-1)*4 + j]; }
#pragma unroll
        for (int j=0;j<4;++j) dx[last_idx][j] = Jx[j];
        ModSub256(dx[last_idx], dx[last_idx], X);

        // prod
#pragma unroll
        for (int j=0;j<4;++j) prod[0][j] = dx[0][j];
        for (uint32_t i=1;i<=last_idx;++i) { _ModMult(prod[i], prod[i-1], dx[i]); }

        uint64_t inv_total[5]; for (int j=0;j<4;++j) inv_total[j]=prod[last_idx][j]; inv_total[4]=0ull; _ModInv(inv_total);

        uint64_t inv_dx_last[4];
        if (last_idx > 0) {
            uint64_t t[4]; for (int j=0;j<4;++j) t[j]=prod[last_idx-1][j];
            _ModMult(inv_dx_last, inv_total, t);
        } else {
            for (int j=0;j<4;++j) inv_dx_last[j]=inv_total[j];
        }

        uint64_t acc_suffix[4] = { 1ull, 0ull, 0ull, 0ull };
        _ModMult(acc_suffix, acc_suffix, dx[last_idx]);

        // k=active-1..0: только +Pk
        for (int k=(int)active-1; k>=0; --k) {
            uint64_t left_prod[4];
            if (k>0) { for (int j=0;j<4;++j) left_prod[j]=prod[k-1][j]; }
            else { left_prod[0]=1ull; left_prod[1]=0ull; left_prod[2]=0ull; left_prod[3]=0ull; }

            uint64_t inv_dx_k[4], tmp[4];
            _ModMult(tmp, inv_total, left_prod);
            _ModMult(inv_dx_k, tmp, acc_suffix);

            // +Pk
            {
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=g_pGx[(size_t)k*4 + j]; py_i[j]=g_pGy[(size_t)k*4 + j]; }
                uint64_t dy[4], lam[4], x3[4], s[4];
                ModSub256(dy, py_i, Y);
                _ModMult(lam, dy, inv_dx_k);
                _ModSqr(x3, lam);
                ModSub256(x3, x3, X);
                ModSub256(x3, x3, px_i);
                ModSub256(s, X, x3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, Y, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, x3, h20);
                ++local_hashes;
#if DEBUG_BATCH_CHECK
                ++tested;
#endif
                bool pref = (*(const uint32_t*)h20 == c_target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, c_target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4];
#pragma unroll
                            for (int t=0;t<4;++t) fs[t]=S[t];
                            uint64_t addv=(uint64_t)(k+1);
#pragma unroll
                            for (int t=0;t<4 && addv;++t){ uint64_t old=fs[t]; fs[t]=old+addv; addv=(fs[t]<old)?1ull:0ull; }
#pragma unroll
                            for (int t=0;t<4;++t){ d_found_result->scalar[t]=fs[t]; d_found_result->Rx[t]=x3[t]; }
                            uint64_t y3_full[4]; ModSub256(y3_full, s, Y);
#pragma unroll
                            for (int t=0;t<4;++t) d_found_result->Ry[t]=y3_full[t];
                            d_found_result->threadId=(int)gid; d_found_result->iter=0;
                            __threadfence_system(); atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    atomicAdd(hashes_accum, (unsigned long long)local_hashes);
                    return;
                }
            }

            _ModMult(acc_suffix, acc_suffix, dx[k]);
        }

#if DEBUG_BATCH_CHECK
        if (gid == 0) printf("DEBUG tail tested=%u expected=%u\n", tested, active);
        tested = 0;
#endif

        // перенос якоря на active*G
        {
            uint64_t GxA[4], GyA[4];
#pragma unroll
            for (int j=0;j<4;++j) { GxA[j]=g_pGx[(size_t)(active-1)*4 + j]; GyA[j]=g_pGy[(size_t)(active-1)*4 + j]; }
            uint64_t dxA[4], dyA[4], lamA[4], xA[4], sA[4];
            ModSub256(dxA, GxA, X);
            uint64_t invA[5]; for (int j=0;j<4;++j) invA[j]=dxA[j]; invA[4]=0ull; _ModInv(invA);
            ModSub256(dyA, GyA, Y);
            _ModMult(lamA, dyA, invA);
            _ModSqr(xA, lamA);
            ModSub256(xA, xA, X);
            ModSub256(xA, xA, GxA);
            ModSub256(sA, X, xA);
            _ModMult(sA, sA, lamA);
            ModSub256(sA, sA, Y);
#pragma unroll
            for (int j=0;j<4;++j){ X[j]=xA[j]; Y[j]=sA[j]; }
        }

        // S += active; rem -= active
        {
            uint64_t addv=(uint64_t)active;
#pragma unroll
            for (int i=0;i<4 && addv;++i) { uint64_t old=S[i]; S[i]=old+addv; addv=(S[i]<old)?1ull:0ull; }
        }
        sub256_u64_inplace(rem, (uint64_t)active);
    }

    // write back
#pragma unroll
    for (int i=0;i<4;++i) {
        Rx[gid*4+i] = X[i];
        Ry[gid*4+i] = Y[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    atomicAdd(hashes_accum, (unsigned long long)local_hashes);
}

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);

int main(int argc, char** argv) {
    std::string target_hash_hex, range_hex, address_b58;
    uint32_t runtime_points_batch_size = 512;
    uint32_t runtime_batches_per_sm    = 256;
    bool grid_provided=false;

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

    for (int i=1;i<argc;++i) {
        std::string arg = argv[i];
        if      (arg == "--target-hash160" && i + 1 < argc) target_hash_hex = argv[++i];
        else if (arg == "--address"        && i + 1 < argc) address_b58     = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) { std::cerr<<"Error: --grid expects \"A,B\".\n"; return EXIT_FAILURE; }
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
        std::cerr << "Error: provide either --address or --target-hash160, not both.\n"; return EXIT_FAILURE;
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

    // range_len = end - start + 1
    uint64_t range_len[4]; sub256(range_end, range_start, range_len); add256_u64(range_len, 1ull, range_len);

    // проверки степени двойки и выравнивания старта
    auto is_zero_256 = [](const uint64_t a[4])->bool { return (a[0]|a[1]|a[2]|a[3])==0ull; };
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if (is_zero_256(a)) return false;
        uint64_t am1[4]; uint64_t borrow=1ull;
        for (int i=0;i<4;++i){ uint64_t v=a[i]-borrow; borrow=(a[i]<borrow)?1ull:0ull; am1[i]=v; if (!borrow && i+1<4){ for(int k=i+1;k<4;++k) am1[k]=a[k]; break; } }
        uint64_t and0=a[0]&am1[0],and1=a[1]&am1[1],and2=a[2]&am1[2],and3=a[3]&am1[3];
        return (and0|and1|and2|and3)==0ull;
    };
    if (!is_power_of_two_256(range_len)) { std::cerr<<"Error: range length must be a power of two.\n"; return EXIT_FAILURE; }
    uint64_t len_minus1[4]; { uint64_t borrow=1ull; for (int i=0;i<4;++i){ uint64_t v=range_len[i]-borrow; borrow=(range_len[i]<borrow)?1ull:0ull; len_minus1[i]=v; if(!borrow && i+1<4){ for(int k=i+1;k<4;++k) len_minus1[k]=range_len[k]; break; } } }
    { uint64_t and0=range_start[0]&len_minus1[0],and1=range_start[1]&len_minus1[1],and2=range_start[2]&len_minus1[2],and3=range_start[3]&len_minus1[3];
      if ((and0|and1|and2|and3)!=0ull) { std::cerr<<"Error: start must be aligned to the range length.\n"; return EXIT_FAILURE; } }

    // CUDA init
    int device=0; cudaDeviceProp prop{}; if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }
    int threadsPerBlock=256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock=prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock=32;

    // подбор threadsTotal
    uint64_t bytesPerThread = 2ull*4ull*sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    uint64_t reserveBytes = 64ull*1024ull*1024ull;
    uint64_t usableMem = (totalGlobalMem>reserveBytes) ? (totalGlobalMem-reserveBytes) : (totalGlobalMem/2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    uint64_t q_div_batch[4], r_div_batch=0ull;
    divmod_256_by_u64_safe(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull) { std::cerr<<"Error: range length must be divisible by batch size.\n"; return EXIT_FAILURE; }

    bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1])==0ull;
    uint64_t q_u64  = q_fits_u64 ? q_div_batch[0] : UINT64_MAX;
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
    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;
    if (q_fits_u64 && q_u64 < upper) upper = q_u64;
    if (userUpper   < upper)         upper = userUpper;

    uint64_t threadsTotal = pick_threads_total(upper);
    if (threadsTotal == 0ull) { std::cerr<<"Error: failed to pick threadsTotal.\n"; return EXIT_FAILURE; }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    // per-thread count
    uint64_t per_thread_count[4], r_u64=0ull;
    divmod_256_by_u64_safe(range_len, threadsTotal, per_thread_count, r_u64);
    if (r_u64 != 0ull) { std::cerr<<"Internal error: per_thread_count not integer.\n"; return EXIT_FAILURE; }

    // Хост-буферы
    uint64_t* h_counts256     = new uint64_t[threadsTotal * 4];
    uint64_t* h_start_scalars = new uint64_t[threadsTotal * 4];
    {
        for (uint64_t i=0;i<threadsTotal;++i) {
            h_counts256[i*4+0]=per_thread_count[0];
            h_counts256[i*4+1]=per_thread_count[1];
            h_counts256[i*4+2]=per_thread_count[2];
            h_counts256[i*4+3]=per_thread_count[3];
        }
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (uint64_t i=0;i<threadsTotal;++i) {
            h_start_scalars[i*4+0]=cur[0];
            h_start_scalars[i*4+1]=cur[1];
            h_start_scalars[i*4+2]=cur[2];
            h_start_scalars[i*4+3]=cur[3];
            uint64_t next[4]; add256(cur, per_thread_count, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    }

    // target prefix/hash
    {
        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    }

    // Device buffers
    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr, *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    int *d_found_flag=nullptr; FoundResult *d_found_result=nullptr; unsigned long long *d_hashes_accum=nullptr;
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
        int zero = FOUND_NONE; unsigned long long zero64=0ull;
        cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }

    // Начальные XY = S*G
    {
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        cudaDeviceSynchronize();
    }

    // Предрасчёт 1..B·G → g_pGx/g_pGy
    {
        const uint32_t B = runtime_points_batch_size;
        uint64_t *d_pGx=nullptr, *d_pGy=nullptr; cudaMalloc(&d_pGx, (size_t)B * 4 * sizeof(uint64_t)); cudaMalloc(&d_pGy, (size_t)B * 4 * sizeof(uint64_t));
        uint64_t* h_scal = (uint64_t*)malloc((size_t)B * 4 * sizeof(uint64_t));
        std::memset(h_scal, 0, (size_t)B * 4 * sizeof(uint64_t));
        for (uint32_t k=0;k<B;++k) h_scal[(size_t)k*4 + 0] = (uint64_t)(k + 1);
        uint64_t *d_pG_scalars=nullptr; cudaMalloc(&d_pG_scalars, (size_t)B * 4 * sizeof(uint64_t));
        cudaMemcpy(d_pG_scalars, h_scal, (size_t)B * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        int blocks_scal = (int)((B + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_pG_scalars, d_pGx, d_pGy, (int)B);
        cudaDeviceSynchronize();
        cudaMemcpyToSymbol(g_pGx, d_pGx, (size_t)B * 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(g_pGy, d_pGy, (size_t)B * 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);
        cudaFree(d_pG_scalars); free(h_scal); cudaFree(d_pGx); cudaFree(d_pGy);
    }

    // Info
    size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
    double util = totalB ? (double)(totalB-freeB)*100.0/(double)totalB : 0.0;
    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "SM"                << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(20) << "ThreadsPerBlock"   << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(20) << "Blocks"            << " : " << blocks << "\n";
    std::cout << std::left << std::setw(20) << "Points batch size" << " : " << runtime_points_batch_size << "\n";
    std::cout << std::left << std::setw(20) << "Memory utilization"<< " : " << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)(totalB-freeB)) << " / " << human_bytes((double)totalB) << ")\n";
    std::cout << "------------------------------------------------------- \n";
    std::cout << std::left << std::setw(20) << "Total threads"     << " : " << threadsTotal << "\n\n";
    std::cout << "======== Phase-1: BruteForce (1 batch / launch, ±, 1 inv) =====\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0; unsigned long long lastHashes=0ull;
    bool stop_all=false; bool first_launch = true;

    while (!stop_all) {
        kernel_one_batch_pm_both<<<blocks, threadsPerBlock>>>(
            d_Px, d_Py, d_Rx, d_Ry,
            d_start_scalars, d_counts256,
            threadsTotal,
            runtime_points_batch_size,
            first_launch ? 1 : 0,
            d_found_flag, d_found_result,
            d_hashes_accum
        );
        cudaError_t ke = cudaGetLastError();
        if (ke != cudaSuccess) { std::cerr << "Kernel error: " << cudaGetErrorString(ke) << "\n"; return EXIT_FAILURE; }
        cudaDeviceSynchronize();

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            unsigned long long h_hashes = 0ull; cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
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
            lastHashes = h_hashes; tLast = now;
        }

        int h_found=0; cudaMemcpy(&h_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_found == FOUND_READY) stop_all = true;

        std::swap(d_Px, d_Rx); std::swap(d_Py, d_Ry);
        first_launch = false;

        if (stop_all) break;
        // (Останов по исчерпанию диапазона можно реализовать анализом counts256; опущено для краткости.)
    }
    std::cout << "\n";

    int h_found_flag=0; cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{}; cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
    }

    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py); cudaFree(d_Rx); cudaFree(d_Ry);
    cudaFree(d_counts256); cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum);
    delete[] h_start_scalars; delete[] h_counts256;

    return 0;
}

