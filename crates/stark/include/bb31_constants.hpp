#pragma once

#include "bb31_t.hpp" 

#include <cuda_runtime.h>

constexpr const bb31_t HOST_TWO_ADIC_GENERATORS[28] = {
    bb31_t(0x1), bb31_t(0x78000000), bb31_t(0x67055c21), bb31_t(0x5ee99486),
    bb31_t(0xbb4c4e4), bb31_t(0x2d4cc4da), bb31_t(0x669d6090), bb31_t(0x17b56c64),
    bb31_t(0x67456167), bb31_t(0x688442f9), bb31_t(0x145e952d), bb31_t(0x4fe61226),
    bb31_t(0x4c734715), bb31_t(0x11c33e2a), bb31_t(0x62c3d2b1), bb31_t(0x77cad399),
    bb31_t(0x54c131f4), bb31_t(0x4cabd6a6), bb31_t(0x5cf5713f), bb31_t(0x3e9430e8),
    bb31_t(0xba067a3), bb31_t(0x18adc27d), bb31_t(0x21fd55bc), bb31_t(0x4b859b3d),
    bb31_t(0x3bd57996), bb31_t(0x4483d85a), bb31_t(0x3a26eef8), bb31_t(0x1a427a41)
};

__constant__ bb31_t DEVICE_TWO_ADIC_GENERATORS[28];



// 初始化函数（需要在主机代码中调用一次）
void init_two_adic_generators() {
#ifdef __CUDACC__
    cudaMemcpyToSymbol(DEVICE_TWO_ADIC_GENERATORS, 
                      HOST_TWO_ADIC_GENERATORS, 
                      sizeof(bb31_t)*28);
#endif
}

__host__ __device__ bb31_t two_adic_generator(int bits) {
    assert(bits <= 27); //p3 babybear is 27
#ifdef __CUDA_ARCH__
    return DEVICE_TWO_ADIC_GENERATORS[bits]; //device
#else
    return HOST_TWO_ADIC_GENERATORS[bits]; //host
#endif
}

__host__ __device__ inline unsigned int reverse_bits_len(unsigned int x, int bit_len) {
    if (bit_len == 0) return 0;
    unsigned int r = x;

    r = ((r & 0x55555555) << 1) | ((r >> 1) & 0x55555555);
    r = ((r & 0x33333333) << 2) | ((r >> 2) & 0x33333333);
    r = ((r & 0x0F0F0F0F) << 4) | ((r >> 4) & 0x0F0F0F0F);
    r = ((r & 0x00FF00FF) << 8) | ((r >> 8) & 0x00FF00FF);
    r = (r << 16) | (r >> 16);
    return r >> (32 - bit_len);
}


template <typename F>
__host__ __device__ F exp_u64(F base, uint64_t power) {
    if (power == 0) return F::one();
    if (power == 1) return base;
    
    if (power <= INT_MAX) {
        return base ^ (int)power;
    }
    
    F result = F::one();
    uint64_t remaining = power;
    

    constexpr uint64_t chunk_size = 1 << 30;
    while (remaining > 0) {
        uint64_t chunk = (remaining > chunk_size) ? chunk_size : remaining;
        F term = base ^ (int)chunk;
        result = result * term;
        remaining -= chunk;
    }
    
    return result;
}