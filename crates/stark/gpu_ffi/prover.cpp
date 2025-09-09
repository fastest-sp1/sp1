// In gpu_ffi/prover.cpp

#include "bb31_t.hpp"
#include "bb31_quartic_extension_t.hpp"
#include "utils.hpp"
#include <vector>

// =========================================================================
//            CUDA KERNEL for fold_even_odd
// =========================================================================
// This kernel implements the logic from p3_fri's `fold_even_odd` function.
__global__ void fold_even_odd_kernel(
    const bb31_quartic_extension_t* current_evals, // Input evals on GPU
    int num_current_evals,
    bb31_quartic_extension_t beta,
    const bb31_quartic_extension_t* powers,      // Precomputed (beta/2 * g_inv^i) on GPU
    bb31_quartic_extension_t* next_evals_out     // Output buffer on GPU
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_current_evals / 2) return;

    // The input `current_evals` corresponds to a matrix of width 2
    const bb31_quartic_extension_t r0 = current_evals[idx * 2 + 0];
    const bb31_quartic_extension_t r1 = current_evals[idx * 2 + 1];

    // Get the precomputed power: (beta/2) * g_inv^i
    const bb31_quartic_extension_t power = powers[idx];
    
    // Precompute 1/2
    const bb31_quartic_extension_t one_half(bb31_t(2).reciprocal());

    // result = (1/2 + power) * r0 + (1/2 - power) * r1
    next_evals_out[idx] = (one_half + power) * r0 + (one_half - power) * r1;
}

__global__ void vector_add_kernel(
    bb31_quartic_extension_t* a_inout,       // The vector to add to (on GPU)
    const bb31_quartic_extension_t* b_in,  // The vector to add (on GPU)
    int n                                    // The number of elements in the vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a_inout[idx] += b_in[idx];
    }
}


// =========================================================================
//            NEW FFI FUNCTION for FRI Value Injection
// =========================================================================
extern "C" int fri_injection_gpu(
    void* d_target_evals,      // The `d_current_evals` buffer to be modified
    const bb31_quartic_extension_t* h_injected_evals, // The values to inject, from HOST
    int num_elements
) {
    if (num_elements == 0) {
        return 0;
    }

    // 1. Allocate a temporary buffer on the GPU for the injected values
    bb31_quartic_extension_t* d_injected_evals;
    size_t size_bytes = (size_t)num_elements * sizeof(bb31_quartic_extension_t);
    CUDA_CHECK(cudaMalloc(&d_injected_evals, size_bytes));

    // 2. Copy the injected values from Host to Device
    CUDA_CHECK(cudaMemcpy(d_injected_evals, h_injected_evals, size_bytes, cudaMemcpyHostToDevice));

    // 3. Configure and launch the kernel
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim((num_elements + num_threads - 1) / num_threads);

    vector_add_kernel<<<grid_dim, block_dim>>>(
        (bb31_quartic_extension_t*)d_target_evals,
        d_injected_evals,
        num_elements
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Synchronize and clean up the temporary buffer
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_injected_evals));
    
    return 0;
}

// FFI FUNCTION 2: Fold on GPU
extern "C" int fri_fold_on_gpu(
    void* d_current_evals, // Input evals on DEVICE
    int num_current_evals,
    bb31_quartic_extension_t beta,
    const bb31_quartic_extension_t* h_powers, // Powers are small, computed on HOST
    void** d_next_evals_out // We allocate and return a new DEVICE pointer
) {
    int num_next_evals = num_current_evals / 2;
    if (num_next_evals == 0) {
        *d_next_evals_out = nullptr;
        return 0;
    }

    // Allocate output buffer on GPU
    CUDA_CHECK(cudaMalloc(d_next_evals_out, (size_t)num_next_evals * sizeof(bb31_quartic_extension_t)));
    
    // Allocate and copy powers vector for this fold
    bb31_quartic_extension_t* d_powers;
    CUDA_CHECK(cudaMalloc(&d_powers, (size_t)num_next_evals * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMemcpy(d_powers, h_powers, (size_t)num_next_evals * sizeof(bb31_quartic_extension_t), cudaMemcpyHostToDevice));

    // Configure and launch the kernel
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim((num_next_evals + num_threads - 1) / num_threads);

    fold_even_odd_kernel<<<grid_dim, block_dim>>>(
        (const bb31_quartic_extension_t*)d_current_evals,
        num_current_evals,
        beta,
        d_powers,
        (bb31_quartic_extension_t*)*d_next_evals_out
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Cleanup the temporary powers buffer
    CUDA_CHECK(cudaFree(d_powers));
    return 0;
}


// =========================================================================
//            FFI FUNCTION to call the fold_even_odd kernel
// =========================================================================
extern "C" int fold_even_odd_gpu(
    // Inputs from Host
    const bb31_quartic_extension_t* h_current_evals,
    int num_current_evals,
    bb31_quartic_extension_t beta,
    const bb31_quartic_extension_t* h_powers,

    // Output buffer on Host (pre-allocated by Rust)
    bb31_quartic_extension_t* h_next_evals_out
) {
    if (num_current_evals < 2) {
        return 0; // Nothing to fold
    }
    int num_next_evals = num_current_evals / 2;

    // --- 1. Allocate GPU memory ---
    bb31_quartic_extension_t* d_current_evals;
    bb31_quartic_extension_t* d_powers;
    bb31_quartic_extension_t* d_next_evals_out;

    size_t current_size_bytes = (size_t)num_current_evals * sizeof(bb31_quartic_extension_t);
    size_t next_size_bytes = (size_t)num_next_evals * sizeof(bb31_quartic_extension_t);

    CUDA_CHECK(cudaMalloc(&d_current_evals, current_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_powers, next_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_next_evals_out, next_size_bytes));

    // --- 2. Copy inputs from Host to Device ---
    CUDA_CHECK(cudaMemcpy(d_current_evals, h_current_evals, current_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_powers, h_powers, next_size_bytes, cudaMemcpyHostToDevice));

    // --- 3. Configure and launch the kernel ---
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim((num_next_evals + num_threads - 1) / num_threads);

    fold_even_odd_kernel<<<grid_dim, block_dim>>>(
        d_current_evals,
        num_current_evals,
        beta,
        d_powers,
        d_next_evals_out
    );
    CUDA_CHECK(cudaGetLastError());

    // --- 4. Synchronize and copy result back from Device to Host ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_next_evals_out, d_next_evals_out, next_size_bytes, cudaMemcpyDeviceToHost));
    
    // --- 5. Cleanup ---
    CUDA_CHECK(cudaFree(d_current_evals));
    CUDA_CHECK(cudaFree(d_powers));
    CUDA_CHECK(cudaFree(d_next_evals_out));
    
    return 0;
}
//tools function
extern "C" int cuda_malloc_and_memset_zero(void** devPtr, size_t size) {
    if (cudaMalloc(devPtr, size) != cudaSuccess) return -1;
    if (cudaMemset(*devPtr, 0, size) != cudaSuccess) return -1;
    return 0;
}

// You should already have this one:
extern "C" int cuda_memcpy_dtoh(void* dst, const void* src, size_t count) {
    if (cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

// You should already have this one:
extern "C" int cuda_free(void* devPtr) {
    if (cudaFree(devPtr) != cudaSuccess) return -1;
    return 0;
}


extern "C" int cuda_malloc(void** devPtr, size_t size) {
    if (cudaMalloc(devPtr, size) != cudaSuccess) {
        // You can add more detailed error printing here if you want
        return -1; // Return a non-zero error code on failure
    }
    return 0; // Return 0 on success
}


extern "C" int cuda_memcpy_htod(void* dst_device, const void* src_host, size_t count) {
    if (cudaMemcpy(dst_device, src_host, count, cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }
    return 0;
}
