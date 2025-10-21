#pragma once
#include "bb31_t.hpp"
#include "bb31_quartic_extension_t.hpp"
#include "gpu_types.hpp"
#include "utils.hpp"

// Kernel to perform column-wise dot product and the final scaling.
// Each block computes one output value (one dot product for one column).
// Each thread within a block handles a portion of the dot product sum.
__global__ void interpolate_kernel(
    const GpuMatrix<bb31_t> coset_evals,      // GpuMatrix<bb31_t>
    const bb31_t* d_coset,              // Device pointer to coset elements
    const Challenge* d_diff_invs,     // Device pointer to precomputed (z - x_i)^-1
    Challenge* d_dot_products_out,    // Output buffer of size `coset_evals.width`
    Challenge final_scaler           // The precomputed (z^N - g^N) / (N * g^N)
) {
    // Each block is responsible for one column of the `coset_evals` matrix.
    int col_idx = blockIdx.x;
    if (col_idx >= coset_evals.width) return;

    // Use shared memory for parallel reduction (summation) within the block.
    extern __shared__ Challenge s_partials[];
    int tid = threadIdx.x;

    Challenge thread_sum = Challenge::zero();
    
    // Grid-stride loop for each thread to sum its portion of the column.
    for (int row_idx = tid; row_idx < coset_evals.height; row_idx += blockDim.x) {
        // Compute `col_scale` on-the-fly.
        Challenge col_scale_i = d_diff_invs[row_idx] * Challenge(d_coset[row_idx]);
        
        // Get the evaluation value for the current row and column.
        const bb31_t* row_ptr = static_cast<const bb31_t*>(coset_evals.d_data) + (size_t)row_idx * coset_evals.width;
        bb31_t eval = row_ptr[col_idx];
        
        thread_sum = thread_sum + Challenge(eval) * col_scale_i;
    }
    
    s_partials[tid] = thread_sum;
    __syncthreads();

    // Perform parallel reduction in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partials[tid] = s_partials[tid] + s_partials[tid + s];
        }
        __syncthreads();
    }

    // The first thread writes the final sum for this column to global memory.
    if (tid == 0) {
        d_dot_products_out[col_idx] = s_partials[0] * final_scaler;
    }
}



extern "C" int interpolate_coset_gpu(
    //const GpuMatrix<bb31_t>* coset_evals,      // GpuMatrix<bb31_t>
    GpuMatrix<bb31_t>* coset_evals, 
    bb31_t shift,
    Challenge point,
    const GpuMatrix<bb31_t>* coset,            // GpuMatrix<bb31_t>
    const GpuMatrix<Challenge>* diff_invs,        // GpuMatrix<bb31_t><>
    Challenge* h_interpolated_values_out // Host output buffer
) {
    //For saving mem
    coset_evals->height = coset->height;//Move the logic from rust to here!

    int height = coset_evals->height;
    int width = coset_evals->width;
    int log_height = integer_log2(height);

    // 1. Calculate the final scaler on the host (it's a single value).
    Challenge point_pow_height = point.pow(height);
    Challenge shift_pow_height = Challenge(shift).pow(height);
    Challenge vanishing_poly = point_pow_height - shift_pow_height;
    
    Challenge denominator = shift_pow_height * Challenge(Val::from_canonical_u32(height));
    Challenge final_scaler = vanishing_poly * denominator.reciprocal();
    
    // 2. Allocate device memory for the intermediate dot products.
    Challenge* d_dot_products;
    CUDA_CHECK(cudaMalloc(&d_dot_products, width * sizeof(Challenge)));

    // 3. Configure and launch the kernel.
    int threads_per_block = 256; // A reasonable default for reduction
    dim3 grid_dim(width); // Launch one block per column
    dim3 block_dim(threads_per_block);
    size_t shared_mem_size = threads_per_block * sizeof(Challenge);

    interpolate_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        *coset_evals,
        static_cast<const bb31_t*>(coset->d_data),
        static_cast<const Challenge*>(diff_invs->d_data),
        d_dot_products,
        final_scaler
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 4. Copy the final results from device to host.
    CUDA_CHECK(cudaMemcpy(h_interpolated_values_out, d_dot_products, width * sizeof(Challenge), cudaMemcpyDeviceToHost));
    
    // 5. Cleanup
    CUDA_CHECK(cudaFree(d_dot_products));
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0; 
}