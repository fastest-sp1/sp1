#include "gpu_types.hpp"
#include "bb31_t.hpp"
#include "bb31_quartic_extension_t.hpp"
#include "utils.hpp"

// This kernel computes inverse denominators using direct inversion.
// It  requires no shared memory.
__global__ void compute_inverse_denominators_direct_kernel(
    Challenge z,                // The query point
    const bb31_t* d_coset,        // The domain elements {x_i}
    int num_elements,             // The number of elements in the coset subset
    Challenge* d_inv_denoms_out // The output buffer
) {
    // Standard grid-stride loop for maximum parallelism and scalability.
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= num_elements) return;

    Challenge denominator  = z - Challenge(d_coset[r]);
    d_inv_denoms_out[r] = denominator.reciprocal();
}


extern "C" int compute_inverse_denominators_for_points_gpu(
    const Challenge* h_points,
    int num_points,
    const int* h_max_log_heights,
    const bb31_t* d_coset,
    Challenge** h_inv_denoms_ptr_array
) {
    // We can launch all kernels asynchronously on the same stream, or use different streams.
    // For simplicity, we'll use the default stream.
    for (int i = 0; i < num_points; ++i) {
        Challenge z = h_points[i];
        int log_height = h_max_log_heights[i];
        int num_elements = 1 << log_height;
        //printf("---gpu222--, num_points=%u , point=[%u %u %u %u], log_height=%u\n", num_points, z.coeffs[0].as_canonical_u32(),
        //    z.coeffs[1].as_canonical_u32(),z.coeffs[2].as_canonical_u32(),z.coeffs[3].as_canonical_u32(),
        //    log_height);

        // 1. Allocate a new GPU buffer for the results.
        Challenge* d_inv_denoms_out;
        CUDA_CHECK(cudaMalloc(&d_inv_denoms_out, num_elements * sizeof(Challenge)));
        
        // 2. Determine kernel launch configuration.
        int threads_per_block = 256;
        int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        
        // 3. Launch the simple, direct inversion kernel.
        compute_inverse_denominators_direct_kernel<<<blocks_per_grid, threads_per_block>>>(
            z,
            d_coset,
            num_elements,
            d_inv_denoms_out
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 4. Store the resulting device pointer.
        h_inv_denoms_ptr_array[i] = d_inv_denoms_out;
    }

    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all kernels to complete.

    return 0; // Success
}