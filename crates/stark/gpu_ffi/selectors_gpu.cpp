// file: selectors_gpu.cu

#include "bb31_t.hpp" 
#include "utils.hpp"
#include "selectors.hpp"
#include <vector>
// A struct to hold pointers to the output device buffers.
// This is defined on the C++ side and will be mirrored on the Rust side.
struct LagrangeSelectorsDevicePtrs {
    Val* is_first_row;
    Val* is_last_row;
    Val* is_transition;
    Val* inv_vanishing;
};


__constant__ int d_rate_bits;

__global__ void generate_coset_points_kernel(Val generator, Val shift, Val* d_points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_points[idx] = generator.exp_u64(idx) * shift;
}



// This kernel calculates the `evals` of Z_H(X) on the coset.
__global__ void calculate_zh_evals_kernel(
    int rate_bits,
    Val s_pow_n,
    Val generator,
    Val* d_evals_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (1 << rate_bits)) return;

    //Val two_adic_gen = two_adic_generator(rate_bits);
    Val current_power = generator.exp_u64(idx); // pow needs to be efficient
    d_evals_out[idx] = s_pow_n * current_power - Val::one();
}

// This kernel fills a large buffer by cycling through a smaller one.
template <typename T>
__global__ void cycle_fill_kernel(
    const T* d_source,
    int rate_bits,
    T* d_dest,
    int dest_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dest_len) return;

    int source_idx = idx & ((1 << rate_bits) - 1);

    d_dest[idx] = d_source[source_idx];
}

// Kernel to compute is_first/is_last selectors
__global__ void single_point_selector_kernel(
    const Val* d_inv_denoms, const Val* d_zh_evals, int num_zh_evals,
    Val* d_selector_out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_selector_out[idx] = d_zh_evals[idx % num_zh_evals] * d_inv_denoms[idx];
}

// Kernel to compute `x - c`
__global__ void subtract_constant_kernel(const Val* d_x_values, Val constant, Val* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_out[idx] = d_x_values[idx] - constant;
}

/**
 * @brief Kernel to compute all Lagrange selectors in parallel.
 */
__global__ void selectors_on_coset_kernel(
    int coset_size,
    Val coset_shift,
    Val coset_subgroup_generator,
    Val trace_subgroup_first, // h^0 = 1
    Val trace_subgroup_last,  // h^{n-1} = h^{-1}
    Val *d_evals_inv,
    Val *d_evals,
    LagrangeSelectorsDevicePtrs selectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coset_size) return;

    // Calculate the point `x` in the coset corresponding to this thread.
    Val x = coset_shift * coset_subgroup_generator.exp_u64(idx);
    
    // --- Compute `is_transition` and `inv_vanishing` ---
    selectors.is_transition[idx] = x - trace_subgroup_last;

    int evals_idx = idx & ((1 << d_rate_bits) - 1);
    selectors.inv_vanishing[idx] = d_evals_inv[evals_idx];
    
    // --- Compute `is_first_row` and `is_last_row` ---
    // This is a simplified (less efficient) version without batch inverse on GPU.
    // It's correct for testing.
    Val z_h = d_evals[evals_idx];
    
    Val denom_first = x - trace_subgroup_first;
    selectors.is_first_row[idx] = z_h * denom_first.reciprocal();
    
    Val denom_last = x - trace_subgroup_last;
    selectors.is_last_row[idx] = z_h * denom_last.reciprocal();
}

//From plonky3: batch_multiplicative_inverse_general
std::vector<Val> batch_multiplicative_inverse_cpu(const std::vector<Val>& x) {
    int n = x.size();
    if (n == 0) {
        return std::vector<Val>();
    }


    // This is a direct, line-by-line translation of the Rust `batch_multiplicative_inverse_general`.
    // The vector `products` here serves the same purpose as `result` in the Rust code.
    std::vector<Val> products_then_inverses(n);
    
    products_then_inverses[0] = Val::one();
    for (int i = 1; i < n; ++i) {
        products_then_inverses[i] = products_then_inverses[i - 1] * x[i - 1];
    }
   

    // 2. Compute the inverse of the total product.
    Val total_product = products_then_inverses[n - 1] * x[n - 1];
  
    if (total_product.is_zero()) {
        printf("ERROR in batch_multiplicative_inverse_cpu: trying to invert a zero element.\n");
        return std::vector<Val>(n, Val::zero());
    }
    Val inv = total_product.reciprocal();
 
    // 3. Suffix scan to compute individual inverses.
    // We will now modify the `products` vector in place, from back to front,
    // to transform it from prefix products into the final inverses.
   for (int i = n - 1; i >= 0; --i) {
        products_then_inverses[i] *= inv;
        inv = inv * x[i];
    }
    
    return products_then_inverses;
}

// --- STUB for Batch Inverse ---
Val* batch_multiplicative_inverse_gpu_stub(const Val* d_in, int n) {
    if (n == 0) return nullptr;
    std::vector<Val> h_in(n);
    cudaMemcpy(h_in.data(), d_in, n * sizeof(Val), cudaMemcpyDeviceToHost);
//printf("---GPU, h_in.0=%u, .1=%u, .2=%u\n", h_in[0].as_canonical_u32(), h_in[1].as_canonical_u32(), h_in[2].as_canonical_u32());
    std::vector<Val> h_out = batch_multiplicative_inverse_cpu(h_in);
//    printf("---GPU, h_out.0=%u, .1=%u, .2=%u\n", h_out[0].as_canonical_u32(), h_out[1].as_canonical_u32(), h_out[2].as_canonical_u32());
    Val* d_out;
    cudaMalloc(&d_out, n * sizeof(Val));
    cudaMemcpy(d_out, h_out.data(), n * sizeof(Val), cudaMemcpyHostToDevice);
    return d_out;
}

// Internal C++ function to generate selectors directly on the GPU.
//   This encapsulates the logic from  FFI function: selectors_on_coset_gpu.
int generate_selectors_on_device(
    int trace_log_size, 
    Val trace_subgroup_generator,
    int coset_log_size, 
    Val coset_shift, 
    Val coset_subgroup_generator,
    Val* d_is_first_row, 
    Val* d_is_last_row,
    Val* d_is_transition, 
    Val* d_inv_vanishing
) {
    int coset_size = 1 << coset_log_size;
    int trace_size = 1 << trace_log_size;
    int rate_bits = coset_log_size - trace_log_size;
    int num_evals = 1 << rate_bits;

    // Common setup
    int threads = 256;
    int blocks_coset = (coset_size + threads - 1) / threads;

    // --- Generate Z_H evals (needed for multiple selectors) ---
    Val* d_evals;
    cudaMalloc(&d_evals, num_evals * sizeof(Val));
    Val s_pow_n = coset_shift.exp_power_of_2(trace_log_size);
    Val two_adic_gen = two_adic_generator(rate_bits);
    int blocks_evals = (num_evals + threads - 1) / threads;
    calculate_zh_evals_kernel<<<blocks_evals, threads>>>(rate_bits, s_pow_n, two_adic_gen, d_evals);

     // --- 1. Compute inv_vanishing ---
    Val* d_inv_evals = batch_multiplicative_inverse_gpu_stub(d_evals, num_evals);
    if (d_inv_evals == nullptr) { 
        cudaFree(d_evals); 
        return -1; 
    }
    cycle_fill_kernel<<<blocks_coset, threads>>>(d_inv_evals, rate_bits, d_inv_vanishing, coset_size);
    cudaFree(d_inv_evals);
     
    // --- 2. Generate coset points (xs) needed for other selectors ---
    Val* d_coset_points;
    cudaMalloc(&d_coset_points, coset_size * sizeof(Val));
    generate_coset_points_kernel<<<blocks_coset, threads>>>(coset_subgroup_generator, coset_shift, d_coset_points, coset_size);

    // --- 3. Compute is_first_row ---
   Val first_point = trace_subgroup_generator.exp_u64(0); // This is just ONE
    Val* d_denoms_first;
   cudaMalloc(&d_denoms_first, coset_size * sizeof(Val));
    subtract_constant_kernel<<<blocks_coset, threads>>>(d_coset_points, first_point, d_denoms_first, coset_size);
    Val* d_inv_denoms_first = batch_multiplicative_inverse_gpu_stub(d_denoms_first, coset_size);
    single_point_selector_kernel<<<blocks_coset, threads>>>(d_inv_denoms_first, d_evals, num_evals, d_is_first_row, coset_size);
    cudaFree(d_denoms_first); 
    cudaFree(d_inv_denoms_first);

     // --- 4. Compute is_last_row ---
    Val last_point = trace_subgroup_generator.reciprocal(); // More robust than pow(n-1)
    Val* d_denoms_last;
    cudaMalloc(&d_denoms_last, coset_size * sizeof(Val));
    subtract_constant_kernel<<<blocks_coset, threads>>>(d_coset_points, last_point, d_denoms_last, coset_size);
    Val* d_inv_denoms_last = batch_multiplicative_inverse_gpu_stub(d_denoms_last, coset_size);
    single_point_selector_kernel<<<blocks_coset, threads>>>(d_inv_denoms_last, d_evals, num_evals, d_is_last_row, coset_size);
    cudaFree(d_denoms_last); cudaFree(d_inv_denoms_last);

    // --- 5. Compute is_transition ---
    Val subgroup_last = trace_subgroup_generator.reciprocal();
    subtract_constant_kernel<<<blocks_coset, threads>>>(d_coset_points, subgroup_last, d_is_transition, coset_size);

    // --- Final Cleanup ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_coset_points));
    CUDA_CHECK(cudaFree(d_evals));

    return 0;
}

extern "C" int generate_selectors_on_device_gpu(
    int trace_log_size, 
    Val trace_subgroup_generator,
    int coset_log_size, 
    Val coset_shift, 
    Val coset_subgroup_generator,
    Val* h_is_first_row, 
    Val* h_is_last_row,
    Val* h_is_transition, 
    Val* h_inv_vanishing
) {

    int coset_size = 1 << coset_log_size;
    int trace_size = 1 << trace_log_size;
    int rate_bits = coset_log_size - trace_log_size;
    int num_evals = 1 << rate_bits;

    Val *d_is_first_row, *d_is_last_row, *d_is_transition, *d_inv_vanishing;

    CUDA_CHECK(cudaMalloc(&d_is_first_row, coset_size * sizeof(Val)));
    CUDA_CHECK(cudaMalloc(&d_is_last_row, coset_size * sizeof(Val)));
    CUDA_CHECK(cudaMalloc(&d_is_transition, coset_size * sizeof(Val)));
    CUDA_CHECK(cudaMalloc(&d_inv_vanishing, coset_size * sizeof(Val)));

    //
    generate_selectors_on_device(
            trace_log_size,
            trace_subgroup_generator,
            coset_log_size,
            coset_shift,
            coset_subgroup_generator,
            d_is_first_row,
            d_is_last_row,
            d_is_transition,
            d_inv_vanishing
        );

    CUDA_CHECK(cudaMemcpy(h_is_first_row, d_is_first_row, coset_size * sizeof(Val), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_is_last_row, d_is_last_row, coset_size * sizeof(Val), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_is_transition, d_is_transition, coset_size * sizeof(Val), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_inv_vanishing, d_inv_vanishing, coset_size * sizeof(Val), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_is_first_row));
    CUDA_CHECK(cudaFree(d_is_last_row));
    CUDA_CHECK(cudaFree(d_is_transition));
    CUDA_CHECK(cudaFree(d_inv_vanishing));

    return 0;
}
// FFI Implementation
extern "C" int selectors_on_coset_gpu( //pass
    // Trace Domain Info
    int trace_log_size,
    const Val* h_trace_subgroup_generator,

    // Quotient Domain (Coset) Info
    int coset_log_size,
    const Val* h_coset_shift,
    const Val* h_coset_subgroup_generator,

    // Host pointers to the output buffers
    Val* h_is_first_row,
    Val* h_is_last_row,
    Val* h_is_transition,
    Val* h_inv_vanishing
) {
    // 1. Host-side pre-computation (mirroring Rust `selectors_on_coset`)
    int rate_bits = coset_log_size - trace_log_size;
    Val shift = *h_coset_shift;
    Val s_pow_n = shift.exp_power_of_2(trace_log_size);
    Val two_adic_gen = two_adic_generator(rate_bits);

    int num_evals = 1 << rate_bits;

    Val* h_evals = new Val[num_evals];
    Val* h_evals_inv = new Val[num_evals];


    //Val h_evals[MAX_EVALS_FOR_SELECTORS];
    Val current_power = Val::one();
    for (int i = 0; i < num_evals; ++i) {
        h_evals[i] = s_pow_n * current_power - Val::one();
        current_power *= two_adic_gen;
    }
  
    //Val h_evals_inv[MAX_EVALS_FOR_SELECTORS];
    // A proper implementation would use a C++ batch inverse function.
    // For the test, individual inversion is acceptable.
    for (int i = 0; i < num_evals; ++i) {
        h_evals_inv[i] = h_evals[i].reciprocal();
    }
    
    // 2. Copy pre-computed data to constant memory
    cudaMemcpyToSymbol(d_rate_bits, &rate_bits, sizeof(int));

    Val *d_evals_inv, *d_evals;
    cudaMalloc(&d_evals_inv, num_evals * sizeof(Val));
    cudaMalloc(&d_evals, num_evals * sizeof(Val));
    cudaMemcpy(d_evals_inv, h_evals_inv, num_evals * sizeof(Val), cudaMemcpyHostToDevice);
    cudaMemcpy(d_evals, h_evals, num_evals * sizeof(Val), cudaMemcpyHostToDevice);
    
    // 3. Allocate device memory for outputs
    int coset_size = 1 << coset_log_size;
    LagrangeSelectorsDevicePtrs d_ptrs;
    size_t buffer_size = (size_t)coset_size * sizeof(Val);
    cudaMalloc(&d_ptrs.is_first_row, buffer_size);
    cudaMalloc(&d_ptrs.is_last_row, buffer_size);
    cudaMalloc(&d_ptrs.is_transition, buffer_size);
    cudaMalloc(&d_ptrs.inv_vanishing, buffer_size);

    // 4. Setup kernel launch
    int threads = 256;
    int blocks = (coset_size + threads - 1) / threads;
    
    Val trace_gen = *h_trace_subgroup_generator;

    // 5. Launch Kernel
    selectors_on_coset_kernel<<<blocks, threads>>>(
        coset_size, 
        shift, 
        *h_coset_subgroup_generator,
        Val::one(), 
        trace_gen.reciprocal(),
        d_evals_inv,
        d_evals,
        d_ptrs
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { 
        // Cleanup memory before returning error
        cudaFree(d_ptrs.is_first_row);
        cudaFree(d_ptrs.is_last_row);
        cudaFree(d_ptrs.is_transition);
        cudaFree(d_ptrs.inv_vanishing);
        cudaFree(d_evals_inv);
        cudaFree(d_evals);

        delete[] h_evals;
        delete[] h_evals_inv;

        return (int)err; 
    }
    
    // 6. Copy results back to host
    cudaMemcpy(h_is_first_row, d_ptrs.is_first_row, buffer_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_is_last_row, d_ptrs.is_last_row, buffer_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_is_transition, d_ptrs.is_transition, buffer_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inv_vanishing, d_ptrs.inv_vanishing, buffer_size, cudaMemcpyDeviceToHost);
    
    // 7. Cleanup
    cudaFree(d_ptrs.is_first_row);
    cudaFree(d_ptrs.is_last_row);
    cudaFree(d_ptrs.is_transition);
    cudaFree(d_ptrs.inv_vanishing);
    cudaFree(d_evals_inv);
    cudaFree(d_evals);

    delete[] h_evals;
    delete[] h_evals_inv;

    return 0; // Success
}

/*
v1
//using namespace sp1_gpu;

// Constant memory for pre-computed values shared by all threads.
#define MAX_EVALS (1 << 8) // Max quotient_degree = 2^8=256, adjust if needed
__constant__ Val d_evals_inv[MAX_EVALS];
__constant__ int d_rate_bits;


//@brief Kernel to compute `is_transition` and `inv_vanishing` selectors.

__global__ void selectors_kernel_part1(
    int coset_size,
    Val coset_shift,
    Val coset_subgroup_generator,
    Val trace_subgroup_last, // h^{-1}
    LagrangeSelectorsDevicePtrs selectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coset_size) return;

    // Calculate the point x in the coset corresponding to this thread.
    Val x = coset_shift * coset_subgroup_generator.exp_u64(idx);
    
    // 1. Compute `is_transition = x - h^{-1}`
    selectors.is_transition[idx] = x - trace_subgroup_last;

    // 2. Compute `inv_vanishing`
    // The index into the cycled `evals` array is `idx % (1 << rate_bits)`.
    int evals_idx = idx & ((1 << d_rate_bits) - 1);
    selectors.inv_vanishing[idx] = d_evals_inv[evals_idx];
}


// NOTE: `is_first_row` and `is_last_row` are much more complex. They require a
// batch inversion on the GPU, which is a multi-step process. For a complete
// implementation, one would use a library like CUB or write a custom parallel scan.
// For now, we will implement a simplified, less efficient version inside the kernel
// to demonstrate the logic. A production version would optimize this heavily.


 //@brief Kernel to compute `is_first_row` and `is_last_row`.
 //     This is a simplified, non-optimal implementation for clarity.

__global__ void selectors_kernel_part2(
    int coset_size,
    Val coset_shift,
    Val coset_subgroup_generator,
    Val trace_subgroup_first, // h^0 = 1
    Val trace_subgroup_last,  // h^{n-1} = h^{-1}
    const Val* d_evals, // The original (non-inverted) evals
    LagrangeSelectorsDevicePtrs selectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= coset_size) return;

    Val x = coset_shift * coset_subgroup_generator.exp_u64(idx);

    // For is_first_row: denoms = x - h^0
    Val denom_first = x - trace_subgroup_first;
    
    // For is_last_row: denoms = x - h^{-1}
    Val denom_last = x - trace_subgroup_last;

    // The Z_H(x) value for this point
    int evals_idx = idx & ((1 << d_rate_bits) - 1);
    Val z_h = d_evals[evals_idx];
    
    // This is NOT batch-inverse, but does the same math.
    selectors.is_first_row[idx] = z_h * denom_first.reciprocal();
    selectors.is_last_row[idx] = z_h * denom_last.reciprocal();
}


// FFI Implementation
extern "C" int selectors_on_coset_gpu(
    int trace_log_size,
    const Val* h_trace_subgroup_generator,
    int coset_log_size,
    const Val* h_coset_shift,
    const Val* h_coset_subgroup_generator,
    LagrangeSelectorsDevicePtrs* d_selectors_ptrs
) {
    // 1. Host-side pre-computation (mirroring Rust)
    int rate_bits = coset_log_size - trace_log_size;
    Val s = *h_coset_shift;
    Val s_pow_n = s.exp_power_of_2(trace_log_size);
    Val two_adic_gen = Val::two_adic_generator(rate_bits);

    int num_evals = 1 << rate_bits;
    if (num_evals > MAX_EVALS) return -1; // Error check

    Val h_evals[MAX_EVALS];
    Val current_power = Val::one();
    for (int i = 0; i < num_evals; ++i) {
        h_evals[i] = s_pow_n * current_power - Val::one();
        current_power *= two_adic_gen;
    }
    
    // Perform batch inverse on the host (much simpler than on GPU).
    Val h_evals_inv[MAX_EVALS];
    // This would be a call to a C++ batch_inverse function, or we can
    // just do individual inverses since num_evals is small.
    for (int i = 0; i < num_evals; ++i) {
        h_evals_inv[i] = h_evals[i].reciprocal();
    }
    
    // 2. Copy pre-computed data to constant memory
    cudaMemcpyToSymbol(d_evals_inv, h_evals_inv, num_evals * sizeof(Val));
    cudaMemcpyToSymbol(d_rate_bits, &rate_bits, sizeof(int));
    
    // 3. Setup kernel launch
    int coset_size = 1 << coset_log_size;
    int threads = 256;
    int blocks = (coset_size + threads - 1) / threads;
    
    Val trace_gen = *h_trace_subgroup_generator;
    Val coset_gen = *h_coset_subgroup_generator;

    // 4. Launch Kernel Part 1
    selectors_kernel_part1<<<blocks, threads>>>(
        coset_size, s, coset_gen, trace_gen.reciprocal(),
        *d_selectors_ptrs
    );
    
    // 5. Launch Kernel Part 2 (Simplified version)
    // First, copy original evals to a temp device buffer.
    Val* d_evals_temp;
    cudaMalloc(&d_evals_temp, num_evals * sizeof(Val));
    cudaMemcpy(d_evals_temp, h_evals, num_evals * sizeof(Val), cudaMemcpyHostToDevice);

    selectors_kernel_part2<<<blocks, threads>>>(
        coset_size, s, coset_gen,
        Val::one(), trace_gen.reciprocal(),
        d_evals_temp,
        *d_selectors_ptrs
    );

    cudaFree(d_evals_temp);
    cudaDeviceSynchronize();
    return 0;
}
*/
