#include "bb31_t.hpp" 
#include "bb31_quartic_extension_t.hpp"
#include "utils.hpp"
#include "gpu_types.hpp"
#include <vector>

struct OpeningPointInfo {
    const bb31_quartic_extension_t* points_z;
    const bb31_quartic_extension_t* opened_values_y;
    int num_points;
};

// --- KERNEL 1: Computes `P_mat(x) = Σ_j α^j * p_j(x)` for all x in the domain. ---
// This compresses a base-field matrix into an extension-field vector.
__global__ void compute_mat_compressed_kernel(
    const bb31_t* lde_data, int h, int w,
    const bb31_quartic_extension_t* alpha_powers,
    bb31_quartic_extension_t* mat_compressed_out)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= h) return;

    bb31_quartic_extension_t sum; // Default constructor initializes to zero
    const bb31_t* row_ptr = lde_data + (size_t)r * w;

    for (int c = 0; c < w; ++c) {
        // Here, we multiply an extension field element (alpha_powers[c])
        // by a base field element (row_ptr[c]).
        sum += alpha_powers[c] * row_ptr[c];
    }
    mat_compressed_out[r] = sum;
}

// --- KERNEL 2: Computes `1 / (z - x)` for a given z and all x in a coset ---
__global__ void compute_inv_denoms_kernel(
    bb31_quartic_extension_t z,
    const bb31_t* coset,
    int h,
    bb31_quartic_extension_t* inv_denoms_out)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= h) return;

    bb31_quartic_extension_t term = z - bb31_quartic_extension_t(coset[r]);
    inv_denoms_out[r] = term.reciprocal();
}


// --- KERNEL 3: The main loop that accumulates the quotient polynomial terms ---
__global__ void compute_quotient_main_loop_kernel(
    const bb31_quartic_extension_t* mat_compressed,
    const bb31_quartic_extension_t* inv_denoms,
    bb31_quartic_extension_t y_mat,
    bb31_quartic_extension_t alpha_pow_offset,
    int h,
    bb31_quartic_extension_t* quotient_evals_inout)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= h) return;
    
    bb31_quartic_extension_t term = (y_mat - mat_compressed[r]) * inv_denoms[r];
    
    // This is an atomic operation on global memory, but since each thread
    // writes to a unique location `r`, it's safe. For accumulation from
    // different matrices, we need separate buffers or atomicAdd.
    // The current design correctly accumulates on the host side loop.
    quotient_evals_inout[r] += alpha_pow_offset * term;
}

//test
extern "C" int stark_test_mat_compress_gpu(
    const bb31_t* lde_data, int h, int w,
    const bb31_quartic_extension_t* alpha_powers,
    bb31_quartic_extension_t* mat_compressed_out)
{
    bb31_t* d_lde;
    bb31_quartic_extension_t *d_alpha, *d_out;
    CUDA_CHECK(cudaMalloc(&d_lde, (size_t)h * w * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_alpha, (size_t)w * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)h * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMemcpy(d_lde, lde_data, (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha, alpha_powers, (size_t)w * sizeof(bb31_quartic_extension_t), cudaMemcpyHostToDevice));

    dim3 grid_dim((h + 255) / 256);
    dim3 block_dim(256);

    compute_mat_compressed_kernel<<<grid_dim, block_dim>>>(d_lde, h, w, d_alpha, d_out);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(mat_compressed_out, d_out, (size_t)h * sizeof(bb31_quartic_extension_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_lde));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

extern "C" int stark_test_inv_denoms_gpu(
    bb31_quartic_extension_t z,
    const bb31_t* coset,
    int h,
    bb31_quartic_extension_t* inv_denoms_out)
{
    if (h == 0) return 0;

    // --- 1. GPU Memory Allocation & Data Transfer ---
    bb31_t* d_coset;
    bb31_quartic_extension_t* d_inv_denoms_out;
    
    CUDA_CHECK(cudaMalloc(&d_coset, (size_t)h * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_inv_denoms_out, (size_t)h * sizeof(bb31_quartic_extension_t)));
    
    CUDA_CHECK(cudaMemcpy(d_coset, coset, (size_t)h * sizeof(bb31_t), cudaMemcpyHostToDevice));

    // --- 2. Kernel Launch Configuration ---
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim((h + num_threads - 1) / num_threads);

    // --- 3. Launch the Kernel ---
    compute_inv_denoms_kernel<<<grid_dim, block_dim>>>(
        z, d_coset, h, d_inv_denoms_out);
    CUDA_CHECK(cudaGetLastError());

    // --- 4. Synchronize and Copy Result Back ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(inv_denoms_out, d_inv_denoms_out, (size_t)h * sizeof(bb31_quartic_extension_t), cudaMemcpyDeviceToHost));
    
    // --- 5. Cleanup ---
    CUDA_CHECK(cudaFree(d_coset));
    CUDA_CHECK(cudaFree(d_inv_denoms_out));
    
    return 0;
}

extern "C" int stark_test_quartic_mul(const bb31_quartic_extension_t* a, const bb31_quartic_extension_t* b, bb31_quartic_extension_t* out) {
    *out = *a * *b;
    return 0;
}

extern "C" int stark_test_quotient_loop_gpu(
    const bb31_quartic_extension_t* mat_compressed,
    const bb31_quartic_extension_t* inv_denoms,
    bb31_quartic_extension_t y_mat,
    bb31_quartic_extension_t alpha_pow_offset,
    int h,
    bb31_quartic_extension_t* quotient_evals_inout)
{
    if (h == 0) return 0;

    // --- 1. GPU Memory Allocation ---
    bb31_quartic_extension_t *d_mat_compressed, *d_inv_denoms, *d_quotient_evals;
    size_t buffer_size_bytes = (size_t)h * sizeof(bb31_quartic_extension_t);

    CUDA_CHECK(cudaMalloc(&d_mat_compressed, buffer_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_inv_denoms, buffer_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_quotient_evals, buffer_size_bytes));

    // --- 2. Host to Device Data Transfer ---
    CUDA_CHECK(cudaMemcpy(d_mat_compressed, mat_compressed, buffer_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv_denoms, inv_denoms, buffer_size_bytes, cudaMemcpyHostToDevice));
    // Copy the initial state of the quotient buffer to the device.
    CUDA_CHECK(cudaMemcpy(d_quotient_evals, quotient_evals_inout, buffer_size_bytes, cudaMemcpyHostToDevice));

    // --- 3. Kernel Launch Configuration ---
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim((h + num_threads - 1) / num_threads);
    
    // --- 4. Launch the Kernel ---
    compute_quotient_main_loop_kernel<<<grid_dim, block_dim>>>(
        d_mat_compressed,
        d_inv_denoms,
        y_mat,
        alpha_pow_offset,
        h,
        d_quotient_evals
    );
    CUDA_CHECK(cudaGetLastError());

    // --- 5. Synchronize and Copy Result Back ---
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy the final, modified quotient buffer back to the host.
    CUDA_CHECK(cudaMemcpy(quotient_evals_inout, d_quotient_evals, buffer_size_bytes, cudaMemcpyDeviceToHost));

    // --- 6. Cleanup ---
    CUDA_CHECK(cudaFree(d_mat_compressed));
    CUDA_CHECK(cudaFree(d_inv_denoms));
    CUDA_CHECK(cudaFree(d_quotient_evals));

    return 0;
}


//pass
extern "C" int stark_test_quartic_pow(
    const bb31_quartic_extension_t* base, 
    uint64_t exponent, 
    bb31_quartic_extension_t* out
) {
    *out = base->pow(exponent);
    return 0; // Return 0 on success
}


extern "C" int compute_and_accumulate_quotient_gpu(
    // Input Matrix Data
    const bb31_t* h_lde_data, // Matrix data on the HOST
    int h,
    int w,

    // Parameters
    const bb31_quartic_extension_t* h_alpha_powers, // on HOST
    int num_alpha_powers,
    const bb31_t* h_coset, // The GLOBAL coset on HOST
    
    // Opening Data for the single point
    bb31_quartic_extension_t point_z,
    const bb31_quartic_extension_t* h_opened_values_y, // For this single point, on HOST
    bb31_quartic_extension_t alpha_pow_offset,        // Pre-computed on HOST
    
    // In-Out Buffer on GPU
    bb31_quartic_extension_t* d_quotient_evals_inout
) {
    if (h == 0) return 0;
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_h((h + num_threads - 1) / num_threads);

    // --- Malloc and Memcpy data needed for this specific computation ---
    bb31_t* d_lde_data;
    bb31_quartic_extension_t* d_alpha_powers;
    bb31_quartic_extension_t* d_mat_compressed;
    bb31_t* d_coset_for_height;
    bb31_quartic_extension_t* d_inv_denoms;

    CUDA_CHECK(cudaMalloc(&d_lde_data, (size_t)h * w * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_alpha_powers, (size_t)num_alpha_powers * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMalloc(&d_mat_compressed, (size_t)h * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMalloc(&d_coset_for_height, (size_t)h * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_inv_denoms, (size_t)h * sizeof(bb31_quartic_extension_t)));

    CUDA_CHECK(cudaMemcpy(d_lde_data, h_lde_data, (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha_powers, h_alpha_powers, (size_t)num_alpha_powers * sizeof(bb31_quartic_extension_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coset_for_height, h_coset, (size_t)h * sizeof(bb31_t), cudaMemcpyHostToDevice));

    // Step A: Compute mat_compressed on GPU
    compute_mat_compressed_kernel<<<grid_dim_h, block_dim>>>(d_lde_data, h, w, d_alpha_powers, d_mat_compressed);
    
    // Step B: Compute y_mat_reduced on HOST (it's small and fast)
    bb31_quartic_extension_t y_mat_reduced;
    for (int k = 0; k < w; ++k) {
        y_mat_reduced += h_alpha_powers[k] * h_opened_values_y[k];
    }
    
    // Step C: Compute inv_denoms on GPU
    compute_inv_denoms_kernel<<<grid_dim_h, block_dim>>>(point_z, d_coset_for_height, h, d_inv_denoms);

    // Step D: Accumulate into the main quotient polynomial buffer on GPU
    compute_quotient_main_loop_kernel<<<grid_dim_h, block_dim>>>(
        d_mat_compressed, d_inv_denoms, y_mat_reduced, alpha_pow_offset, h, d_quotient_evals_inout);
    
    // Synchronize to ensure kernel is done before we free memory
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Cleanup for this call ---
    CUDA_CHECK(cudaFree(d_lde_data));
    CUDA_CHECK(cudaFree(d_alpha_powers));
    CUDA_CHECK(cudaFree(d_mat_compressed));
    CUDA_CHECK(cudaFree(d_coset_for_height));
    CUDA_CHECK(cudaFree(d_inv_denoms));

    return 0;
}

extern "C" int fri_pcs_compute_quotient_for_height_gpu(
    // Input for a whole height group
    int h,
    const bb31_t* const* h_lde_data_ptrs, // Array of host pointers to matrix data
    const int* h_lde_widths,             // Array of matrix widths
    int num_ldes,
    
    // Global parameters
    const bb31_t* h_coset, // Global coset, we'll use the first `h` elements
    const bb31_quartic_extension_t* h_alpha,
    const bb31_quartic_extension_t* h_alpha_powers,
    int num_alpha_powers,
    
    // Flattened data for ALL openings in this height group
    const bb31_quartic_extension_t* h_points_z_flat,
    const bb31_quartic_extension_t* h_opened_values_y_flat,
    const int* h_num_points_per_mat, // Array telling how many points each matrix has
    
    // Output
    bb31_quartic_extension_t* h_quotient_evals_out
) {
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_h((h + num_threads - 1) / num_threads);

    // --- 1. Allocate all constant GPU memory ONCE ---
    bb31_quartic_extension_t* d_quotient_evals;
    CUDA_CHECK(cudaMalloc(&d_quotient_evals, (size_t)h * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMemset(d_quotient_evals, 0, (size_t)h * sizeof(bb31_quartic_extension_t)));

    bb31_quartic_extension_t* d_alpha_powers;
    CUDA_CHECK(cudaMalloc(&d_alpha_powers, (size_t)num_alpha_powers * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMemcpy(d_alpha_powers, h_alpha_powers, (size_t)num_alpha_powers * sizeof(bb31_quartic_extension_t), cudaMemcpyHostToDevice));

    bb31_t* d_coset_for_height;
    CUDA_CHECK(cudaMalloc(&d_coset_for_height, (size_t)h * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_coset_for_height, h_coset, (size_t)h * sizeof(bb31_t), cudaMemcpyHostToDevice));

    // --- 2. Iterate through matrices, applying correct logic ---
    uint64_t num_reduced_tracker = 0;
    const bb31_quartic_extension_t* current_points_z_ptr = h_points_z_flat;
    const bb31_quartic_extension_t* current_opened_values_y_ptr = h_opened_values_y_flat;

    for (int i = 0; i < num_ldes; ++i) {
        int w = h_lde_widths[i];
        int num_points = h_num_points_per_mat[i];

        // --- Per-matrix GPU work ---
        bb31_t* d_lde_data;
        CUDA_CHECK(cudaMalloc(&d_lde_data, (size_t)h * w * sizeof(bb31_t)));
        CUDA_CHECK(cudaMemcpy(d_lde_data, h_lde_data_ptrs[i], (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));
        
        bb31_quartic_extension_t* d_mat_compressed;
        CUDA_CHECK(cudaMalloc(&d_mat_compressed, (size_t)h * sizeof(bb31_quartic_extension_t)));
        compute_mat_compressed_kernel<<<grid_dim_h, block_dim>>>(d_lde_data, h, w, d_alpha_powers, d_mat_compressed);

        for (int j = 0; j < num_points; ++j) {
            // Correct alpha offset for this point
            bb31_quartic_extension_t alpha_pow_offset = h_alpha->pow(num_reduced_tracker);

            // Host-side y_mat_reduced calculation
            bb31_quartic_extension_t y_mat_reduced;
            for (int k = 0; k < w; ++k) {
                y_mat_reduced += h_alpha_powers[k] * current_opened_values_y_ptr[k];
            }
            
            bb31_quartic_extension_t point_z = *current_points_z_ptr;

            // GPU-side inv_denoms and accumulation
            bb31_quartic_extension_t* d_inv_denoms;
            CUDA_CHECK(cudaMalloc(&d_inv_denoms, (size_t)h * sizeof(bb31_quartic_extension_t)));
            compute_inv_denoms_kernel<<<grid_dim_h, block_dim>>>(point_z, d_coset_for_height, h, d_inv_denoms);
            compute_quotient_main_loop_kernel<<<grid_dim_h, block_dim>>>(
                d_mat_compressed, d_inv_denoms, y_mat_reduced, alpha_pow_offset, h, d_quotient_evals);
            CUDA_CHECK(cudaFree(d_inv_denoms));

            // IMPORTANT: Update counter and pointers
            num_reduced_tracker += w;
            current_points_z_ptr++;
            current_opened_values_y_ptr += w;
        }

        CUDA_CHECK(cudaFree(d_lde_data));
        CUDA_CHECK(cudaFree(d_mat_compressed));
    }
    
    // --- 3. Copy final result back ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_quotient_evals_out, d_quotient_evals, (size_t)h * sizeof(bb31_quartic_extension_t), cudaMemcpyDeviceToHost));
    
    // --- 4. Cleanup constant GPU memory ---
    CUDA_CHECK(cudaFree(d_quotient_evals));
    CUDA_CHECK(cudaFree(d_alpha_powers));
    CUDA_CHECK(cudaFree(d_coset_for_height));
    return 0;
}

extern "C" int fri_pcs_compute_quotient_for_height_data_in_gpu(
    // Input for a whole height group
    //int h,  =lde.height
    const GpuMatrix<bb31_t>* h_lde_data_ptrs, // Array of GpuMatrix
    //const int* h_lde_widths,             // Array of matrix widths
    int num_ldes,
    
    // Global parameters
    const GpuMatrix<bb31_t>* h_coset, // Global coset, we'll use the first `h` elements
    const bb31_quartic_extension_t* h_alpha,
    //const GpuMatrix<bb31_quartic_extension_t>* h_alpha_powers,
    const bb31_quartic_extension_t* h_alpha_powers,
    int num_alpha_powers,
    
    // Flattened data for ALL openings in this height group
    const bb31_quartic_extension_t* h_points_z_flat,
    const bb31_quartic_extension_t* h_opened_values_y_flat,
    const int* h_num_points_per_mat, // Array telling how many points each matrix has
    
    // Output
    bb31_quartic_extension_t* h_quotient_evals_out
) {
    int num_threads = 256;
    int h = h_lde_data_ptrs[0].height;
    dim3 block_dim(num_threads);
    dim3 grid_dim_h((h + num_threads - 1) / num_threads);
//printf("+++++++fri_pcs_quotient_data_in_gpu--11111,num_ldes=%u, 0.h=%u, 0.width=%u\n",num_ldes, h, h_lde_data_ptrs[0].width);
    // --- 1. Allocate all constant GPU memory ONCE ---
    bb31_quartic_extension_t* d_quotient_evals;
    CUDA_CHECK(cudaMalloc(&d_quotient_evals, (size_t)h * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMemset(d_quotient_evals, 0, (size_t)h * sizeof(bb31_quartic_extension_t)));

    bb31_quartic_extension_t* d_alpha_powers;// = h_alpha_powers->d_data;
    CUDA_CHECK(cudaMalloc(&d_alpha_powers, (size_t)num_alpha_powers * sizeof(bb31_quartic_extension_t)));
    CUDA_CHECK(cudaMemcpy(d_alpha_powers, h_alpha_powers, (size_t)num_alpha_powers * sizeof(bb31_quartic_extension_t), cudaMemcpyHostToDevice));

    bb31_t* d_coset_for_height = h_coset->d_data;
    //CUDA_CHECK(cudaMalloc(&d_coset_for_height, (size_t)h * sizeof(bb31_t)));
    //CUDA_CHECK(cudaMemcpy(d_coset_for_height, h_coset, (size_t)h * sizeof(bb31_t), cudaMemcpyHostToDevice));

    // --- 2. Iterate through matrices, applying correct logic ---
    uint64_t num_reduced_tracker = 0;
    const bb31_quartic_extension_t* current_points_z_ptr = h_points_z_flat;
    const bb31_quartic_extension_t* current_opened_values_y_ptr = h_opened_values_y_flat;

    for (int i = 0; i < num_ldes; ++i) {
        int w = h_lde_data_ptrs[i].width;
        int num_points = h_num_points_per_mat[i];

        // --- Per-matrix GPU work ---
        bb31_t* d_lde_data = h_lde_data_ptrs[i].d_data;
        //CUDA_CHECK(cudaMalloc(&d_lde_data, (size_t)h * w * sizeof(bb31_t)));
        //CUDA_CHECK(cudaMemcpy(d_lde_data, h_lde_data_ptrs[i], (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));
        
        bb31_quartic_extension_t* d_mat_compressed;
        CUDA_CHECK(cudaMalloc(&d_mat_compressed, (size_t)h * sizeof(bb31_quartic_extension_t)));
        compute_mat_compressed_kernel<<<grid_dim_h, block_dim>>>(d_lde_data, h, w, d_alpha_powers, d_mat_compressed);

        for (int j = 0; j < num_points; ++j) {
            // Correct alpha offset for this point
            bb31_quartic_extension_t alpha_pow_offset = h_alpha->pow(num_reduced_tracker);

            // Host-side y_mat_reduced calculation
            bb31_quartic_extension_t y_mat_reduced;
            for (int k = 0; k < w; ++k) {
                y_mat_reduced += h_alpha_powers[k] * current_opened_values_y_ptr[k];
            }
            
            bb31_quartic_extension_t point_z = *current_points_z_ptr;

            // GPU-side inv_denoms and accumulation
            bb31_quartic_extension_t* d_inv_denoms;
            CUDA_CHECK(cudaMalloc(&d_inv_denoms, (size_t)h * sizeof(bb31_quartic_extension_t)));
            compute_inv_denoms_kernel<<<grid_dim_h, block_dim>>>(point_z, d_coset_for_height, h, d_inv_denoms);
            compute_quotient_main_loop_kernel<<<grid_dim_h, block_dim>>>(
                d_mat_compressed, d_inv_denoms, y_mat_reduced, alpha_pow_offset, h, d_quotient_evals);
            CUDA_CHECK(cudaFree(d_inv_denoms));

            // IMPORTANT: Update counter and pointers
            num_reduced_tracker += w;
            current_points_z_ptr++;
            current_opened_values_y_ptr += w;
        }

        //CUDA_CHECK(cudaFree(d_lde_data));
        CUDA_CHECK(cudaFree(d_mat_compressed));
    }

    // --- 3. Copy final result back ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_quotient_evals_out, d_quotient_evals, (size_t)h * sizeof(bb31_quartic_extension_t), cudaMemcpyDeviceToHost));
    
    // --- 4. Cleanup constant GPU memory ---
    CUDA_CHECK(cudaFree(d_quotient_evals));
    CUDA_CHECK(cudaFree(d_alpha_powers));
    //CUDA_CHECK(cudaFree(d_coset_for_height));
    return 0;
}