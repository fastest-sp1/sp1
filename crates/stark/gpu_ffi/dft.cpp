#include <vector>
#include <cmath>
#include <cassert>
#include "bb31_t.hpp" 
#include "utils.hpp"
#include "gpu_types.hpp" 
#include "dft.hpp"

template <typename F>
__global__ void bit_reverse_rows_kernel(F* data, int h, int w, int log_h) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= h * w) return;

    // Decompose the flat index `idx` into row and column.
    int r = idx / w;
    int c = idx % w;

    int rev_r = reverse_bits(r, log_h);

    // Ensure we only swap once per pair of rows.
    if (r < rev_r) {
        // Calculate addresses for the elements to be swapped.
        size_t addr1 = (size_t)r * w + c;
        size_t addr2 = (size_t)rev_r * w + c;
        
        // Perform the swap.
        F temp = data[addr1];
        data[addr1] = data[addr2];
        data[addr2] = temp;
    }
}

template <typename F>
__global__ void bit_reverse_rows_copy_kernel(F* src_data, F* dst_data,int h, int w, int log_h) {
    // Use a 1D grid-stride loop to be robust for any size.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < h * w; 
         idx += gridDim.x * blockDim.x) 
    {
        // 1. Decompose the linear source index `idx` into its (row, col).
        int r = idx / w;
        int c = idx % w;

        // 2. Calculate the bit-reversed destination row.
        int rev_r = reverse_bits(r, log_h);

        // 3. Calculate the linear destination index.
        size_t dst_idx = (size_t)rev_r * w + c;
        
        // 4. Perform the "scatter": read from source, write to destination.
        // Each thread writes to a unique location, so there are no race conditions.
        dst_data[dst_idx] = src_data[idx];
    }
}

template <typename F>
__global__ void fft_layer_row_major_kernel(F* data, int h, int w, int layer, const F* all_twiddles) {
    // Each block processes a column.
    int c = blockIdx.x;
    if (c >= w) return;

    int m = 1 << layer;
    int m2 = 2 * m;

    // Each thread processes one butterfly pair within the column `c`.
    for (int i = threadIdx.x; i < h / 2; i += blockDim.x) {
        int k = (i / m) * m2;
        int j = i % m;

        size_t addr1 = (size_t)(k + j) * w + c;
        size_t addr2 = (size_t)(k + j + m) * w + c;

        F twiddle = all_twiddles[j * (h / m2)];
        
        F u = data[addr1];
        F v = data[addr2] * twiddle;

        data[addr1] = u + v;
        data[addr2] = u - v;
    }
}


template <typename F>
__global__ void scale_by_inv_h_kernel(F* data, int h, int w) {
    __shared__ F sh_h_inv;

    // Let the first thread of each block (threadIdx.x == 0) do the calculation.
    if (threadIdx.x == 0) {
        F h_val(F::to_monty(h)); // Use static `to_monty` for clarity
        sh_h_inv = h_val.reciprocal();
    }

    __syncthreads();

    // The grid-stride loop remains the same.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= h * w) return;
    
    // All threads in the block now read the SAME value `sh_h_inv` from fast shared memory.
    data[idx] = data[idx] * sh_h_inv;
}


template <typename F>
__global__ void apply_coset_shift_kernel(F* data, int h, int w, F shift) {
    // We use a 1D grid of threads that covers the entire matrix portion to be shifted.
    // Each thread handles ONE element. This is the simplest and most robust parallel model.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // We only want to shift the original `h*w` coefficients, not the padding.
    if (idx >= h * w) return;

    // Decompose the flat index `idx` into row `r` and column `c`.
    int r = idx / w;
    //int c = idx % w;
    
    // Calculate the power for the current row `r`.
    // Each thread calculates its required power independently.
    F shift_power = shift ^ (uint32_t)r;

    // Apply the shift to the single element this thread is responsible for.
    data[idx] = data[idx] * shift_power;
}

// --- KERNEL 1: Efficient Matrix Transpose ---
// Uses shared memory tiling to perform an out-of-place transpose.
#define TILE_DIM 32 // A common tile dimension. Must be a multiple of warp size (32).

template <typename F>
__global__ void transpose_kernel(const F* in, F* out, int h, int w) {
    // Shared memory tile. +1 on the second dimension to avoid shared memory bank conflicts.
    __shared__ F tile[TILE_DIM][TILE_DIM + 1];

    // Calculate global indices for the input matrix (x, y)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load a tile from global `in` memory to shared `tile` memory
    if (x < w && y < h) {
        tile[threadIdx.y][threadIdx.x] = in[(size_t)y * w + x];
    }
    __syncthreads(); // Ensure the entire tile is loaded

    // Calculate global indices for the output matrix (x becomes y, y becomes x)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write the transposed tile from shared `tile` memory to global `out` memory
    if (x < h && y < w) {
        out[(size_t)y * h + x] = tile[threadIdx.x][threadIdx.y];
    }
}



template <typename F>
__global__ void fused_row_lde_kernel_global_mem(
    const F* transposed_in, // Input: w x h matrix (rows are original columns)
    F* transposed_out,      // Output: w x lde_h matrix
    F* temp_buffer,         // A temporary buffer of size lde_h per column
    int h, int w, int log_h,
    int lde_h, int log_lde_h, F shift,
    const F* inv_twiddles, const F* fwd_twiddles)
{
    // Each block handles one row of the transposed matrix (an original column)
    int c = blockIdx.x; 
    if (c >= w) return;
    
    const F* row_in = transposed_in + (size_t)c * h;
    F* row_out = transposed_out + (size_t)c * lde_h;
    F* temp_row = temp_buffer + (size_t)c * lde_h; // Each column gets its own temp space

    // --- Stage A: IDFT ---
    // A.1: Load one row into the temporary global buffer. This is a coalesced read.
    for(int i = threadIdx.x; i < h; i += blockDim.x) {
        temp_row[i] = row_in[i];
    }
    // No __syncthreads() needed yet, as we are writing to distinct parts of temp_row.
    // However, it is good practice to sync after a data loading phase.
    __syncthreads();


    // A.2: Perform DIT IFFT (bit_rev -> layers -> bit_rev -> scale) on `temp_row`.
    // bit-reverse
    for(int i = threadIdx.x; i < h; i += blockDim.x) {
        int rev_i = reverse_bits(i, log_h);
        if (i < rev_i) { F temp = temp_row[i]; temp_row[i] = temp_row[rev_i]; temp_row[rev_i] = temp; }
    }
    __syncthreads();

    // butterfly layers
    for (int layer = 0; layer < log_h; ++layer) {
        int m = 1 << layer, m2 = 2 * m;
        for (int j = threadIdx.x; j < h / 2; j += blockDim.x) {
            int k = (j / m) * m2;
            int j_in_block = j % m;
            F twiddle = inv_twiddles[j_in_block * (h / m2)];
            int idx1 = k + j_in_block;
            int idx2 = idx1 + m;
            F u = temp_row[idx1], v = temp_row[idx2] * twiddle;
            temp_row[idx1] = u + v;
            temp_row[idx2] = u - v;
        }
        __syncthreads(); // Crucial: sync after each layer is complete
    }

    // bit-reverse again for natural order
    /*for(int i = threadIdx.x; i < h; i += blockDim.x) {
        int rev_i = reverse_bits(i, log_h);
        if (i < rev_i) { F temp = temp_row[i]; temp_row[i] = temp_row[rev_i]; temp_row[rev_i] = temp; }
    }
    __syncthreads(); */
    
    // scale
    F h_inv = F(F::to_monty(h)).reciprocal();
    for(int i = threadIdx.x; i < h; i += blockDim.x) {
        temp_row[i] = temp_row[i] * h_inv;
    }
    // `temp_row` (first h elements) now contains natural order coefficients.

    // --- Stage B: Pad and Shift ---
    // Pad the rest of the temporary buffer with zeros.
    for(int i = threadIdx.x; i < lde_h; i += blockDim.x) {
        if (i >= h) temp_row[i] = F(0);
    }
    __syncthreads();

    // Apply shift to all lde_h coefficients.
    if (shift != F(F::to_monty(1))) {
        for(int i = threadIdx.x; i < lde_h; i += blockDim.x) {
            temp_row[i] = temp_row[i] * (shift ^ (uint32_t)i);
        }
        __syncthreads();
    }

    // --- Stage C: Forward DFT ---
    // Perform DIT FFT (bit_rev -> layers) on the full `temp_row` buffer.
    // Result will be natural order LDE evaluations.
    for(int i = threadIdx.x; i < lde_h; i += blockDim.x) {
        int rev_i = reverse_bits(i, log_lde_h);
        if (i < rev_i) { F temp = temp_row[i]; temp_row[i] = temp_row[rev_i]; temp_row[rev_i] = temp; }
    }
    __syncthreads();
    
    for (int layer = 0; layer < log_lde_h; ++layer) {
        int m = 1 << layer, m2 = 2 * m;
        for (int j = threadIdx.x; j < lde_h / 2; j += blockDim.x) {
             int k = (j / m) * m2;
             int j_in_block = j % m;
             F twiddle = fwd_twiddles[j_in_block * (lde_h / m2)];
             int idx1 = k + j_in_block;
             int idx2 = idx1 + m;
             F u = temp_row[idx1], v = temp_row[idx2] * twiddle;
             temp_row[idx1] = u + v;
             temp_row[idx2] = u - v;
        }
        __syncthreads();
    }

    // --- Stage D: Write final result to output buffer ---
    // The final output should be bit-reversed to match Radix2DitParallel
    for(int i = threadIdx.x; i < lde_h; i += blockDim.x) {
        //row_out[reverse_bits(i, log_lde_h)] = temp_row[i];
        row_out[i] = temp_row[i];
    }
}


//////////// FFI
extern "C" int fast_dft_batch_gpu(//pass
    bb31_t* data, int h, int w, const bb31_t* forward_twiddles)
{
    if (h == 0 || w == 0) return 0;
    int log_h = integer_log2(h);
    if (log_h == -1) return -1; // Error if not power of 2

    bb31_t *d_data, *d_fwd_twiddles;
    size_t total_size_bytes = (size_t)h * w * sizeof(bb31_t);
    CUDA_CHECK(cudaMalloc(&d_data, total_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_fwd_twiddles, (size_t)h / 2 * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_data, data, total_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fwd_twiddles, forward_twiddles, (size_t)h / 2 * sizeof(bb31_t), cudaMemcpyHostToDevice));

    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_flat((h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols(w);

    // Plonky3's `dft_batch` flow: bit_rev -> layers -> bit_rev.
    // The final data in memory is bit-reversed.
    
    // 1. In-place bit-reverse the input data.
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat, block_dim>>>(d_data, h, w, log_h);
    
    // 2. Apply butterfly layers. This produces natural order output.
    for (int layer = 0; layer < log_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols, block_dim>>>(d_data, h, w, layer, d_fwd_twiddles);
    }
    
    // 3. Final bit-reversal to match Plonky3's expected memory layout.
    //bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat, block_dim>>>(d_data, h, w, log_h);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_data, total_size_bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_fwd_twiddles));
    return 0;
}


extern "C" int fast_idft_gpu(//pass
   bb31_t* data,
    int h,
    int w,
    const bb31_t* inverse_twiddles) 
{
    if (h == 0 || w == 0) return 0;
    
    // --- 1. Setup ---
    init_two_adic_generators();
    int log_h = integer_log2(h);

    // --- 2. GPU Memory & Data Transfer ---
    bb31_t *d_data, *d_inv_twiddles;
    size_t total_size_bytes = h * w * sizeof(bb31_t);
    CUDA_CHECK(cudaMalloc(&d_data, total_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_inv_twiddles, (size_t)h / 2 * sizeof(bb31_t)));
    
    CUDA_CHECK(cudaMemcpy(d_data, data, (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv_twiddles, inverse_twiddles, (size_t)h / 2 * sizeof(bb31_t), cudaMemcpyHostToDevice));
    
    int num_threads = 256;
    dim3 block_dim(num_threads);

    // === Stage 1: IDFT to get coefficients in NATURAL order ===
    // This is the verified DIT IFFT flow.
    dim3 grid_dim_flat_h((h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_h(w);
    
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_h, block_dim>>>(d_data, h, w, log_h);
    for (int layer = 0; layer < log_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_h, block_dim>>>(d_data, h, w, layer, d_inv_twiddles);
    }

    scale_by_inv_h_kernel<bb31_t><<<grid_dim_flat_h, block_dim>>>(d_data, h, w);


    // The result in `d_temp` is now the final natural order coefficients.
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_data, total_size_bytes, cudaMemcpyDeviceToHost));
    
    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_inv_twiddles));
    
    return 0;
}

//v2
//this function is faster than op_fast_coset_lde_batch_gpu
extern "C" int fast_coset_lde_batch_gpu(//pass
    bb31_t* data, 
    int h, 
    int w, 
    int added_bits, 
    bb31_t shift,
    const bb31_t* inverse_twiddles, 
    const bb31_t* forward_twiddles_lde)
{
    if (h == 0 || w == 0) return 0;
    
    // --- 1. Setup ---
    init_two_adic_generators();
    int log_h = integer_log2(h);
    int log_lde_h = log_h + added_bits;
    size_t lde_h = 1 << log_lde_h;
    
    // --- 2. GPU Memory & Data Transfer ---
    bb31_t *d_data, *d_inv_twiddles, *d_fwd_twiddles;
    size_t lde_size_bytes = lde_h * w * sizeof(bb31_t);
    CUDA_CHECK(cudaMalloc(&d_data, lde_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_inv_twiddles, (size_t)h / 2 * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_fwd_twiddles, (size_t)lde_h / 2 * sizeof(bb31_t)));
    
    CUDA_CHECK(cudaMemcpy(d_data, data, (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv_twiddles, inverse_twiddles, (size_t)h / 2 * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fwd_twiddles, forward_twiddles_lde, (size_t)lde_h / 2 * sizeof(bb31_t), cudaMemcpyHostToDevice));
    
    int num_threads = 256;
    dim3 block_dim(num_threads);

    // === Stage 1: IDFT to get coefficients in NATURAL order ===
    // This is the verified DIT IFFT flow.
    dim3 grid_dim_flat_h((h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_h(w);
    
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_h, block_dim>>>(d_data, h, w, log_h);
    for (int layer = 0; layer < log_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_h, block_dim>>>(d_data, h, w, layer, d_inv_twiddles);
    }

    scale_by_inv_h_kernel<bb31_t><<<grid_dim_flat_h, block_dim>>>(d_data, h, w); //the result is ok.(=idft)

   /* //test
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_data, h * w * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    printf("---CUDA coeffs:");
    for(int i=0; i< h*w; ++i){
        printf(" %u ", data[i].as_canonical_u32());
    }*/


    // === Stage 2: Zero-Padding ===
    // This must happen BEFORE the shift, as the shift applies to padded coefficients.
    if (added_bits > 0) {
        CUDA_CHECK(cudaMemset((char*)d_data + (size_t)h * w * sizeof(bb31_t), 0, (lde_h - h) * w * sizeof(bb31_t)));
    }
    
    // === Stage 3: Apply Coset Shift to PADDED, NATURAL order coefficients ===
    if (shift != bb31_t(bb31_t::to_monty(1))) {
        // We now launch enough threads to cover the entire LDE buffer.
        dim3 grid_dim_flat_ldeh((lde_h * w + num_threads - 1) / num_threads);
        apply_coset_shift_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(d_data, lde_h, w, shift);
    }

    // === Stage 4: Forward DFT on the large buffer to get NATURAL order evaluations ===
    // This is a standard DIT FFT flow.
    dim3 grid_dim_flat_ldeh((lde_h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_ldeh(w);

    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(d_data, lde_h, w, log_lde_h);
    for (int layer = 0; layer < log_lde_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_ldeh, block_dim>>>(d_data, lde_h, w, layer, d_fwd_twiddles);
    }
    //bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(d_data, lde_h, w, log_lde_h);
    
    // The result is now in NATURAL order, matching NaiveDft's output.
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_data, lde_size_bytes, cudaMemcpyDeviceToHost));
    
    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_inv_twiddles));
    CUDA_CHECK(cudaFree(d_fwd_twiddles));
    
    return 0;
}

//use GpuMatrix
extern "C" int fast_coset_lde_batch_data_in_gpu(
    const GpuMatrix<bb31_t>* d_input_ptr, 
    GpuMatrix<bb31_t>* d_lde_ptr,  
    int h, 
    int w, 
    int added_bits, 
    bb31_t shift,
    const bb31_t* inverse_twiddles, 
    const bb31_t* forward_twiddles_lde)
{
    if (h == 0 || w == 0) return 0;

    //debug
    if (d_input_ptr == nullptr) {
        fprintf(stderr, "ERROR: input_matrix_ptr is null!\n");
        return -1; // Or some error code
    }
    if (d_lde_ptr == nullptr) {
        fprintf(stderr, "ERROR: lde_matrix_ptr is null!\n");
        return -1;
    }
    
    //printf("Host-side check: input_matrix_ptr->width = %zu, height = %zu, ptr = %p\n",
    //    d_input_ptr->width, d_input_ptr->height, d_input_ptr->d_data);
    //printf("Host-side check: lde_matrix_ptr->width = %zu, height = %zu, ptr = %p\n",
    //    d_lde_ptr->width, d_lde_ptr->height, d_lde_ptr->d_data);

    if (d_input_ptr->d_data == nullptr || d_lde_ptr->d_data == nullptr) {
         fprintf(stderr, "ERROR: One of the device pointers (d_data) is null!\n");
         return -1;
    }
    
    // --- 1. Setup ---
    init_two_adic_generators();
    int log_h = integer_log2(h);
    int log_lde_h = log_h + added_bits;
    size_t lde_h = 1 << log_lde_h;
    
    // --- 2. GPU Memory & Data Transfer ---
    bb31_t  *d_inv_twiddles, *d_fwd_twiddles;
    //size_t lde_size_bytes = lde_h * w * sizeof(bb31_t);
    //CUDA_CHECK(cudaMalloc(&d_data, lde_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_inv_twiddles, (size_t)h / 2 * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_fwd_twiddles, (size_t)lde_h / 2 * sizeof(bb31_t)));
    
//CUDA_CHECK(cudaMemcpy(d_data, data, (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lde_ptr->d_data, d_input_ptr->d_data, d_input_ptr->width * d_input_ptr->height * sizeof(bb31_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv_twiddles, inverse_twiddles, (size_t)h / 2 * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fwd_twiddles, forward_twiddles_lde, (size_t)lde_h / 2 * sizeof(bb31_t), cudaMemcpyHostToDevice));
    
    int num_threads = 256;
    dim3 block_dim(num_threads);

    // === Stage 1: IDFT to get coefficients in NATURAL order ===
    // This is the verified DIT IFFT flow.
    dim3 grid_dim_flat_h((h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_h(w);
    
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_h, block_dim>>>(d_lde_ptr->d_data, h, w, log_h);
    for (int layer = 0; layer < log_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_h, block_dim>>>(d_lde_ptr->d_data, h, w, layer, d_inv_twiddles);
    }

    scale_by_inv_h_kernel<bb31_t><<<grid_dim_flat_h, block_dim>>>(d_lde_ptr->d_data, h, w); //the result is ok.(=idft)

   /* //test
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_data, h * w * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    printf("---CUDA coeffs:");
    for(int i=0; i< h*w; ++i){
        printf(" %u ", data[i].as_canonical_u32());
    }*/


    // === Stage 2: Zero-Padding ===
    // This must happen BEFORE the shift, as the shift applies to padded coefficients.
    if (added_bits > 0) {
        CUDA_CHECK(cudaMemset((char*)d_lde_ptr->d_data + (size_t)h * w * sizeof(bb31_t), 0, (lde_h - h) * w * sizeof(bb31_t)));
    }
    
    // === Stage 3: Apply Coset Shift to PADDED, NATURAL order coefficients ===
    if (shift != bb31_t(bb31_t::to_monty(1))) {
        // We now launch enough threads to cover the entire LDE buffer.
        dim3 grid_dim_flat_ldeh((lde_h * w + num_threads - 1) / num_threads);
        apply_coset_shift_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(d_lde_ptr->d_data, lde_h, w, shift);
    }

    // === Stage 4: Forward DFT on the large buffer to get NATURAL order evaluations ===
    // This is a standard DIT FFT flow.
    dim3 grid_dim_flat_ldeh((lde_h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_ldeh(w);

    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(d_lde_ptr->d_data, lde_h, w, log_lde_h);
    for (int layer = 0; layer < log_lde_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_ldeh, block_dim>>>(d_lde_ptr->d_data, lde_h, w, layer, d_fwd_twiddles);
    }
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(d_lde_ptr->d_data, lde_h, w, log_lde_h);
  
    CUDA_CHECK(cudaDeviceSynchronize());
     
    // --- Cleanup ---
    //CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_inv_twiddles));
    CUDA_CHECK(cudaFree(d_fwd_twiddles));
    
    return 0;
}

extern "C" int bit_reverse_rows_gpu(
    const GpuMatrix<bb31_t>* d_input_ptr, 
    GpuMatrix<bb31_t>* d_out_ptr
   )
{    
    size_t h  = d_out_ptr->height;
    int log_h = integer_log2(h);
    int w =d_out_ptr->width;
    
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_flat_ldeh((h * w + num_threads - 1) / num_threads);

    //CUDA_CHECK(cudaMemcpy(d_out_ptr->d_data, d_input_ptr->d_data,  w * h * sizeof(bb31_t), cudaMemcpyDeviceToDevice));
    //bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(d_out_ptr->d_data, h, w, log_h);

    bit_reverse_rows_copy_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(d_input_ptr->d_data, d_out_ptr->d_data, h, w, log_h);
    CUDA_CHECK(cudaGetLastError());

    return 0;
}

extern "C" int stark_transpose_gpu(
    const bb31_t* in, bb31_t* out, int h, int w)
{
    if (h == 0 || w == 0) return 0;

    bb31_t *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)h * w * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)w * h * sizeof(bb31_t))); // Note transposed size
    CUDA_CHECK(cudaMemcpy(d_in, in, (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));

    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((w + TILE_DIM - 1) / TILE_DIM, (h + TILE_DIM - 1) / TILE_DIM);

    transpose_kernel<bb31_t><<<grid_dim, block_dim>>>(d_in, d_out, h, w);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, d_out, (size_t)w * h * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

extern "C" int op_fast_coset_lde_batch_gpu(
    bb31_t* data, 
    int h, 
    int w, 
    int added_bits, 
    bb31_t shift,
    const bb31_t* inverse_twiddles, 
    const bb31_t* forward_twiddles_lde)
{
    if (h == 0 || w == 0) return 0;
    
    int log_h = integer_log2(h);
    int log_lde_h = log_h + added_bits;
    size_t lde_h = 1 << log_lde_h;

    // --- 1. GPU Memory Allocation ---
    bb31_t *d_in, *d_out, *d_transposed_in, *d_transposed_out, *d_inv_twiddles, *d_fwd_twiddles;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)h * w * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_out, lde_h * w * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_transposed_in, (size_t)w * h * sizeof(bb31_t))); // Note transposed dimensions
    CUDA_CHECK(cudaMalloc(&d_transposed_out, (size_t)w * lde_h * sizeof(bb31_t)));
    
    CUDA_CHECK(cudaMalloc(&d_inv_twiddles, (size_t)h / 2 * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_fwd_twiddles, (size_t)lde_h / 2 * sizeof(bb31_t)));

    bb31_t *d_temp_buffer;
    CUDA_CHECK(cudaMalloc(&d_temp_buffer, (size_t)w * lde_h * sizeof(bb31_t)));

    CUDA_CHECK(cudaMemcpy(d_inv_twiddles, inverse_twiddles, (size_t)h / 2 * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fwd_twiddles, forward_twiddles_lde, (size_t)lde_h / 2 * sizeof(bb31_t), cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMemcpy(d_in, data, (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));

    // --- 2. Transpose h x w -> w x h ---
    dim3 grid_dim_t1((w + TILE_DIM - 1) / TILE_DIM, (h + TILE_DIM - 1) / TILE_DIM);
    dim3 block_dim_t(TILE_DIM, TILE_DIM);
    transpose_kernel<bb31_t><<<grid_dim_t1, block_dim_t>>>(d_in, d_transposed_in, h, w);
    CUDA_CHECK(cudaGetLastError());

    // --- 3. Launch the Fused LDE Kernel ---
   
    dim3 grid_dim(w);
    dim3 block_dim(256);
    fused_row_lde_kernel_global_mem<bb31_t><<<grid_dim, block_dim>>>(
        d_transposed_in, d_transposed_out, d_temp_buffer,
        h, w, log_h, lde_h, log_lde_h,
        shift, d_inv_twiddles, d_fwd_twiddles);


    CUDA_CHECK(cudaGetLastError());
    
    // --- 4. Transpose back w x lde_h -> lde_h x w ---
    dim3 grid_dim_t2((lde_h + TILE_DIM - 1) / TILE_DIM, (w + TILE_DIM - 1) / TILE_DIM);
    transpose_kernel<bb31_t><<<grid_dim_t2, block_dim_t>>>(d_transposed_out, d_out, w, lde_h);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // --- 5. Copy Final Result Back ---
    CUDA_CHECK(cudaMemcpy(data, d_out, lde_h * w * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    // --- 6. Cleanup ---
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_transposed_in));
    CUDA_CHECK(cudaFree(d_transposed_out));
    CUDA_CHECK(cudaFree(d_fwd_twiddles));
    CUDA_CHECK(cudaFree(d_inv_twiddles));
    return 0;
}
