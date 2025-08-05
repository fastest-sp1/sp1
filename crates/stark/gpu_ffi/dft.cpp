#include <vector>
#include <cmath>
#include <cassert>
#include "bb31_t.hpp" 
#include "utils.hpp"
//#include "bb31_constants.hpp" 


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


// It works on pre-allocated device pointers and runs on a specific stream.
void run_single_lde_from_evals_async(
    const bb31_t* d_evals_in,      // Input: device pointer to evaluations
    bb31_t* d_lde_out,             // Output: device pointer for LDE results
    int h, int w, int log_h,
    int lde_h, int log_lde_h, bb31_t shift,
    const bb31_t* d_inv_twiddles,
    const bb31_t* d_fwd_twiddles,
    cudaStream_t stream
) {
    // We use `d_lde_out` as our scratch space. It must be large enough for the LDE.
    bb31_t* d_temp_buffer = d_lde_out;

    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_flat_h(((size_t)h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_h(w);

    // Copy the input evals into the temp buffer to start.
    CUDA_CHECK(cudaMemcpyAsync(d_temp_buffer, d_evals_in, (size_t)h * w * sizeof(bb31_t), cudaMemcpyDeviceToDevice, stream));

    // === Stage 1: IDFT to get coefficients (in-place on `d_temp_buffer`) ===
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_h, block_dim, 0, stream>>>(d_temp_buffer, h, w, log_h);
    for (int layer = 0; layer < log_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_h, block_dim, 0, stream>>>(d_temp_buffer, h, w, layer, d_inv_twiddles);
    }
    scale_by_inv_h_kernel<bb31_t><<<grid_dim_flat_h, block_dim, 0, stream>>>(d_temp_buffer, h, w);
    
    // `d_temp_buffer` now contains coefficients up to h*w.
    
    // === Stage 2: Pad with zeros and apply shift (in-place) ===
    if (lde_h > h) {
        CUDA_CHECK(cudaMemsetAsync(d_temp_buffer + (size_t)h * w, 0, (size_t)(lde_h - h) * w * sizeof(bb31_t), stream));
    }
    if (shift != bb31_t(bb31_t::to_monty(1))) {
        dim3 grid_dim_shift(((size_t)lde_h * w + num_threads - 1) / num_threads);
        apply_coset_shift_kernel<bb31_t><<<grid_dim_shift, block_dim, 0, stream>>>(d_temp_buffer, lde_h, w, shift);
    }

    // === Stage 3: Forward DFT to get final LDEs (in-place) ===
    dim3 grid_dim_flat_lde(((size_t)lde_h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_lde(w);
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_lde, block_dim, 0, stream>>>(d_temp_buffer, lde_h, w, log_lde_h);
    for (int layer = 0; layer < log_lde_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_lde, block_dim, 0, stream>>>(d_temp_buffer, lde_h, w, layer, d_fwd_twiddles);
    }
    // Final bit reversal to match Radix2DitParallel
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_lde, block_dim, 0, stream>>>(d_temp_buffer, lde_h, w, log_lde_h);
}


//sync
void run_single_lde_from_evals(
    const bb31_t* d_evals_in,      // Input: device pointer to evaluations
    bb31_t* d_lde_out,             // Output: device pointer for LDE results
    int h, int w, int log_h,
    int lde_h, int log_lde_h, bb31_t shift,
    const bb31_t* d_inv_twiddles,
    const bb31_t* d_fwd_twiddles
) {
    // We use `d_lde_out` as our scratch space. It must be large enough for the LDE.
    bb31_t* d_temp_buffer = d_lde_out;

    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_flat_h(((size_t)h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_h(w);

    // Copy the input evals into the temp buffer to start.
    CUDA_CHECK(cudaMemcpy(d_temp_buffer, d_evals_in, (size_t)h * w * sizeof(bb31_t), cudaMemcpyDeviceToDevice));

    // === Stage 1: IDFT to get coefficients (in-place on `d_temp_buffer`) ===
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_h, block_dim, 0>>>(d_temp_buffer, h, w, log_h);
    for (int layer = 0; layer < log_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_h, block_dim, 0>>>(d_temp_buffer, h, w, layer, d_inv_twiddles);
    }
    scale_by_inv_h_kernel<bb31_t><<<grid_dim_flat_h, block_dim, 0>>>(d_temp_buffer, h, w);
    
    // `d_temp_buffer` now contains coefficients up to h*w.
    
    // === Stage 2: Pad with zeros and apply shift (in-place) ===
    if (lde_h > h) {
        CUDA_CHECK(cudaMemset(d_temp_buffer + (size_t)h * w, 0, (size_t)(lde_h - h) * w * sizeof(bb31_t)));
    }
    if (shift != bb31_t(bb31_t::to_monty(1))) {
        dim3 grid_dim_shift(((size_t)lde_h * w + num_threads - 1) / num_threads);
        apply_coset_shift_kernel<bb31_t><<<grid_dim_shift, block_dim, 0>>>(d_temp_buffer, lde_h, w, shift);
    }

    // === Stage 3: Forward DFT to get final LDEs (in-place) ===
    dim3 grid_dim_flat_lde(((size_t)lde_h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_lde(w);
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_lde, block_dim, 0>>>(d_temp_buffer, lde_h, w, log_lde_h);
    for (int layer = 0; layer < log_lde_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_lde, block_dim, 0>>>(d_temp_buffer, lde_h, w, layer, d_fwd_twiddles);
    }
    // Final bit reversal to match Radix2DitParallel
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_lde, block_dim, 0>>>(d_temp_buffer, lde_h, w, log_lde_h);
    CUDA_CHECK(cudaDeviceSynchronize());

}

//async
// The FFI function that orchestrates the batch.
extern "C" int dft_batch_lde_on_gpu(
    const bb31_t* h_evals_flat,
    size_t total_evals_elements,
    const int* h_poly_info,         // Array of [h, w, log_blowup] triples
    const bb31_t* h_shifts,
    int num_polys,

    // ================== Twiddles for BOTH directions ==================
    const bb31_t* h_all_inv_twiddles,
    const int* h_inv_twiddle_offsets,
    const bb31_t* h_all_fwd_twiddles,
    const int* h_fwd_twiddle_offsets,
    // ===============================================================

    // Outputs
    void** d_ldes_flat_out,
    size_t* total_lde_elements_out
) {
    if (num_polys == 0) {
        *d_ldes_flat_out = nullptr;
        *total_lde_elements_out = 0;
        return 0;
    }

    // --- 1. Calculate output size and other metadata ---
    std::vector<size_t> lde_element_counts;
    size_t total_lde_elements = 0;
    int max_log_h = 0;
    int max_log_lde_h = 0;
    std::vector<int> log_hs;
    std::vector<int> log_lde_hs;

    for (int i = 0; i < num_polys; ++i) {
        int h = h_poly_info[i * 3 + 0];
        int w = h_poly_info[i * 3 + 1];
        int log_blowup = h_poly_info[i * 3 + 2];
        int log_h = integer_log2(h);
        int log_lde_h = log_h + log_blowup;

        log_hs.push_back(log_h);
        log_lde_hs.push_back(log_lde_h);
        if (log_h > max_log_h) max_log_h = log_h;
        if (log_lde_h > max_log_lde_h) max_log_lde_h = log_lde_h;
        
        size_t lde_size = (size_t)(1 << log_lde_h) * w;
        lde_element_counts.push_back(lde_size);
        total_lde_elements += lde_size;
    }

    *total_lde_elements_out = total_lde_elements;
    CUDA_CHECK(cudaMalloc(d_ldes_flat_out, total_lde_elements * sizeof(bb31_t)));

    // --- 2. Upload all inputs to GPU ---
    bb31_t* d_evals_flat;
    CUDA_CHECK(cudaMalloc(&d_evals_flat, total_evals_elements * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_evals_flat, h_evals_flat, total_evals_elements * sizeof(bb31_t), cudaMemcpyHostToDevice));

    // ================== COMPLETE TWIDDLE HANDLING ==================
    bb31_t* d_all_inv_twiddles;
    size_t total_inv_twiddles = h_inv_twiddle_offsets[max_log_h + 1];
    CUDA_CHECK(cudaMalloc(&d_all_inv_twiddles, total_inv_twiddles * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_all_inv_twiddles, h_all_inv_twiddles, total_inv_twiddles * sizeof(bb31_t), cudaMemcpyHostToDevice));

    bb31_t* d_all_fwd_twiddles;
    size_t total_fwd_twiddles = h_fwd_twiddle_offsets[max_log_lde_h + 1];
    CUDA_CHECK(cudaMalloc(&d_all_fwd_twiddles, total_fwd_twiddles * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_all_fwd_twiddles, h_all_fwd_twiddles, total_fwd_twiddles * sizeof(bb31_t), cudaMemcpyHostToDevice));
    // ===============================================================

    // --- 3. Create streams and launch tasks ---
    std::vector<cudaStream_t> streams(num_polys);
    for (int i = 0; i < num_polys; ++i) cudaStreamCreate(&streams[i]);

    size_t current_eval_offset = 0;
    size_t current_lde_offset = 0;
    for (int i = 0; i < num_polys; ++i) {
        int h = h_poly_info[i * 3 + 0];
        int w = h_poly_info[i * 3 + 1];
        int log_h = log_hs[i];
        int log_lde_h = log_lde_hs[i];
        size_t lde_h = 1 << log_lde_h;
        bb31_t shift = h_shifts[i];
        
        const bb31_t* d_current_evals = d_evals_flat + current_eval_offset;
        bb31_t* d_current_ldes = (bb31_t*)*d_ldes_flat_out + current_lde_offset;

        // Get correct twiddle pointers for this specific size
        const bb31_t* d_inv_twiddles_for_h = d_all_inv_twiddles + h_inv_twiddle_offsets[log_h];
        const bb31_t* d_fwd_twiddles_for_lde_h = d_all_fwd_twiddles + h_fwd_twiddle_offsets[log_lde_h];

        // Launch the helper function on this polynomial's stream
        run_single_lde_from_evals_async(
            d_current_evals,
            d_current_ldes,
            h, w, log_h,
            lde_h, log_lde_h, shift,
            d_inv_twiddles_for_h,
            d_fwd_twiddles_for_lde_h,
            streams[i]
        );

        current_eval_offset += (size_t)h * w;
        current_lde_offset += lde_element_counts[i];
    }

    // --- 4. Sync and cleanup ---
    for (int i = 0; i < num_polys; ++i) cudaStreamDestroy(streams[i]);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_evals_flat));
    CUDA_CHECK(cudaFree(d_all_inv_twiddles));
    CUDA_CHECK(cudaFree(d_all_fwd_twiddles));
    // Do NOT free `*d_ldes_flat_out`, it's the return value.
    
    return 0;
}

//async
// The FFI function that orchestrates the batch.
extern "C" int dft_batch_lde_on_gpu_sync(
    const bb31_t* h_evals_flat,
    size_t total_evals_elements,
    const int* h_poly_info,         // Array of [h, w, log_blowup] triples
    const bb31_t* h_shifts,
    int num_polys,

    // ================== Twiddles for BOTH directions ==================
    const bb31_t* h_all_inv_twiddles,
    const int* h_inv_twiddle_offsets,
    const bb31_t* h_all_fwd_twiddles,
    const int* h_fwd_twiddle_offsets,
    // ===============================================================

    // Outputs
    void** d_ldes_flat_out,
    size_t* total_lde_elements_out
) {
    if (num_polys == 0) {
        *d_ldes_flat_out = nullptr;
        *total_lde_elements_out = 0;
        return 0;
    }

    // --- 1. Calculate output size and other metadata ---
    std::vector<size_t> lde_element_counts;
    size_t total_lde_elements = 0;
    int max_log_h = 0;
    int max_log_lde_h = 0;
    std::vector<int> log_hs;
    std::vector<int> log_lde_hs;

    for (int i = 0; i < num_polys; ++i) {
        int h = h_poly_info[i * 3 + 0];
        int w = h_poly_info[i * 3 + 1];
        int log_blowup = h_poly_info[i * 3 + 2];
        int log_h = integer_log2(h);
        int log_lde_h = log_h + log_blowup;

        log_hs.push_back(log_h);
        log_lde_hs.push_back(log_lde_h);
        if (log_h > max_log_h) max_log_h = log_h;
        if (log_lde_h > max_log_lde_h) max_log_lde_h = log_lde_h;
        
        size_t lde_size = (size_t)(1 << log_lde_h) * w;
        lde_element_counts.push_back(lde_size);
        total_lde_elements += lde_size;
    }

    *total_lde_elements_out = total_lde_elements;
    CUDA_CHECK(cudaMalloc(d_ldes_flat_out, total_lde_elements * sizeof(bb31_t)));

    // --- 2. Upload all inputs to GPU ---
    bb31_t* d_evals_flat;
    CUDA_CHECK(cudaMalloc(&d_evals_flat, total_evals_elements * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_evals_flat, h_evals_flat, total_evals_elements * sizeof(bb31_t), cudaMemcpyHostToDevice));

    // ================== COMPLETE TWIDDLE HANDLING ==================
    bb31_t* d_all_inv_twiddles;
    size_t total_inv_twiddles = h_inv_twiddle_offsets[max_log_h + 1];
    CUDA_CHECK(cudaMalloc(&d_all_inv_twiddles, total_inv_twiddles * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_all_inv_twiddles, h_all_inv_twiddles, total_inv_twiddles * sizeof(bb31_t), cudaMemcpyHostToDevice));

    bb31_t* d_all_fwd_twiddles;
    size_t total_fwd_twiddles = h_fwd_twiddle_offsets[max_log_lde_h + 1];
    CUDA_CHECK(cudaMalloc(&d_all_fwd_twiddles, total_fwd_twiddles * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_all_fwd_twiddles, h_all_fwd_twiddles, total_fwd_twiddles * sizeof(bb31_t), cudaMemcpyHostToDevice));
    // ===============================================================

    // --- 3. Create streams and launch tasks ---
    //std::vector<cudaStream_t> streams(num_polys);
    //for (int i = 0; i < num_polys; ++i) cudaStreamCreate(&streams[i]);

    size_t current_eval_offset = 0;
    size_t current_lde_offset = 0;
    for (int i = 0; i < num_polys; ++i) {
        int h = h_poly_info[i * 3 + 0];
        int w = h_poly_info[i * 3 + 1];
        int log_h = log_hs[i];
        int log_lde_h = log_lde_hs[i];
        size_t lde_h = 1 << log_lde_h;
        bb31_t shift = h_shifts[i];
        
        const bb31_t* d_current_evals = d_evals_flat + current_eval_offset;
        bb31_t* d_current_ldes = (bb31_t*)*d_ldes_flat_out + current_lde_offset;

        // Get correct twiddle pointers for this specific size
        const bb31_t* d_inv_twiddles_for_h = d_all_inv_twiddles + h_inv_twiddle_offsets[log_h];
        const bb31_t* d_fwd_twiddles_for_lde_h = d_all_fwd_twiddles + h_fwd_twiddle_offsets[log_lde_h];

        // Launch the helper function on this polynomial's stream
        run_single_lde_from_evals(
            d_current_evals,
            d_current_ldes,
            h, w, log_h,
            lde_h, log_lde_h, shift,
            d_inv_twiddles_for_h,
            d_fwd_twiddles_for_lde_h
        );

        current_eval_offset += (size_t)h * w;
        current_lde_offset += lde_element_counts[i];
    }

    // --- 4.cleanup ---
    
    CUDA_CHECK(cudaFree(d_evals_flat));
    CUDA_CHECK(cudaFree(d_all_inv_twiddles));
    CUDA_CHECK(cudaFree(d_all_fwd_twiddles));
    // Do NOT free `*d_ldes_flat_out`, it's the return value.
    
    return 0;
}
