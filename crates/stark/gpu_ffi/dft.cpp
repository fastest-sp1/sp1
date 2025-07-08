#include <vector>
#include <cmath>
#include <cassert>
#include "bb31_t.hpp" 
#include "utils.hpp"
#include "bb31_constants.hpp" 

inline int integer_log2(int n) {
    if (n <= 0) return -1;
    int log = 0;
    while ((1 << log) < n) { log++; }
    if ((1 << log) != n) return -1; // Ensure power of 2
    return log;
}

// --- Kernel Implementations (Optimized) ---
__device__ inline unsigned int reverse_bits(unsigned int v, int log_n) {
    unsigned int r = 0;
    for (int i = 0; i < log_n; i++) {
        if ((v >> i) & 1) {
            r |= 1 << (log_n - 1 - i);
        }
    }
    return r;
}

#define TILE_DIM 32 // Use a 32x32 tile for transposition
//op
template <typename F>
__global__ void transpose_kernel(const F* in, F* out, int h, int w) {
    __shared__ F tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int block_x = blockIdx.x * TILE_DIM + threadIdx.x;
    int block_y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (block_x < w && block_y < h) {
        tile[threadIdx.y][threadIdx.x] = in[block_y * w + block_x];
    }
    __syncthreads();

    // The threads now write from the transposed position in the tile.
    block_x = blockIdx.y * TILE_DIM + threadIdx.x;
    block_y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (block_x < h && block_y < w) {
        out[block_y * h + block_x] = tile[threadIdx.x][threadIdx.y];
    }
}

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

// This version uses shared memory to handle the non-contiguous column data.
template <typename F>
__global__ void fft_layer_row_major_optimized(
    F* data, int h, int w, int layer, const F* all_twiddles) 
{
    int c = blockIdx.x; // Each block handles a column
    if (c >= w) return;
    
    // Shared memory to hold ONE entire column.
    extern __shared__ F sh_col[];

    // 1. Load the non-contiguous column `c` into contiguous shared memory.
    for (int r = threadIdx.x; r < h; r += blockDim.x) {
        sh_col[r] = data[(size_t)r * w + c];
    }
    __syncthreads();

    // 2. Perform the butterfly operations entirely within fast shared memory.
    // This part is now operating on contiguous data.
    int m = 1 << layer;
    int m2 = 2 * m;
    for (int i = threadIdx.x; i < h / 2; i += blockDim.x) {
        int k = (i / m) * m2;
        int j = i % m;
        
        F twiddle = all_twiddles[j * (h / m2)];
        
        int idx1 = k + j;
        int idx2 = k + j + m;

        F u = sh_col[idx1];
        F v = sh_col[idx2] * twiddle;

        sh_col[idx1] = u + v;
        sh_col[idx2] = u - v;
    }
    __syncthreads();

    // 3. Write the results from shared memory back to the non-contiguous global memory.
    for (int r = threadIdx.x; r < h; r += blockDim.x) {
        data[(size_t)r * w + c] = sh_col[r];
    }
}

// --- THE NEW OPTIMIZED KERNEL for FFT Layers ---
// This version uses tiling to process very tall columns with limited shared memory.
#define FFT_TILE_HEIGHT 1024 // A tile size that fits comfortably in shared memory.
                             // 1024 * 4 bytes = 4 KB. This should be tunable.

template <typename F>
__global__ void fft_layer_row_major_optimized_kernel(
    F* data, int h, int w, int layer, const F* all_twiddles) 
{
    int c = blockIdx.x; // Each block handles a column
    if (c >= w) return;
    
    // Shared memory for one TILE of a column.
    extern __shared__ F sh_tile[];

    int m = 1 << layer;
    int m2 = 2 * m;

    // The butterfly operations for a given layer are local within blocks of size m2.
    // We only need to load one such block (or part of it) into shared memory at a time.
    
    // If the butterfly block size is smaller than our tile size, we can do many in parallel.
    if (m2 <= FFT_TILE_HEIGHT) {
        // This loop iterates over the tiles of the column.
        for (int tile_start_row = 0; tile_start_row < h; tile_start_row += FFT_TILE_HEIGHT) {
            
            // 1. Load a tile from global memory into shared memory.
            for (int i = threadIdx.x; i < FFT_TILE_HEIGHT; i += blockDim.x) {
                int r_global = tile_start_row + i;
                if (r_global < h) {
                    sh_tile[i] = data[(size_t)r_global * w + c];
                }
            }
            __syncthreads();

            // 2. Perform butterflies within the tile in shared memory.
            for (int k = 0; k < FFT_TILE_HEIGHT; k += m2) {
                 if ((k / m2) % blockDim.x == threadIdx.x) { // Distribute blocks among threads
                    for (int j = 0; j < m; j++) {
                        int r_global = tile_start_row + k + j;
                        F twiddle = all_twiddles[ (r_global % m) * (h / m2) ]; // Careful with twiddle index
                        
                        int idx1_sh = k + j;
                        int idx2_sh = idx1_sh + m;

                        F u = sh_tile[idx1_sh];
                        F v = sh_tile[idx2_sh] * twiddle;

                        sh_tile[idx1_sh] = u + v;
                        sh_tile[idx2_sh] = u - v;
                    }
                 }
            }
            __syncthreads();

            // 3. Write the updated tile back to global memory.
            for (int i = threadIdx.x; i < FFT_TILE_HEIGHT; i += blockDim.x) {
                int r_global = tile_start_row + i;
                if (r_global < h) {
                    data[(size_t)r_global * w + c] = sh_tile[i];
                }
            }
            __syncthreads();
        }
    } else { // Butterfly block size m2 > TILE_HEIGHT
        // This case is more complex. The butterfly partners are in different tiles.
        // We revert to the unoptimized global memory version for these final, large layers.
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



//v1 ok
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


template <typename F>
__global__ void naive_dft_kernel(//pass
    const F* mat,   // Input matrix (natural order coefficients)
    F* res,         // Output matrix (natural order evaluations)
    int h, int w,
    F g             // The two-adic generator for size h
) {
    // Each block computes one output element res[res_r][c]
    int c = blockIdx.x;
    int res_r = blockIdx.y;

    if (c >= w || res_r >= h) return;
    
    // Shared memory for parallel reduction (summation) within the block
    extern __shared__ F sh_sum[];

    // Initialize shared memory for this thread
    sh_sum[threadIdx.x] = F(0);

    F point = g ^ (uint32_t)res_r; // point = g^res_r
    
    // Each thread in the block computes a partial sum
    // The loop iterates over all input rows `src_r`
    for (int src_r = threadIdx.x; src_r < h; src_r += blockDim.x) {
        F point_power = point ^ (uint32_t)src_r; // (g^res_r)^src_r
        
        size_t mat_addr = (size_t)src_r * w + c;
        
        sh_sum[threadIdx.x] = sh_sum[threadIdx.x] + point_power * mat[mat_addr];
    }
    __syncthreads();

    // Perform parallel reduction in shared memory to get the final sum
    // This is a standard CUDA pattern.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_sum[threadIdx.x] = sh_sum[threadIdx.x] + sh_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 of the block writes the final result
    if (threadIdx.x == 0) {
        size_t res_addr = (size_t)res_r * w + c;
        res[res_addr] = sh_sum[0];
    }
}

// Kernel to perform the final row swapping for naive IDFT
template <typename F>
__global__ void naive_idft_swap_kernel(F* data, int h, int w) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // We only need to cover the first half of the rows
    if (idx >= (h / 2) * w) return;
    
    int r = idx / w;
    int c = idx % w;

    // Only process rows from 1 to h/2 - 1
    if (r > 0) {
        int r_swap = h - r;
        
        size_t addr1 = (size_t)r * w + c;
        size_t addr2 = (size_t)r_swap * w + c;

        F temp = data[addr1];
        data[addr1] = data[addr2];
        data[addr2] = temp;
    }
}


// --- FFI implementation for dft_batch ---
extern "C" int naive_dft_gpu(//pass
    bb31_t* data, int h, int w)
{
    if (h == 0 || w == 0) return 0;
    int log_h = integer_log2(h);
    if (log_h == -1) return -1; // Error if not power of 2

    bb31_t *d_data, *d_evals_nat;
    size_t total_size_bytes = (size_t)h * w * sizeof(bb31_t);
    CUDA_CHECK(cudaMalloc(&d_data, total_size_bytes));
    //CUDA_CHECK(cudaMalloc(&d_fwd_twiddles, (size_t)h  * sizeof(bb31_t)));//radix2 len: h/2 *  sizeof(bb31_t)
     CUDA_CHECK(cudaMalloc(&d_evals_nat, h * w * sizeof(bb31_t)));

    CUDA_CHECK(cudaMemcpy(d_data, data, total_size_bytes, cudaMemcpyHostToDevice));
    //CUDA_CHECK(cudaMemcpy(d_fwd_twiddles, forward_twiddles, (size_t)h  * sizeof(bb31_t), cudaMemcpyHostToDevice));

    int num_threads = 256;
    bb31_t g = two_adic_generator(log_h);
    dim3 grid_dim_2d(w, h);
    dim3 block_dim(num_threads);

    size_t shmem_size = num_threads * sizeof(bb31_t);
    naive_dft_kernel<bb31_t><<<grid_dim_2d, block_dim, shmem_size>>>(
        d_data, d_evals_nat, h, w, g);


    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_evals_nat, total_size_bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_evals_nat));
    //CUDA_CHECK(cudaFree(d_fwd_twiddles));
    return 0;
}

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


// --- FFI implementation for idft_batch ---
extern "C" int naive_idft_gpu( //pass
   bb31_t* data,
    int h,
    int w,
    const bb31_t* inverse_twiddles) 
{
    init_two_adic_generators();

    if (h == 0 || w == 0) return 0;
    int log_h = integer_log2(h);

    // --- GPU Memory Setup ---
    bb31_t *d_data, *d_temp;
    size_t total_size_bytes = (size_t)h * w * sizeof(bb31_t);
    CUDA_CHECK(cudaMalloc(&d_data, total_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, total_size_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data, total_size_bytes, cudaMemcpyHostToDevice));

    // --- Stage 1: Perform a Naive FORWARD DFT ---
    bb31_t g = two_adic_generator(log_h);
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_2d(w, h);
    size_t shmem_size = num_threads * sizeof(bb31_t);

    naive_dft_kernel<bb31_t><<<grid_dim_2d, block_dim, shmem_size>>>(
        d_data, d_temp, h, w, g); //the result is as same as cpu !
    // Result `d_temp` is now natural order evaluations from the forward DFT.

    // --- Stage 2: Scale by 1/h ---
    dim3 grid_dim_flat((h * w + num_threads - 1) / num_threads);
    scale_by_inv_h_kernel<bb31_t><<<grid_dim_flat, block_dim>>>(d_temp, h, w);


    // --- Stage 3: Swap rows ---
    // The grid should cover the first half of the rows.
    dim3 grid_dim_flat_half(((h / 2) * w + num_threads - 1) / num_threads);
    naive_idft_swap_kernel<bb31_t><<<grid_dim_flat_half, block_dim>>>(d_temp, h, w);

    // The result in `d_temp` is now the final natural order coefficients.
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_temp, total_size_bytes, cudaMemcpyDeviceToHost));
    
    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_temp));
    
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


extern "C" int naive_coset_dft_gpu(//pass
    bb31_t* data, int h, int w, bb31_t shift)
{
    if (h == 0 || w == 0) return 0;
    int log_h = integer_log2(h);
    if (log_h == -1) return -1; // Error if not power of 2

    bb31_t *d_data, *d_evals_nat;
    size_t total_size_bytes = (size_t)h * w * sizeof(bb31_t);
    CUDA_CHECK(cudaMalloc(&d_data, total_size_bytes));
    //CUDA_CHECK(cudaMalloc(&d_fwd_twiddles, (size_t)h  * sizeof(bb31_t)));//radix2 len: h/2 *  sizeof(bb31_t)
     CUDA_CHECK(cudaMalloc(&d_evals_nat, h * w * sizeof(bb31_t)));

    CUDA_CHECK(cudaMemcpy(d_data, data, total_size_bytes, cudaMemcpyHostToDevice));
  
    //1. coset_shift_cols(&mut mat, shift)
    int num_threads = 256;
    int num_blocks = (h * w + num_threads - 1) / num_threads;
    dim3 grid_dim(num_blocks);
    dim3 block_dim(num_threads);

    apply_coset_shift_kernel<bb31_t><<<grid_dim, block_dim>>>(d_data, h, w, shift);


  ///2. dft_batch
    bb31_t g = two_adic_generator(log_h);
    dim3 grid_dim_2d(w, h);
    //dim3 block_dim(num_threads);

    size_t shmem_size = num_threads * sizeof(bb31_t);
    naive_dft_kernel<bb31_t><<<grid_dim_2d, block_dim, shmem_size>>>(
        d_data, d_evals_nat, h, w, g);


    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_evals_nat, total_size_bytes, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_evals_nat));
    //CUDA_CHECK(cudaFree(d_fwd_twiddles));
    return 0;
}

//v1 ok
extern "C" int naive_coset_lde_batch_gpu(//pass
    bb31_t* data,
    int h,
    int w,
    int added_bits,
    bb31_t shift) 
{
    init_two_adic_generators();

    if (h == 0 || w == 0) return 0;
    int log_h = integer_log2(h);
    int log_lde_h = log_h + added_bits;
    size_t lde_h = 1 << log_lde_h;
    
    // --- GPU Memory & Data Transfer ---
    bb31_t *d_data, *d_evals_nat;
    size_t lde_size_bytes = lde_h * w * sizeof(bb31_t);
    CUDA_CHECK(cudaMalloc(&d_data, lde_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_evals_nat, lde_size_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data, (size_t)h * w * sizeof(bb31_t), cudaMemcpyHostToDevice));
      

    //Stage 1: idft
    // --- Stage 1.1: Perform a Naive FORWARD DFT ---
    bb31_t g = two_adic_generator(log_h);
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_2d(w, h);
    size_t shmem_size = num_threads * sizeof(bb31_t);

    naive_dft_kernel<bb31_t><<<grid_dim_2d, block_dim, shmem_size>>>(
        d_data, d_evals_nat, h, w, g); //the result is as same as cpu !
    // Result `d_temp` is now natural order evaluations from the forward DFT.

    // --- Stage 1.2: Scale by 1/h ---
    dim3 grid_dim_flat((h * w + num_threads - 1) / num_threads);
    scale_by_inv_h_kernel<bb31_t><<<grid_dim_flat, block_dim>>>(d_evals_nat, h, w);


    // --- Stage 1.3: Swap rows ---
    // The grid should cover the first half of the rows.
    dim3 grid_dim_flat_half(((h / 2) * w + num_threads - 1) / num_threads);
    naive_idft_swap_kernel<bb31_t><<<grid_dim_flat_half, block_dim>>>(d_evals_nat, h, w); //the above is ok


    // === Stage 2: Zero-Padding ===
    // We pad the buffer that now contains natural-order coefficients.
    if (added_bits > 0) {
        CUDA_CHECK(cudaMemset((char*)d_evals_nat + (size_t)h * w * sizeof(bb31_t), 0, (lde_h - h) * w * sizeof(bb31_t)));
    }
    
    // === Stage 3: Apply Coset Shift to PADDED, NATURAL order coefficients ===
    // The NaiveDft logic applies shift AFTER padding.
    // The shift should be applied to all `lde_h` coefficients, but since c_k=0 for k>=h,
    // we only need to compute for the first `h` rows' worth of coefficients.
    //3. coset_shift_cols(&mut mat, shift)
    //int num_threads = 256;
    int num_blocks = (h * w + num_threads - 1) / num_threads;
    dim3 grid_dim(num_blocks);

    apply_coset_shift_kernel<bb31_t><<<grid_dim, block_dim>>>(d_evals_nat, lde_h, w, shift);


  ///4. dft_batch
    bb31_t g_lde = two_adic_generator(log_lde_h);
    dim3 grid_dim_2d_4(w, lde_h);
    //dim3 block_dim(num_threads);

    //size_t shmem_size = num_threads * sizeof(bb31_t);
    naive_dft_kernel<bb31_t><<<grid_dim_2d_4, block_dim, shmem_size>>>(
        d_evals_nat, d_data, lde_h, w, g_lde);

   /* // 4b: Apply butterfly layers.
    for (int layer = 0; layer < log_lde_h; ++layer) {
        fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_ldeh, block_dim>>>(d_data, lde_h, w, layer, d_fwd_twiddles);
        CUDA_CHECK(cudaGetLastError());
    }*/


    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_data, lde_size_bytes, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_evals_nat));
    
    return 0;
}

//v2
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
    

    // === Stage 1: IDFT to get coefficients in NATURAL order ===
    // This is the verified DIT IFFT flow.
    int num_threads = 256;
    dim3 block_dim_256(num_threads);
    dim3 grid_dim_flat_h((h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_h(w);
    
    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_h, block_dim_256>>>(d_data, h, w, log_h);
    
    //optimize
    //dim3 block_dim(FFT_BLOCK_SIZE);
    size_t shmem_size = FFT_TILE_HEIGHT * sizeof(bb31_t);
    for (int layer = 0; layer < log_h; ++layer) {
        // Use the new optimized kernel
        fft_layer_row_major_optimized_kernel<bb31_t><<<grid_dim_cols_h, block_dim_256, shmem_size>>>(
            d_data, h, w, layer, d_inv_twiddles);
    }
    //

    scale_by_inv_h_kernel<bb31_t><<<grid_dim_flat_h, block_dim_256>>>(d_data, h, w); //the result is ok.(=idft)

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
        apply_coset_shift_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim_256>>>(d_data, lde_h, w, shift);
    }

    // === Stage 4: Forward DFT on the large buffer to get NATURAL order evaluations ===
    // This is a standard DIT FFT flow.
    dim3 grid_dim_flat_ldeh((lde_h * w + num_threads - 1) / num_threads);
    dim3 grid_dim_cols_ldeh(w);

    bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim_256>>>(d_data, lde_h, w, log_lde_h);
    

    //for (int layer = 0; layer < log_lde_h; ++layer) {
    //    fft_layer_row_major_kernel<bb31_t><<<grid_dim_cols_ldeh, block_dim>>>(d_data, lde_h, w, layer, d_fwd_twiddles);
    //}

    size_t lde_shmem_size = FFT_TILE_HEIGHT * sizeof(bb31_t); // Assuming lde_h also fits tile-wise
    if (lde_h < FFT_TILE_HEIGHT) { // For very small final transforms
        lde_shmem_size = lde_h * sizeof(bb31_t);
    }

    for (int layer = 0; layer < log_lde_h; ++layer) {
        // Use the new optimized kernel again
        fft_layer_row_major_optimized_kernel<bb31_t><<<grid_dim_cols_ldeh, block_dim_256, lde_shmem_size>>>(
            d_data, lde_h, w, layer, d_fwd_twiddles);
    }
       
    // The result is now in NATURAL order, matching NaiveDft's output.
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(data, d_data, lde_size_bytes, cudaMemcpyDeviceToHost));
    
    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_inv_twiddles));
    CUDA_CHECK(cudaFree(d_fwd_twiddles));
    
    return 0;
}