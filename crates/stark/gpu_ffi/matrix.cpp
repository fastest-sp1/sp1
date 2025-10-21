#pragma once
#include "gpu_types.hpp"
//#include "bb31_t.hpp" 
#include "utils.hpp"

//1D
__global__ void flatten_kernel(
     const GpuMatrix<Challenge> in_matrix, 
    GpuMatrix<bb31_t> out_matrix 
) {
    // Total number of Challenge elements to process
    size_t num_elements = (size_t)in_matrix.width * in_matrix.height;
    
    // Use a 1D grid-stride loop to iterate over all elements.
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < num_elements; 
         i += gridDim.x * blockDim.x) 
    {
        // Calculate the 2D (row, col) index from the 1D linear index `i`.
        int row = i / in_matrix.width;
        int col = i % in_matrix.width;

        // Pointers to the start of the current row (same logic as before)
        const Challenge* in_row_ptr = static_cast<const Challenge*>(in_matrix.d_data) + (size_t)row * in_matrix.width;
        bb31_t* out_row_ptr = static_cast<bb31_t*>(out_matrix.d_data) + (size_t)row * out_matrix.width;
    
        // Get the input Challenge element
        const Challenge& in_element = in_row_ptr[col];
    
        // Write the 4 base field coefficients to the output matrix
        out_row_ptr[col * 4 + 0] = in_element.coeffs[0];
        out_row_ptr[col * 4 + 1] = in_element.coeffs[1];
        out_row_ptr[col * 4 + 2] = in_element.coeffs[2];
        out_row_ptr[col * 4 + 3] = in_element.coeffs[3];
    }
}

// in matrix_kernels.cuh

__global__ void split_by_stride_kernel(
    const GpuMatrix<bb31_t> in_matrix,
    GpuMatrix<bb31_t>* d_out_matrices, // Array of output matrix descriptors (ON DEVICE)
    int num_chunks
) {
    // Each thread is responsible for one element in one of the output chunks.
    // Calculate the 1D global thread ID.
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Get the dimensions of a single output chunk (they are all the same size).
    const int out_chunk_height = in_matrix.height / num_chunks;
    const int chunk_width = in_matrix.width;
    const size_t chunk_size_elements = (size_t)out_chunk_height * chunk_width;
    
    // Check if the thread is out of the total bounds of all chunks.
    if (global_tid >= num_chunks * chunk_size_elements) {
        return;
    }

    // Determine which chunk this thread belongs to.
    int chunk_idx = global_tid / chunk_size_elements;
    
    // Determine the 1D index of the element WITHIN its chunk.
    int local_idx = global_tid % chunk_size_elements;
    
    // Convert the local 1D index to a 2D (row, col) index.
    int out_row = local_idx / chunk_width;
    int col = local_idx % chunk_width;

    // Get the descriptor for the correct output matrix.
    const GpuMatrix<bb31_t>& out_matrix = d_out_matrices[chunk_idx];

    // Calculate the source row in the large input matrix.
    int in_row = out_row * num_chunks + chunk_idx;

    // Get pointers to the source and destination elements.
    const bb31_t* in_ptr = static_cast<const bb31_t*>(in_matrix.d_data);
    bb31_t* out_ptr = static_cast<bb31_t*>(out_matrix.d_data);

    // Perform the single element copy.
    out_ptr[(size_t)out_row * chunk_width + col] = in_ptr[(size_t)in_row * chunk_width + col];
}


extern "C" int split_matrix_gpu(
    const GpuMatrix<bb31_t>* in_matrix,
    const GpuMatrix<bb31_t>* h_out_matrices, 
    int num_chunks
) {
    // Copy the output matrix descriptors to the device
    GpuMatrix<bb31_t>* d_out_matrices;
    CUDA_CHECK(cudaMalloc(&d_out_matrices, num_chunks * sizeof(GpuMatrix<bb31_t>)));
    CUDA_CHECK(cudaMemcpy(d_out_matrices, h_out_matrices, num_chunks * sizeof(GpuMatrix<bb31_t>), cudaMemcpyHostToDevice));

    // Calculate the TOTAL number of elements across ALL output chunks.
    size_t out_chunk_height = in_matrix->height / num_chunks;
    size_t out_chunk_width = in_matrix->width;
    size_t total_output_elements = num_chunks * out_chunk_height * out_chunk_width;
    
    if (total_output_elements == 0) return 0;
    
    // Launch a simple, large 1D grid.
    int threads_per_block = 256;
    int blocks_per_grid = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    dim3 grid_dim(blocks_per_grid);
    dim3 block_dim(threads_per_block);

    split_by_stride_kernel<<<grid_dim, block_dim>>>(*in_matrix, d_out_matrices, num_chunks);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_out_matrices));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return 0; // Success
}

extern "C" int flatten_permutation_trace_gpu(
    const GpuMatrix<Challenge>* in_matrix,
    GpuMatrix<Val>* out_matrix
) {
    size_t num_elements = (size_t)in_matrix->width * in_matrix->height;
    if (num_elements == 0) return 0;
    
    // Launch a simple, large 1D grid.
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    
    dim3 grid_dim(blocks_per_grid); //  1D grid
    dim3 block_dim(threads_per_block);

    flatten_kernel<<<grid_dim, block_dim>>>(*in_matrix, *out_matrix);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return 0; // Success
}
