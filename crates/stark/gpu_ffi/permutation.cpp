#include <cub/cub.cuh> // For CUB's DeviceScan
#include "gpu_types.hpp"
#include "bb31_t.hpp"
#include "virtual_pair_col.hpp"
#include "utils.hpp"
#include "permutation.hpp"

// Kernel to populate the "entry" columns of the permutation trace.
__global__ void populate_entries_kernel(
    const GpuMatrix<bb31_t> main_lde,
    const GpuMatrix<bb31_t> prep_lde,
    GpuMatrix<Challenge> perm_trace_out,
    const Interaction* interaction_buffer,
    int   num_interaction,
    Challenge alpha,
    Challenge beta,
    int batch_size
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= main_lde.height) return;

    const bb31_t* main_row = static_cast<const bb31_t*>(main_lde.d_data) + row_idx * main_lde.width;
    const bb31_t* prep_row = static_cast<const bb31_t*>(prep_lde.d_data) + row_idx * prep_lde.width;
    Challenge* perm_row_out = static_cast<Challenge*>(perm_trace_out.d_data) + row_idx * perm_trace_out.width;
    
    int num_chunks = (num_interaction + batch_size - 1) / batch_size;

    for (int i = 0; i < num_chunks; ++i) {
        //const Challenge& entry = perm_local[i];
        Challenge chunk_sum = Challenge::zero();
        int start_idx = i * batch_size;
        // chunk_len calculation is safe because we know the interactions exist due to the loop bound.
        int chunk_len = min(batch_size, num_interaction - start_idx);

          
        // 1. Collect RLCs and multiplicities for the current chunk.
        // This part of the logic remains unchanged.
        for (int j = 0; j < chunk_len; ++j) {
            const auto& interaction = interaction_buffer[start_idx + j];
            Challenge rlc = calculate_rlc(interaction, alpha, beta, main_row, prep_row);
              
            Val mult_val = interaction.multiplicity.apply(main_row, prep_row);
            Challenge mult = Challenge(mult_val) * (interaction.is_send ? Challenge::one() : Challenge(Val::neg_one()));
            chunk_sum = chunk_sum + mult * rlc.reciprocal();
        }

        perm_row_out[i] = chunk_sum; 
    }
   

}

// Kernel to sum the entry columns for each row into a temporary buffer.
__global__ void sum_entry_columns_kernel(
    const GpuMatrix<Challenge> perm_trace,
    Challenge* d_row_sums // Output buffer of size perm_trace.height
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= perm_trace.height) return;

    const Challenge* perm_row = static_cast<const Challenge*>(perm_trace.d_data) + row_idx * perm_trace.width;
    int num_entries = perm_trace.width - 1;

    Challenge sum = Challenge::zero();
    for (int i = 0; i < num_entries; ++i) {
        sum = sum + perm_row[i];
    }
    d_row_sums[row_idx] = sum;
}

// Kernel to write the final prefix sum back into the last column (phi column).
__global__ void write_phi_column_kernel(
    const Challenge* d_prefix_sums,
    GpuMatrix<Challenge> perm_trace
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= perm_trace.height) return;

    Challenge* perm_row = static_cast<Challenge*>(perm_trace.d_data) + row_idx * perm_trace.width;
    int phi_col_idx = perm_trace.width - 1;

    perm_row[phi_col_idx] = d_prefix_sums[row_idx];
}


extern "C" int generate_permutation_trace_gpu(
    int chip_id,
    const GpuMatrix<bb31_t>* main_lde,
    const GpuMatrix<bb31_t>* prep_lde,
    GpuMatrix<Challenge>* perm_trace_out,
    const Challenge* h_random_elements, // From HOST
    int batch_size,
    Challenge* h_local_cumulative_sum_out // To HOST
) {
    Interaction *interaction_buffer;
    int interaction_count=0;
    switch (chip_id) {
        case ChipId::SELECT:
            interaction_count = 5;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_select_interactions(interaction_buffer);
            break;
        case ChipId::BASE_ALU:
            interaction_count = 12;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_base_alu_interactions(interaction_buffer);
            break;
        case ChipId::EXT_ALU:
            interaction_count = 12;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_ext_alu_interactions(interaction_buffer);
            break;
        case ChipId::BATCH_FRI:
            interaction_count = 4;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_batch_fri_interactions(interaction_buffer);
            break;
        case ChipId::EXP_REVERSE_BITS:
            interaction_count = 3;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_exp_reverse_fri_interactions(interaction_buffer);
            break;
        case ChipId::FRI_FOLD:
            interaction_count = 10;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_fri_fold_interactions(interaction_buffer);
            break;

        case ChipId::PUBLIC_VALUES:
            interaction_count = 1;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_public_values_interactions(interaction_buffer);
            break;
        case ChipId::P2_WIDE:
            interaction_count = POSEIDON2_STATE_WIDTH * 2;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_p2_wide_interactions(interaction_buffer);
            break;
        case ChipId::P2_SKINNY:
            interaction_count = POSEIDON2_STATE_WIDTH;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_p2_skinny_interactions(interaction_buffer);
            break;

        case ChipId::MEM_CONST:
            interaction_count = NUM_CONST_MEM_ENTRIES_PER_ROW;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_memory_const_interactions(interaction_buffer);
            break;

        case ChipId::MEM_VAR:
            interaction_count = NUM_VAR_MEM_ENTRIES_PER_ROW;
            interaction_buffer = (Interaction *)malloc(interaction_count * sizeof(struct Interaction));
            if (interaction_buffer == NULL) {         
                printf("interaction_buffer: Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            get_memory_var_interactions(interaction_buffer);
            break;

        default:
            printf("exit: permutation does not support chip_id=%u\n", chip_id);
            exit(EXIT_FAILURE);
            break;
    }
    Interaction *d_interaction;
    CUDA_CHECK(cudaMalloc(&d_interaction, interaction_count * sizeof(Interaction)));
    CUDA_CHECK(cudaMemcpy(d_interaction, interaction_buffer, interaction_count * sizeof(Interaction), cudaMemcpyHostToDevice));

    int num_threads = 256;
    dim3 grid_pop((main_lde->height + num_threads - 1) / num_threads);
    populate_entries_kernel<<<grid_pop, num_threads>>>(
        *main_lde, 
        *prep_lde, 
        *perm_trace_out,
        d_interaction,
        interaction_count,
        h_random_elements[0], 
        h_random_elements[1], 
        batch_size
    );

    // === 4. Compute Cumulative Sum (Phi Column) using CUB ===
    Challenge* d_row_sums;
    Challenge* d_prefix_sums;
    CUDA_CHECK(cudaMalloc(&d_row_sums, main_lde->height * sizeof(Challenge)));
    CUDA_CHECK(cudaMalloc(&d_prefix_sums, main_lde->height * sizeof(Challenge)));

    CUDA_CHECK(cudaDeviceSynchronize());
    // a) Sum the entry columns for each row
    sum_entry_columns_kernel<<<grid_pop, num_threads>>>(*perm_trace_out, d_row_sums);
    
    // b) Use CUB for efficient device-wide inclusive scan (prefix sum)
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_row_sums, d_prefix_sums, main_lde->height);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_row_sums, d_prefix_sums, main_lde->height);
    
     // Check for any async errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // c) Write the phi column
    write_phi_column_kernel<<<grid_pop, num_threads>>>(d_prefix_sums, *perm_trace_out);

    // === 5. Copy Final Cumulative Sum Back to Host ===
    if (main_lde->height > 0) {
        CUDA_CHECK(cudaMemcpy(h_local_cumulative_sum_out, d_prefix_sums + (main_lde->height - 1), sizeof(Challenge), cudaMemcpyDeviceToHost));
    }

    // === 6. Cleanup ===
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_row_sums));
    CUDA_CHECK(cudaFree(d_prefix_sums));
    free(interaction_buffer);
   
    return 0; // Success
}


