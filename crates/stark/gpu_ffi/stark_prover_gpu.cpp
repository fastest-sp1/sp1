// file: stark_prover_gpu.cpp
#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "bb31_t.hpp"
#include "bb31_quartic_extension_t.hpp"
#include "permutation_eval.hpp"
#include "base_alu_eval.hpp"
#include "ext_alu_eval.hpp"
#include "batch_fri_eval.hpp"
#include "exp_reverse_bits_eval.hpp"
#include "fri_fold_eval.hpp"
#include "public_values_eval.hpp"
#include "select_eval.hpp"
#include "poseidon2_wide_eval.hpp"
#include "poseidon2_skinny_eval.hpp"
#include "gpu_types.hpp"
#include "air_folder.hpp"
#include "selectors.hpp"

#include "utils.hpp"

using namespace sp1_recursion_core_sys;

// =================================================================// ===
// These must be large enough for the widest chip.
#define MAX_MAIN_COLS 48 // ExtAlu has 48 columns
#define MAX_PREP_COLS 32 // BaseAlu has 32 columns
#define MAX_PERM_WIDTH_EF 16 // ExtAlu has 13, so 16 is a safe max

__constant__ Val d_public_values_digest[DIGEST_SIZE];
//__constant__ SepticDigest_bb31 d_global_cumulative_sum;
__constant__ unsigned char d_global_cumulative_sum_bytes[sizeof(SepticDigest_bb31)];

__global__ void quotient_values_kernel(
    int chip_id,

    // Trace Data
    const Val* d_main_trace,
    int main_width,
    const Val* d_prep_trace,
    int prep_width,
    const Val* d_perm_trace,
    int perm_width,
    int main_trace_height,
    // Domain & Selector Data
    int q_size,
    const Val* d_inv_z,
    const Val* d_is_first_row,
    const Val* d_is_last_row,
    const Val* d_is_transition,
    int next_step,
    int batch_size,
    // Challenge Data
    const Challenge* d_powers_of_alpha,
    int alpha_offset,
    const Challenge* d_perm_challenges,
    const Challenge* d_local_cumulative_sum,

    // Output
    Challenge* d_out
){
    
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= q_size) return;

    //debug
    //if (row_idx > 0 ) return;

    // --- Calculate the index for the 'next' row WITH WRAPPING ---
    int next_row_idx = (row_idx + next_step) % q_size;
   
    const Val* main_row = d_main_trace + (size_t)row_idx * main_width;
    const Val* main_next_row = d_main_trace + ((row_idx + next_step) % main_trace_height) * main_width;

    const Val* prep_row = d_prep_trace + (size_t)row_idx * prep_width;
    const Val* prep_next_row = d_prep_trace + ((row_idx + next_step) % main_trace_height) * prep_width;

    const Val* perm_row = d_perm_trace + (size_t)row_idx * perm_width;
    const Val* perm_next_row = d_perm_trace + (size_t)next_row_idx * perm_width;


    Challenge accumulator = Challenge::zero();
    int constraint_idx = 0;
    const SepticDigest_bb31* d_global_cumulative_sum = 
        reinterpret_cast<const SepticDigest_bb31*>(d_global_cumulative_sum_bytes);
    ProverConstraintFolder<Challenge> folder = {&accumulator, d_powers_of_alpha, &constraint_idx, alpha_offset, d_global_cumulative_sum};
    

    // --- CHIP-SPECIFIC DISPATCH ---
    switch (chip_id) {
        case ChipId::BASE_ALU:
            eval_base_alu_chip(
                folder, 
                main_row,
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
        
            );
            break;
        case ChipId::EXT_ALU:
             eval_ext_alu_chip(
                folder, 
                main_row, 
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
            );
            break;
        case ChipId::BATCH_FRI:
             eval_batch_fri_chip(
                folder, 
                main_row, 
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
            );
            break;
        case ChipId::EXP_REVERSE_BITS:
            eval_exp_reverse_bits_chip(
                folder, 
                main_row, 
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
            );
            break;
        case ChipId::FRI_FOLD:
            eval_fri_fold_chip(
                folder, 
                main_row, 
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
            );
            break;
        case ChipId::PUBLIC_VALUES:
            //const Val* public_values = d_public_values + (size_t)row_idx * DIGEST_SIZE;
            eval_public_values_chip(
                folder, 
                main_row, 
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_public_values_digest,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
                );
            break;
        case ChipId::SELECT:
            eval_select_chip(
                folder, 
                main_row, 
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
            );
            break;
        case ChipId::P2_WIDE:
            eval_poseidon2_wide_chip(
                folder, 
                main_row, 
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
            );
            break;
        case ChipId::P2_SKINNY:
            eval_poseidon2_skinny_chip(
                folder, 
                main_row, 
                main_next_row,
                prep_row,
                prep_next_row,
                perm_row, 
                perm_next_row, 
                perm_width,
                batch_size,
                d_perm_challenges, 
                *d_local_cumulative_sum,
                d_is_first_row[row_idx], 
                d_is_last_row[row_idx], 
                d_is_transition[row_idx]
            );
            break;
    }

    d_out[row_idx] = accumulator * Challenge(d_inv_z[row_idx]);
}


// =================================================================
//        HELPER AND FFI FUNCTIONS
// =================================================================
extern "C" int quotient_values_gpu(
    int chip_id,
    const Val* h_main_trace,
    int main_width,
    int main_height,
    const Val* h_prep_trace,
    int prep_width,
    const Challenge* h_powers_of_alpha, 
    int num_constraints,
    int quotient_domain_size, 
    //const Val* h_inv_zerofier,
    const Val* h_perm_trace, 
    int perm_width,
    int next_step,
    int batch_size,
    const Challenge* h_perm_challenges,
    const Challenge* h_local_cumulative_sum,
    const Val* h_public_values_digest, 
    int public_values_digest_len,
    const SepticDigest<bb31_t>* h_global_cumulative_sum,
    int alpha_offset,
    int trace_log_size,
    int coset_log_size,
    const Val trace_subgroup_generator,
    const Val coset_shift,
    const Val coset_subgroup_generator,
    Challenge* h_quotient_values_out
) {
    //if (num_public_values > MAX_PUBLIC_VALUES) return -3;
    Val *d_main_trace, *d_prep_trace, *d_inv_zerofier, *d_perm_trace;
    Val *d_is_first_row, *d_is_last_row, *d_is_transition, *d_inv_vanishing;
    Challenge *d_powers_of_alpha, *d_quotient_values_out, *d_perm_challenges, *d_local_cumulative_sum;

    size_t main_size = (size_t)quotient_domain_size * main_width * sizeof(Val);
    size_t prep_size = (size_t)quotient_domain_size * prep_width * sizeof(Val);
    size_t alpha_size = (size_t)num_constraints * sizeof(Challenge);
    size_t inv_z_size = (size_t)quotient_domain_size * sizeof(Val);
    size_t perm_size = (size_t)quotient_domain_size * perm_width * sizeof(Val);
    size_t quotient_size = (size_t)quotient_domain_size * sizeof(Challenge);

    CUDA_CHECK(cudaMalloc(&d_main_trace, main_size));
    CUDA_CHECK(cudaMalloc(&d_prep_trace, prep_size));
    CUDA_CHECK(cudaMalloc(&d_powers_of_alpha, alpha_size));
    CUDA_CHECK(cudaMalloc(&d_perm_trace, perm_size));
    CUDA_CHECK(cudaMalloc(&d_perm_challenges, 2 * sizeof(Challenge)));
    CUDA_CHECK(cudaMalloc(&d_local_cumulative_sum, sizeof(Challenge)));
    CUDA_CHECK(cudaMalloc(&d_quotient_values_out, quotient_size));

    CUDA_CHECK(cudaMalloc(&d_is_first_row, (size_t)quotient_domain_size * sizeof(Val)));
    CUDA_CHECK(cudaMalloc(&d_is_last_row, (size_t)quotient_domain_size * sizeof(Val)));
    CUDA_CHECK(cudaMalloc(&d_is_transition, (size_t)quotient_domain_size * sizeof(Val)));
    CUDA_CHECK(cudaMalloc(&d_inv_vanishing, (size_t)quotient_domain_size * sizeof(Val)));

    generate_selectors_on_device(
        trace_log_size, 
        trace_subgroup_generator,
        coset_log_size, 
        coset_shift, 
        coset_subgroup_generator,
        d_is_first_row, d_is_last_row, d_is_transition, // Pass null for selectors we don't need yet
        d_inv_vanishing
    );

    CUDA_CHECK(cudaMemcpy(d_main_trace, h_main_trace, main_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prep_trace, h_prep_trace, prep_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_powers_of_alpha, h_powers_of_alpha, alpha_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_perm_trace, h_perm_trace, perm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_perm_challenges, h_perm_challenges, 2 * sizeof(Challenge), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_cumulative_sum, h_local_cumulative_sum, sizeof(Challenge), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_global_cumulative_sum_bytes, 
        h_global_cumulative_sum, 
        sizeof(SepticDigest_bb31)  //Notice it
    ));

    if (chip_id == ChipId::PUBLIC_VALUES) {
        CUDA_CHECK(cudaMemcpyToSymbol(d_public_values_digest, h_public_values_digest, DIGEST_SIZE * sizeof(Val)));
    }

    int num_packs = quotient_domain_size  - 1 ;
    int threads_per_block = 256;
    int blocks_per_grid = (num_packs + threads_per_block - 1) / threads_per_block;

    //int main_trace_height = 1 << trace_log_size;

    void* args[] = {&chip_id, &d_main_trace, &main_width, &d_prep_trace, &prep_width, &d_perm_trace, &perm_width,
                    &main_height, &quotient_domain_size,  &d_inv_vanishing, &d_is_first_row, &d_is_last_row, &d_is_transition, 
                    &next_step, &batch_size, &d_powers_of_alpha, &alpha_offset, &d_perm_challenges, &d_local_cumulative_sum,  &d_quotient_values_out};

    CUDA_CHECK(cudaLaunchKernel((void*)quotient_values_kernel, blocks_per_grid, threads_per_block, args, 0, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) { return -1; }
    
    CUDA_CHECK(cudaMemcpy(h_quotient_values_out, d_quotient_values_out, quotient_size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_main_trace)); 
    CUDA_CHECK(cudaFree(d_prep_trace)); 
    CUDA_CHECK(cudaFree(d_powers_of_alpha)); 
    CUDA_CHECK(cudaFree(d_perm_trace)); 
    CUDA_CHECK(cudaFree(d_perm_challenges)); 
    CUDA_CHECK(cudaFree(d_local_cumulative_sum));
    CUDA_CHECK(cudaFree(d_quotient_values_out));

    CUDA_CHECK(cudaFree(d_is_first_row));
    CUDA_CHECK(cudaFree(d_is_last_row));
    CUDA_CHECK(cudaFree(d_is_transition));
    CUDA_CHECK(cudaFree(d_inv_vanishing));

    return 0;
}
