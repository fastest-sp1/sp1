//file: permutation_eval.hpp
#pragma once
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"
#include "air_folder.hpp"
//#include "sp1-recursion-core-sys-cbindgen.hpp" 

__device__ inline Challenge calculate_rlc(
    const Interaction& interaction,
    const Challenge& alpha,
    const Challenge& beta,
    const Val* main_row,
    const Val* prep_row
) {
    Challenge rlc = alpha;
    //Challenge beta_pow = beta;
    Challenge first_beta_pow = Challenge::one();

    rlc += first_beta_pow * Challenge(Val::from_canonical_u32(uint32_t(interaction.kind)));
 
    Challenge beta_pow = beta;
    for (int i = 0; i < interaction.num_values; ++i) {
        Val val = interaction.values[i].apply(main_row, prep_row);

        rlc += beta_pow * Challenge(val);
        beta_pow *= beta;
    }
    return rlc;
}


__device__ void eval_permutation_constraints(
      ProverConstraintFolder<Challenge>& folder,
      const Interaction* interaction_buffer,
      int num_interaction,
      const Val* main_row,
      const Val* prep_row,
      const Challenge* perm_local,
      const Challenge* perm_next,
      int perm_width,
      int batch_size,
      const Challenge* perm_challenges,
      const Challenge& local_cumulative_sum,
      Val is_first_row, 
      Val is_last_row, 
      Val is_transition
) {
      const Challenge& alpha = perm_challenges[0];
      const Challenge& beta = perm_challenges[1];

      int perm_width_ef = perm_width / 4;

      // The number of chunks is determined by the number of interactions.
      int num_chunks = (num_interaction + batch_size - 1) / batch_size;

      // The number of entries available for batch sum verification in the permutation trace.
      // It's the total width of the permutation trace minus the last column (which is phi).
      int num_entries_available = perm_width_ef > 0 ? perm_width_ef - 1 : 0;


      // The loop must run for the minimum of the number of available chunks and the number of available entries.
      // This exactly mimics the behavior of `zip`.
      int loop_iterations = min(num_chunks, num_entries_available);

      // === STAGE 1: BATCHED SUM VERIFICATION ===
      #define MAX_BATCH_SIZE 8 
      Challenge rlcs[MAX_BATCH_SIZE];
      Challenge multiplicities[MAX_BATCH_SIZE];

      for (int i = 0; i < loop_iterations; ++i) {
          const Challenge& entry = perm_local[i];
          
          int start_idx = i * batch_size;
          // chunk_len calculation is safe because we know the interactions exist due to the loop bound.
          int chunk_len = min(batch_size, num_interaction - start_idx);

          
          // 1. Collect RLCs and multiplicities for the current chunk.
          // This part of the logic remains unchanged.
          for (int j = 0; j < chunk_len; ++j) {
              const auto& interaction = interaction_buffer[start_idx + j];
              rlcs[j] = calculate_rlc(interaction, alpha, beta, main_row, prep_row);
              
              Val mult_val = interaction.multiplicity.apply(main_row, prep_row);
              multiplicities[j] = Challenge(mult_val) * (interaction.is_send ? Challenge::one() : Challenge(Val::neg_one()));
          }

          // 2. Calculate the combined constraint.
          // This part of the logic also remains unchanged.
          Challenge product = Challenge::one();
          for (int j = 0; j < chunk_len; ++j) {
              product = product * rlcs[j];
          }

          Challenge numerator = Challenge::zero();
          for (int j = 0; j < chunk_len; ++j) {
              Challenge all_but_current = Challenge::one();
              for (int k = 0; k < chunk_len; ++k) {
                  if (j != k) {
                      all_but_current = all_but_current * rlcs[k];
                  }
              }
              numerator = numerator + multiplicities[j] * all_but_current;
          }

          // 3. Apply the constraint.
          folder_assert_zero_ext(folder, product * entry - numerator);
      }
   
      // === STAGE 2: RUNNING SUM VERIFICATION (This part remains the same) ===
      if (perm_width_ef > 0) {
          Challenge sum_local = Challenge::zero();
          for (int i = 0; i < perm_width_ef - 1; ++i) sum_local += perm_local[i];
          
          Challenge sum_next = Challenge::zero();
          for (int i = 0; i < perm_width_ef - 1; ++i) sum_next += perm_next[i];

          const Challenge& phi_local = perm_local[perm_width_ef - 1];
          const Challenge& phi_next = perm_next[perm_width_ef - 1];

          folder_assert_zero_ext(folder, (phi_local - sum_local) * Challenge(is_first_row));
          folder_assert_zero_ext(folder, (phi_next - phi_local - sum_next) * Challenge(is_transition));
          folder_assert_zero_ext(folder, (phi_local - local_cumulative_sum) * Challenge(is_last_row));
      }
}