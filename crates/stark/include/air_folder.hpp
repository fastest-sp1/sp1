// file: air_folder.hpp (PERFECTED VERSION)
#pragma once
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp" 
#include <vector>


/**
 * @brief A templated GPU constraint evaluation context.
 * @tparam AccumulatorT The type of the accumulator (e.g., Challenge or PackedChallenge<N>).
 */
template <typename AccumulatorT>
struct ProverConstraintFolder {
    AccumulatorT* accumulator;
    const Challenge* powers_of_alpha_base; // Pointer to the START of the global array
    int* constraint_index; // Pointer to a LOCAL int for this thread
    int alpha_offset;      // The starting offset for this chip's alphas
    
    const SepticDigest_bb31* global_cumulative_sum;
};

__device__ void folder_assert_zero(ProverConstraintFolder<Challenge>& folder, Val val) {
  int local_idx = (*folder.constraint_index)++;
  int global_idx = folder.alpha_offset + local_idx;  
  
  *folder.accumulator += folder.powers_of_alpha_base[global_idx] * Challenge(val);
}

__device__ void folder_assert_zero_ext(ProverConstraintFolder<Challenge>& folder, Challenge challenge) {
  int local_idx = (*folder.constraint_index)++;
  // Calculate the global index.
  int global_idx = folder.alpha_offset + local_idx;
  
  // Access the global alpha array.
  *folder.accumulator += folder.powers_of_alpha_base[global_idx] * challenge;
 }

__device__ void folder_assert_bool(ProverConstraintFolder<Challenge>& folder, Val val) {
  // p3-field/bool_check(): We use `x * (1 - x)` instead of `x * (x - 1)` as this lets us delegate to the `andn` function.
  folder_assert_zero(folder, val * (Val::one() - val));
}

__device__ void folder_assert_ext_eq(ProverConstraintFolder<Challenge>& folder, Challenge left, Challenge right, Val factor) {
    for (int i =0; i < 4; ++i){
      folder_assert_zero(folder, (left.coeffs[i] - right.coeffs[i]) * factor);
    }
}
