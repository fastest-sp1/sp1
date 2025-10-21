#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"
#include "permutation.hpp"

__device__ void eval_select_chip(
    ProverConstraintFolder<Challenge>& folder,
    const Val* main_local_row,
    const Val* main_next_row, // Not used, but kept for signature consistency
    const Val* prep_local_row,
    const Val* prep_next_row, // Not used
    const Val* perm_local_flat,
    const Val* perm_next_flat,
    int perm_width,
    int batch_size,
    const Challenge* perm_challenges,
    const Challenge& local_cumulative_sum,
    Val is_first_row, 
    Val is_last_row, 
    Val is_transition
) {
    // Cast raw pointers to structured views
    const auto* local = reinterpret_cast<const SelectCols<Val>*>(main_local_row);
   // const auto* prep_local = reinterpret_cast<const SelectPreprocessedCols<Val>*>(prep_local_row);

    // 1. Core constraint logic (only applied when is_real is true)
    Val expected_out1 = local->vals.bit * local->vals.in2 + (Val::one() - local->vals.bit) * local->vals.in1;
    folder_assert_zero(folder, local->vals.out1 - expected_out1);

    // Constraint 2
    Val expected_out2 = local->vals.bit * local->vals.in1 + (Val::one() - local->vals.bit) * local->vals.in2;
    folder_assert_zero(folder, local->vals.out2 - expected_out2);
    
    // 2. Memory Interactions (Permutation Argument)
    Interaction interaction_buffer[5];
    int interaction_count = get_select_interactions(interaction_buffer);
    
    //const int perm_width_ef = perm_width / 4;
    const Challenge* perm_local = reinterpret_cast<const Challenge*>(perm_local_flat);
    const Challenge* perm_next = reinterpret_cast<const Challenge*>(perm_next_flat);

    eval_permutation_constraints(
        folder, 
        interaction_buffer, 
        interaction_count,
        main_local_row, 
        prep_local_row, 
        perm_local, perm_next, 
        perm_width, 
        batch_size,
        perm_challenges,
        local_cumulative_sum, 
        is_first_row, 
        is_last_row, 
        is_transition
    );
}
