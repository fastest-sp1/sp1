#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"
#include "permutation.hpp"


// We only need the digest part for this chip
#define  DIGEST_SIZE  8

__device__ void eval_public_values_chip(
    ProverConstraintFolder<Challenge>& folder,
    const Val* main_local_row,
    const Val* main_next_row,
    const Val* prep_local_row,
    const Val* prep_next_row,
    const Val* perm_local_flat,
    const Val* perm_next_flat,
    int perm_width,
    int batch_size,
    const Challenge* perm_challenges,
    const Challenge& local_cumulative_sum,
    const Val* public_values, // New parameter: Pointer to public values array
    Val is_first_row, 
    Val is_last_row, 
    Val is_transition
) {
    // Cast raw pointers to structured views
    const auto* local = reinterpret_cast<const PublicValuesCols<Val>*>(main_local_row);
    const auto* prep_local = reinterpret_cast<const PublicValuesPreprocessedCols<Val>*>(prep_local_row);

    // 1. Core constraint logic
    // Rust: builder.when(local_prepr.pv_idx[i]).assert_eq(pv_elm.clone(), local.pv_element);
    // This loops through the public values and applies a constraint for each.
    for (int i = 0; i < DIGEST_SIZE; ++i) { 
        Val selector = prep_local->pv_idx[i];
        folder_assert_zero(folder, (public_values[i] - local->pv_element) * selector);
    }
    
    // 2. Memory Interactions (Permutation Argument)
    // There is only one memory interaction.
    Interaction interaction_buffer[1];
    int interaction_count = get_public_values_interactions(interaction_buffer);
    
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
