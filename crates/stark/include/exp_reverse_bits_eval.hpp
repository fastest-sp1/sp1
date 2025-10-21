#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"

#include "permutation.hpp"

#define EXP_REVERSE_BITS_DEGREE 3

// --- UNPACKED EVAL FUNCTION ---
__device__ void eval_exp_reverse_bits_chip(
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
    Val is_first_row, 
    Val is_last_row, 
    Val is_transition
) {
    // Cast raw pointers to structured views
    const auto* main_local = reinterpret_cast<const ExpReverseBitsLenCols<Val>*>(main_local_row);
    const auto* main_next = reinterpret_cast<const ExpReverseBitsLenCols<Val>*>(main_next_row);
    const auto* prep_local = reinterpret_cast<const ExpReverseBitsLenPreprocessedCols<Val>*>(prep_local_row);
    const auto* prep_next = reinterpret_cast<const ExpReverseBitsLenPreprocessedCols<Val>*>(prep_next_row);

    // 1. Dummy constraint (if needed)
    if (EXP_REVERSE_BITS_DEGREE > 3) {
        Val lhs = prep_local->is_real;
        for (int i = 1; i < EXP_REVERSE_BITS_DEGREE; ++i) {
            lhs = lhs * prep_local->is_real;
        }
        folder_assert_zero(folder, lhs - lhs);
    }
    
    // 2. Internal Chip Constraints
    
    // Ensure that the value at the x memory access is unchanged when not `is_last`.
    folder_assert_zero(folder, (main_local->x - main_next->x) * is_transition * prep_next->is_real * 
                                            (prep_local->is_last - Val::one()));

    // The accumulator needs to start with the multiplier for every `is_first` row.
    folder_assert_zero(folder, (main_local->accum -  main_local->multiplier) * prep_local->is_first);

    // `multiplier` is x if the current bit is 1, and 1 if the current bit is 0.
    folder_assert_zero(folder, (main_local->multiplier - main_local->x) * prep_local->is_real * main_local->current_bit);

    folder_assert_zero(folder, (main_local->multiplier - Val::one()) * prep_local->is_real * (main_local->current_bit - Val::one()));

    // To get `next.accum`, we multiply `local.prev_accum_squared` by `local.multiplier` when
    // not `is_last`.
    folder_assert_zero(folder, (main_local->prev_accum_squared_times_multiplier - main_local->prev_accum_squared * main_local->multiplier)
                                * prep_local->is_real);

    folder_assert_zero(folder, (main_local->accum - main_local->prev_accum_squared_times_multiplier) *
                                    prep_local->is_real * (prep_local->is_first - Val::one()));

    // Constrain the accum_squared column.
    //    builder.when(local_prepr.is_real).assert_eq(local.accum_squared, local.accum * local.accum);
    folder_assert_zero(folder, (main_local->accum_squared - main_local->accum * main_local->accum) * prep_local->is_real);

    folder_assert_zero(folder, (main_next->prev_accum_squared - main_local->accum_squared) * is_transition * prep_next->is_real * (prep_local->is_last - Val::one()));

// 3. Memory Interactions (Permutation Argument)
    Interaction interaction_buffer[3];
    int interaction_count = get_exp_reverse_fri_interactions(interaction_buffer);
    
    
    //const int perm_width_ef = perm_width / 4;
    const Challenge* perm_local = reinterpret_cast<const Challenge*>(perm_local_flat);
    const Challenge* perm_next = reinterpret_cast<const Challenge*>(perm_next_flat);

    eval_permutation_constraints(
        folder, 
        interaction_buffer, 
        interaction_count,
        main_local_row, 
        prep_local_row, 
        perm_local, 
        perm_next, 
        perm_width,
        batch_size,
        perm_challenges,
        local_cumulative_sum, 
        is_first_row, 
        is_last_row,
        is_transition
    ); 
}


