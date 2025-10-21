#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"

#include "permutation.hpp"

#define BATCH_FRI_DEGREE 3


__device__ void eval_batch_fri_chip(
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
    const auto* local = reinterpret_cast<const BatchFRICols<Val>*>(main_local_row);
    const auto* next = reinterpret_cast<const BatchFRICols<Val>*>(main_next_row);
    const auto* prep_local = reinterpret_cast<const BatchFRIPreprocessedCols<Val>*>(prep_local_row);
    // prep_next is not used in the Rust logic, so we can ignore it.

    // 1. Dummy constraint to raise degree
    Val is_real = prep_local->is_real;
    Val lhs = is_real;
    for (int i = 1; i < BATCH_FRI_DEGREE; ++i) {
        lhs = lhs * is_real;
    }
    Val rhs = lhs; // It's a tautology
    folder_assert_zero(folder, lhs - rhs);

    //2. Constrain the accumulator value of the first row.
    Challenge acc_local_ext = Challenge::from_block(local->acc);
    Challenge alpha_pow_local_ext = Challenge::from_block(local->alpha_pow);
    Challenge p_at_z_local_ext = Challenge::from_block(local->p_at_z);
    Challenge p_at_x_local_ext = Challenge(local->p_at_x);
    folder_assert_ext_eq(folder, acc_local_ext, alpha_pow_local_ext * (p_at_z_local_ext - p_at_x_local_ext), is_first_row);
    
    //3. Constrain the accumulator of the next row when the current row is the end of loop.
    Challenge acc_next_ext = Challenge::from_block(next->acc);
    Challenge alpha_pow_next_ext = Challenge::from_block(next->alpha_pow);
    Challenge p_at_z_next_ext = Challenge::from_block(next->p_at_z);
    Challenge p_at_x_next_ext = Challenge(next->p_at_x );
    folder_assert_ext_eq(folder, acc_next_ext, alpha_pow_next_ext * (p_at_z_next_ext - p_at_x_next_ext), is_transition * prep_local->is_end);
    

    //4. Constrain the accumulator of the next row when the current row is not the end of loop.
    folder_assert_ext_eq(folder, acc_next_ext, acc_local_ext + alpha_pow_next_ext * (p_at_z_next_ext - p_at_x_next_ext), is_transition * (prep_local->is_end - Val::one()  ));
    

    // 3. Memory Interactions (Permutation Argument)
    Interaction interaction_buffer[4];
    int interaction_count = get_batch_fri_interactions(interaction_buffer);

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


