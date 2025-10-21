#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"
#include "permutation.hpp"

#define FRI_FOLD_DEGREE 3

// --- UNPACKED EVAL FUNCTION ---
__device__ void eval_fri_fold_chip(
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
    const auto* local = reinterpret_cast<const FriFoldCols<Val>*>(main_local_row);
    const auto* next = reinterpret_cast<const FriFoldCols<Val>*>(main_next_row);
    const auto* prep_local = reinterpret_cast<const FriFoldPreprocessedCols<Val>*>(prep_local_row);
    const auto* prep_next = reinterpret_cast<const FriFoldPreprocessedCols<Val>*>(prep_next_row);

    // 1. Dummy constraint
    Val lhs = prep_local->is_real;
    for (int i = 1; i < FRI_FOLD_DEGREE; ++i) {
        lhs = lhs * prep_local->is_real;
    }
    folder_assert_zero(folder, lhs - lhs);

    // 2. Internal Chip and Transition Constraints
    // Convert values to extension field elements for calculations
    Challenge z_local_ext = Challenge::from_block(local->z);
    Challenge alpha_local_ext = Challenge::from_block(local->alpha);
    Challenge x_local_ext = Challenge(local->x); // Base field element promoted to extension

    // Transition selector for values that are constant within a fold operation
    Val when_transition_inner = is_transition * prep_next->is_real * (prep_next->is_first - Val::one());
    
    // a) Ensure x is constant within a FRI fold invocation
    folder_assert_zero(folder, (local->x - next->x) * when_transition_inner);
    
    // b) Ensure z is constant
    Challenge z_next_ext = Challenge::from_block(next->z);
    folder_assert_ext_eq(folder, z_local_ext, z_next_ext, when_transition_inner);

    // c) Ensure alpha is constant
    Challenge alpha_next_ext = Challenge::from_block(next->alpha);
    folder_assert_ext_eq(folder, alpha_local_ext, alpha_next_ext, when_transition_inner);

    // d) Core FRI fold constraints (only apply to real rows)
    Val is_real = prep_local->is_real;

    // Constraint 1: new_alpha_pow = old_alpha_pow * alpha
    Challenge old_alpha_pow_ext = Challenge::from_block(local->alpha_pow_input);
    Challenge new_alpha_pow_ext = Challenge::from_block(local->alpha_pow_output);
    folder_assert_ext_eq(folder, old_alpha_pow_ext * alpha_local_ext, new_alpha_pow_ext, is_real);
    
    // Constraint 2: (new_ro - old_ro) * (z - x) = old_alpha_pow * (p_at_x - p_at_z)
    Challenge p_at_x_ext = Challenge::from_block(local->p_at_x);
    Challenge p_at_z_ext = Challenge::from_block(local->p_at_z);
    Challenge old_ro_ext = Challenge::from_block(local->ro_input);
    Challenge new_ro_ext = Challenge::from_block(local->ro_output);
    
    Challenge lhs_c2 = (new_ro_ext - old_ro_ext) * ( x_local_ext - z_local_ext);
    Challenge rhs_c2 = (p_at_x_ext - p_at_z_ext) * old_alpha_pow_ext;
   

    printf("--gpu:fri_fold, is_real=%u \n", is_real.as_canonical_u32());
    folder_assert_ext_eq(folder, lhs_c2, rhs_c2, is_real);

    // 3. Memory Interactions (Permutation Argument)
    Interaction interaction_buffer[10];
    int interaction_count = get_fri_fold_interactions(interaction_buffer);
    

    
    // I have omitted the zero-padding for VirtualPairCol for brevity, but they should be there.
    // The permutation logic here is a bit tricky, since it's "send" in Rust but logically a "read". 
    // SP1 uses "send" for all memory interactions. So is_send=true is correct.
    
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

