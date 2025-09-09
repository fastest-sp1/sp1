#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"
#include "permutation_eval.hpp"
#include "poseidon2.hpp"


#define POSEIDON2_SKINNY_DEGREE 9

// --- eval_input_round ---
__device__ void eval_input_round_skinny(
    ProverConstraintFolder<Challenge>& folder,
    const Poseidon2SkinnyCols<Val>* local,
    const Poseidon2SkinnyCols<Val>* next,
    const Poseidon2PreprocessedColsSkinny<Val>* prep_local,
    Val is_transition
) {
    Val state_computed[POSEIDON2_STATE_WIDTH];
    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        state_computed[i] = local->state_var[i];
    }
    
    sp1_recursion_core_sys::poseidon2::external_linear_layer(state_computed);

    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        folder_assert_zero(folder, (next->state_var[i] - state_computed[i]) * is_transition * prep_local->round_counters_preprocessed.is_input_round);
    }
}

// --- eval_external_round ---
__device__ void eval_external_round_skinny(
    ProverConstraintFolder<Challenge>& folder,
    const Poseidon2SkinnyCols<Val>* local,
    const Poseidon2SkinnyCols<Val>* next,
    const Poseidon2PreprocessedColsSkinny<Val>* prep_local,
    Val is_transition
) {
    // a) Add round constants
    Val state_rc[POSEIDON2_STATE_WIDTH];
    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        state_rc[i] = local->state_var[i] + prep_local->round_counters_preprocessed.round_constants[i];
    }

    // b) S-Box (x^7)
    Val sbox_deg_7[POSEIDON2_STATE_WIDTH];
    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        Val sbox_deg_3 = state_rc[i] * state_rc[i] * state_rc[i];
        sbox_deg_7[i] = sbox_deg_3 * sbox_deg_3 * state_rc[i];
    }

    // c) Linear Layer
    sp1_recursion_core_sys::poseidon2::external_linear_layer(sbox_deg_7);

    // d) Assert equality with next state
    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        folder_assert_zero(folder, (next->state_var[i] - sbox_deg_7[i]) * is_transition * prep_local->round_counters_preprocessed.is_external_round);
    }
}

// --- eval_internal_rounds ---
__device__ void eval_internal_rounds_skinny(
    ProverConstraintFolder<Challenge>& folder,
    const Poseidon2SkinnyCols<Val>* local,
    const Poseidon2SkinnyCols<Val>* next,
    const Val* round_constants,
    Val is_internal_row
) {
    Val state_computed[POSEIDON2_STATE_WIDTH];
    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        state_computed[i] = local->state_var[i];
    }

    for (int r = 0; r < NUM_INTERNAL_ROUNDS; ++r) {
        Val s0_input = (r == 0) ? state_computed[0] : local->internal_rounds_s0[r-1];
        Val add_rc = s0_input + round_constants[r];
        
        Val sbox_deg_3 = add_rc * add_rc * add_rc;
        Val sbox_deg_7 = sbox_deg_3 * sbox_deg_3 * add_rc;

        state_computed[0] = sbox_deg_7;
        sp1_recursion_core_sys::poseidon2::internal_linear_layer(state_computed);
        
        if (r < NUM_INTERNAL_ROUNDS - 1) {
            folder_assert_zero(folder, (local->internal_rounds_s0[r] - state_computed[0]) * is_internal_row);
        }
    }

    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        folder_assert_zero(folder, (next->state_var[i] - state_computed[i]) * is_internal_row);
    }
}


// --- Main UNPACKED EVAL FUNCTION ---
__device__ void eval_poseidon2_skinny_chip(
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
    // Cast pointers to structured views
    const auto* local = reinterpret_cast<const Poseidon2SkinnyCols<Val>*>(main_local_row);
    const auto* next = reinterpret_cast<const Poseidon2SkinnyCols<Val>*>(main_next_row);
    const auto* prep_local = reinterpret_cast<const Poseidon2PreprocessedColsSkinny<Val>*>(prep_local_row);
    const auto* prep_counters = &prep_local->round_counters_preprocessed;

    // 1. Dummy constraint
    Val dummy_lhs = local->state_var[0];
    for (int i = 1; i < POSEIDON2_SKINNY_DEGREE; ++i) {
        dummy_lhs = dummy_lhs * local->state_var[0];
    }
    folder_assert_zero(folder, dummy_lhs - dummy_lhs);
    
    // 2. Dispatch to the correct round evaluation function based on selectors
    eval_input_round_skinny(folder, local, next, prep_local, is_transition);
    eval_external_round_skinny(folder, local, next, prep_local, is_transition);
    eval_internal_rounds_skinny(folder, local, next, prep_counters->round_constants, prep_counters->is_internal_round);
    
    // 3. Memory Interactions
    Interaction interaction_buffer[POSEIDON2_STATE_WIDTH];
    int interaction_count = 0;
    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        interaction_buffer[interaction_count++] = {
            .values = {
                VirtualPairCol::single_preprocessed(offsetof(Poseidon2PreprocessedColsSkinny<Val>, memory_preprocessed[i].addr) / sizeof(Val)),
                VirtualPairCol::single_main(offsetof(Poseidon2SkinnyCols<Val>, state_var[i]) / sizeof(Val))
                
            }, 
            .num_values = 2,
            .multiplicity = VirtualPairCol::single_preprocessed(offsetof(Poseidon2PreprocessedColsSkinny<Val>, memory_preprocessed[i].mult) / sizeof(Val)),
            .kind = InteractionKind::Memory, 
            .scope = InteractionScope::Local, 
            .is_send = true
        };
    }

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

