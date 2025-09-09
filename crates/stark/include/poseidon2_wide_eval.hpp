#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"
#include "permutation_eval.hpp"
#include "poseidon_common.hpp"

#include "poseidon2.hpp"
#include "poseidon2_wide.hpp"


#define POSEIDON2_WIDE_DEGREE 3 // Or 9, passed as a template parameter

__device__ inline int get_p2_wide_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;

    // a) Input memory interactions
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        interaction_buffer[interaction_count++] = {
            .values = {
                VirtualPairCol::single_preprocessed(offsetof(Poseidon2PreprocessedColsWide<Val>, input[i]) / sizeof(Val)),
                VirtualPairCol::single_main(offsetof(Poseidon2Degree3Cols<Val>, state.external_rounds_state) / sizeof(Val) + i) // input is the first 16 columns

            }, 
            .num_values = 2,
            .multiplicity = VirtualPairCol::single_preprocessed(offsetof(Poseidon2PreprocessedColsWide<Val>, is_real_neg) / sizeof(Val)),
            .kind = InteractionKind::Memory, 
            .scope = InteractionScope::Local, 
            .is_send = true
        };
    }

    // b) Output memory interactions
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        interaction_buffer[interaction_count++] = {
            .values = {
                VirtualPairCol::single_preprocessed(offsetof(Poseidon2PreprocessedColsWide<Val>, output[i].addr) / sizeof(Val)),
                VirtualPairCol::single_main(offsetof(Poseidon2Degree3Cols<Val>, state.output_state) / sizeof(Val) + i) // Need correct offset for output
               
            }, 
            .num_values = 2,
            .multiplicity = VirtualPairCol::single_preprocessed(offsetof(Poseidon2PreprocessedColsWide<Val>, output[i].mult) / sizeof(Val)),
            .kind = InteractionKind::Memory, 
            .scope = InteractionScope::Local, 
            .is_send = true
        };
    }

    return interaction_count;
}



// --- UNPACKED EVAL FUNCTION ---
__device__ void eval_poseidon2_wide_chip(
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
 
    const auto* prep_local = reinterpret_cast<const Poseidon2PreprocessedColsWide<Val>*>(prep_local_row);
    //Val is_real = prep_local->is_real_neg * Val::neg_one(); // is_real_neg is -1, so is_real is 1

    // 2. Dummy constraint
    Val dummy_lhs = main_local_row[0];
    for (int i = 1; i < POSEIDON2_WIDE_DEGREE; ++i) {
        dummy_lhs = dummy_lhs * main_local_row[0];
    }
    folder_assert_zero(folder, dummy_lhs - dummy_lhs);


    const Val* external_rounds_state = main_local_row;
    const Val* internal_rounds_state = external_rounds_state + POSEIDON2_STATE_WIDTH * NUM_EXTERNAL_ROUNDS;
    const Val* internal_rounds_s0 = internal_rounds_state + POSEIDON2_STATE_WIDTH;
    const Val* perm_output = internal_rounds_s0 + (NUM_INTERNAL_ROUNDS - 1);
    
    // a) external round
    for (int r = 0; r < NUM_EXTERNAL_ROUNDS; ++r) {
        eval_external_round(folder, main_local_row, r);
    } 

    // b) internal round
    eval_internal_rounds(folder, main_local_row);

    // 4. Memory Interactions
    Interaction interaction_buffer[POSEIDON2_STATE_WIDTH * 2];
    int interaction_count = get_p2_wide_interactions(interaction_buffer);

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
