#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "poseidon2.hpp" 
#include "poseidon2_constants.hpp"
#include "air_folder.hpp"

//use namespace sp1_recursion_core_sys;

// r : round index (0 to NUM_EXTERNAL_ROUNDS-1)
__device__ __forceinline__ const Val* get_external_round_state_ptr(const Val* main_row, int r) {
    return main_row + r * POSEIDON2_STATE_WIDTH;
}

// Helper to get sbox_deg_3 values pointer for a given external round.
// This assumes the trace layout for DEGREE > 3.
__device__ __forceinline__ const Val* get_external_sbox_deg3_ptr(const Val* main_row, int r) {
    // Offset for the start of the sbox columns.
    return main_row + SBOX_BASE_OFFSET + r * POSEIDON2_STATE_WIDTH;
}


/**
 * @brief Evaluates the constraints for a single external round of the Poseidon2 permutation.
 * This function is a direct C++/CUDA translation of the Rust AIR `eval_external_round`
 * for the DEGREE=3 trace layout (without explicit S-Box columns).
 * @tparam Folder The constraint folder type.
 * @param folder The constraint folder to add constraints to.
 * @param main_row Pointer to the start of the current row in the main trace.
 * @param r The index of the external round to evaluate (0 to 7).
 * @param selector A selector value (e.g., is_real) that is multiplied by each constraint.
 */
__device__ void eval_external_round(
    ProverConstraintFolder<Challenge>& folder,
    const Val* main_row,
    int r // Round index: 0 to 7
) {
    // 1. Get pointers to the relevant parts of the trace for this round.
    const Val* current_state_trace = get_external_round_state_ptr(main_row, r);
    const Val* sbox_deg3_trace = get_external_sbox_deg3_ptr(main_row, r);
    const Val* next_state_trace;

    // Determine the pointer to the next state based on the round number.
    if (r == NUM_EXTERNAL_ROUNDS / 2 - 1) { // Transition from first half of external to internal rounds.
        size_t offset = POSEIDON2_STATE_WIDTH * NUM_EXTERNAL_ROUNDS;
        next_state_trace = main_row + offset; // Pointer to internal_rounds_state[0].
    } else if (r == NUM_EXTERNAL_ROUNDS - 1) { // The final external round.
        size_t offset = POSEIDON2_STATE_WIDTH * NUM_EXTERNAL_ROUNDS + POSEIDON2_STATE_WIDTH + P2_GHOST;
        next_state_trace = main_row + offset; // Pointer to perm_output.
    } else { // Standard transition to the next external round state.
        next_state_trace = get_external_round_state_ptr(main_row, r + 1);
    }
    
    // 2. Begin symbolic computation, starting from the current state in the trace.
    Val state_after_linear_layer[POSEIDON2_STATE_WIDTH];
    for(int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        state_after_linear_layer[i] = current_state_trace[i];
   }

    // a) In the very first round (r=0), an extra linear layer is applied at the beginning.
    if (r == 0) {
        sp1_recursion_core_sys::poseidon2::external_linear_layer(state_after_linear_layer);
    }
   
   // b) Add Round Constants.
    int round_idx_rc = (r < NUM_EXTERNAL_ROUNDS / 2) ? r : r + NUM_INTERNAL_ROUNDS;
    Val state_rc[POSEIDON2_STATE_WIDTH];
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        state_rc[i] = state_after_linear_layer[i] + Val(Val::to_monty(sp1_recursion_core_sys::constants::RC_16_30_U32[round_idx_rc][i]));
    }
    
    // c) Apply S-Box and verify the intermediate sbox_deg_3 values from the trace.
    Val sbox_deg_7[POSEIDON2_STATE_WIDTH];
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        // Symbolically compute what sbox_deg_3 should be.
        Val calculated_sbox_deg_3 = state_rc[i] * state_rc[i] * state_rc[i];

        // THE CRUCIAL CONSTRAINT:
        // Assert that the value from the trace (`sbox_deg3_trace`) matches the computed value.
        // This corresponds to: `builder.assert_eq(external_sbox[r][i].into(), calculated_sbox_deg_3);`
        folder_assert_zero(folder, sbox_deg3_trace[i] - calculated_sbox_deg_3);

        // For the next computational step, use the (now verified) value from the trace
        // to keep the constraint degree low. This corresponds to:
        // `sbox_deg_3[i] = external_sbox[r][i].into();`
        Val sbox_deg3_from_trace = sbox_deg3_trace[i];

        // Now compute sbox_deg_7 using the value from the trace.
        // `sbox_deg_7[i] = sbox_deg_3[i].clone() * sbox_deg_3[i].clone() * add_rc[i].clone();`
        Val sbox_deg3_sq = sbox_deg3_from_trace * sbox_deg3_from_trace;
        sbox_deg_7[i] = sbox_deg3_sq * state_rc[i];
    }

    // d) Apply the final External Linear Layer for the round.
    Val state_after_final_linear_layer[POSEIDON2_STATE_WIDTH];
     for(int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        state_after_final_linear_layer[i] = sbox_deg_7[i];
    }
    sp1_recursion_core_sys::poseidon2::external_linear_layer(state_after_final_linear_layer);

    // 3. Assert that the fully computed state equals the state recorded in the next step of the trace.
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        folder_assert_zero(folder, next_state_trace[i] - state_after_final_linear_layer[i]);
    }
}


// eval_external_round_unpack 

// Helper to get a pointer to the start of internal_rounds_state
__device__ __forceinline__ const Val* get_internal_rounds_state_ptr(const Val* main_row) {
    return main_row + POSEIDON2_STATE_WIDTH * NUM_EXTERNAL_ROUNDS;
}

// Helper to get a pointer to the start of internal_rounds_s0
__device__ __forceinline__ const Val* get_internal_rounds_s0_ptr(const Val* main_row) {
    return main_row + POSEIDON2_STATE_WIDTH * NUM_EXTERNAL_ROUNDS + POSEIDON2_STATE_WIDTH;
}

// Helper to get a pointer to the start of internal_rounds_sbox (for DEGREE > 3)
__device__ __forceinline__ const Val* get_internal_sbox_deg3_ptr(const Val* main_row) {
    // Offset for the start of the sbox columns, after external sbox columns.
    size_t sbox_base_offset = SBOX_BASE_OFFSET + POSEIDON2_STATE_WIDTH * NUM_EXTERNAL_ROUNDS;
    return main_row + sbox_base_offset;
}


/**
 * @brief Evaluates the constraints for all internal rounds of the Poseidon2 permutation.
 * This is a C++/CUDA translation of the Rust AIR `eval_internal_rounds`.
 */
__device__ void eval_internal_rounds(
    ProverConstraintFolder<Challenge>& folder,
    const Val* main_row
) {
    // 1. Get pointers to the relevant parts of the trace.
    const Val* internal_state_trace_initial = get_internal_rounds_state_ptr(main_row);
    const Val* s0_trace = get_internal_rounds_s0_ptr(main_row);
    const Val* internal_sbox_trace = get_internal_sbox_deg3_ptr(main_row);

    // 2. Initialize the state for the symbolic computation.
    // The state will be updated through each of the internal rounds.
    Val state_computed[POSEIDON2_STATE_WIDTH];
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        state_computed[i] = internal_state_trace_initial[i];
    }

    // 3. Loop through all internal rounds and apply/verify constraints.
    for (int r = 0; r < NUM_INTERNAL_ROUNDS; ++r) {
        // a) Add Round Constant to state[0].
        // The input to this operation depends on the round.
        // `let add_rc = if r == 0 { state[0].clone() } else { s0[r - 1].into() } + ...`
        Val s0_input = (r == 0) ? state_computed[0] : s0_trace[r - 1];
        int round_idx_rc = r + NUM_EXTERNAL_ROUNDS / 2;
        Val add_rc = s0_input + Val(Val::to_monty(sp1_recursion_core_sys::constants::RC_16_30_U32[round_idx_rc][0]));

        // b) S-Box and its verification.
        // `let mut sbox_deg_3 = add_rc.clone() * add_rc.clone() * add_rc.clone();`
        Val calculated_sbox_deg_3 = add_rc * add_rc * add_rc;
        
        // This corresponds to the `if let Some(internal_sbox) = ...` block.
        // We assume the trace always has these columns for generality.
        // `builder.assert_eq(internal_sbox[r], sbox_deg_3);`
        folder_assert_zero(folder, internal_sbox_trace[r] - calculated_sbox_deg_3);
      
        // Use the verified value from the trace for degree reduction.
        // `sbox_deg_3 = internal_sbox[r].into();`
        Val sbox_deg3_from_trace = internal_sbox_trace[r];

        // `let sbox_deg_7 = sbox_deg_3.clone() * sbox_deg_3.clone() * add_rc.clone();`
        Val sbox_deg_7 = sbox_deg3_from_trace * sbox_deg3_from_trace * add_rc;

        // c) Apply the linear layer.
        // Update the first element of the state with the S-Box output.
        // `state[0] = sbox_deg_7.clone();`
        state_computed[0] = sbox_deg_7;
        // `internal_linear_layer_mut(&mut state);`
        sp1_recursion_core_sys::poseidon2::internal_linear_layer(state_computed);

        // d) Verify the intermediate `s0` values.
        // `if r < NUM_INTERNAL_ROUNDS - 1 { builder.assert_eq(s0[r], state[0].clone()); }`
        if (r < NUM_INTERNAL_ROUNDS - 1) {
            folder_assert_zero(folder, s0_trace[r] - state_computed[0]);
        }
    }

    // 4. Final constraint: The state after all internal rounds must match the state
    //    recorded in the trace for the beginning of the second half of external rounds.
    // `let external_state = local_row.external_rounds_state()[NUM_EXTERNAL_ROUNDS / 2];`
    // `builder.assert_eq(external_state[i], state[i].clone())`
    const Val* external_state_after_internal = get_external_round_state_ptr(main_row, NUM_EXTERNAL_ROUNDS / 2);
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        folder_assert_zero(folder, external_state_after_internal[i] - state_computed[i]);

    }
}
