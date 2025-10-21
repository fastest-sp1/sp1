#pragma once
//#include "air_folder.hpp"
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"
#include "permutation.hpp"

__device__ void eval_memory_var_chip(
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
    // This chip has no arithmetic constraints, only memory interactions.
    Interaction interaction_buffer[NUM_CONST_MEM_ENTRIES_PER_ROW];
    int num_interactions = get_memory_var_interactions(interaction_buffer);

    const Challenge* perm_local = reinterpret_cast<const Challenge*>(perm_local_flat);
    const Challenge* perm_next = reinterpret_cast<const Challenge*>(perm_next_flat);

    //refactor
    eval_permutation_constraints(
          folder,
          interaction_buffer,
          num_interactions,
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


    // Note: No global sum verification, because MemoryConstant is `InteractionScope::Local`.
}

