// file: base_alu_eval.hpp
#pragma once
#include "air_folder.hpp"
#include "gpu_types.hpp"
//#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "virtual_pair_col.hpp"

#include "permutation.hpp"


__device__ void eval_base_alu_chip(
    ProverConstraintFolder<Challenge>& folder,
    //const BaseAluCols<Val>& main_cols,
    //const BaseAluPreprocessedCols<Val>& prep_cols,
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

    const Challenge* perm_local = reinterpret_cast<const Challenge*>(perm_local_flat);
    const Challenge* perm_next = reinterpret_cast<const Challenge*>(perm_next_flat);

    const auto* main_cols = reinterpret_cast<const BaseAluCols<Val>*>(main_local_row);
    const auto* prep_cols = reinterpret_cast<const BaseAluPreprocessedCols<Val>*>(prep_local_row);
    
    // We have 12 interactions total. With batch_size=2, we have 6 chunks/batches.
    // interaction_buffer will hold the components for all 12 interactions.
    Interaction interaction_buffer[12];

    Challenge rlcs[12];
    Challenge multiplicities[12];
   
    for (int i = 0; i < 4; ++i) {
         const auto& vals = main_cols->values[i].vals;
         const auto& access = prep_cols->accesses[i];
          
          // These VirtualPairCols are created dynamically, just like in Rust's symbolic builder.
          // Note: The column indices (e.g., `offsetof(...)`) must be correct.
          VirtualPairCol vpc_is_add = VirtualPairCol::single_preprocessed(offsetof(BaseAluPreprocessedCols<Val>, accesses[i].is_add) / sizeof(Val));
          VirtualPairCol vpc_is_sub = VirtualPairCol::single_preprocessed(offsetof(BaseAluPreprocessedCols<Val>, accesses[i].is_sub) / sizeof(Val));
          VirtualPairCol vpc_is_mul = VirtualPairCol::single_preprocessed(offsetof(BaseAluPreprocessedCols<Val>, accesses[i].is_mul) / sizeof(Val));
          VirtualPairCol vpc_is_div = VirtualPairCol::single_preprocessed(offsetof(BaseAluPreprocessedCols<Val>, accesses[i].is_div) / sizeof(Val));

          VirtualPairCol vpc_is_real = vpc_is_add + vpc_is_sub + vpc_is_mul + vpc_is_div;

          // INTERNAL CONSTRAINTS (5 constraints per op)
          Val is_real = vpc_is_real.apply(main_local_row, prep_local_row);
         
          folder_assert_bool(folder, is_real);

          folder_assert_zero(folder, (vals.in1 + vals.in2 - vals.out) * access.is_add);
          folder_assert_zero(folder, (vals.in1 - vals.in2 - vals.out ) * access.is_sub);
          folder_assert_zero(folder, (vals.out - vals.in1 * vals.in2 ) * access.is_mul);
          folder_assert_zero(folder, (vals.in2 * vals.out - vals.in1) * access.is_div); 
    }

    int num_interactions = get_base_alu_interactions(interaction_buffer);

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

    // Note: No global sum verification, because BaseAluChip is `InteractionScope::Local`.
}
  
