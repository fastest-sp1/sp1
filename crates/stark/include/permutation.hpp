//file: permutation.hpp
#pragma once
#include "gpu_types.hpp"
#include "virtual_pair_col.hpp"
#include "air_folder.hpp"

__device__ inline Challenge calculate_rlc(
    const Interaction& interaction,
    const Challenge& alpha,
    const Challenge& beta,
    const Val* main_row,
    const Val* prep_row
) {
    Challenge rlc = alpha;
    //Challenge beta_pow = beta;
    Challenge first_beta_pow = Challenge::one();

    rlc += first_beta_pow * Challenge(Val::from_canonical_u32(uint32_t(interaction.kind)));

    Challenge beta_pow = beta;
    for (int i = 0; i < interaction.num_values; ++i) {
        Val val = interaction.values[i].apply(main_row, prep_row);

        rlc += beta_pow * Challenge(val);
        beta_pow *= beta;
    }
    return rlc;
}


__device__  inline void eval_permutation_constraints(
      ProverConstraintFolder<Challenge>& folder,
      const Interaction* interaction_buffer,
      int num_interaction,
      const Val* main_row,
      const Val* prep_row,
      const Challenge* perm_local,
      const Challenge* perm_next,
      int perm_width,
      int batch_size,
      const Challenge* perm_challenges,
      const Challenge& local_cumulative_sum,
      Val is_first_row, 
      Val is_last_row, 
      Val is_transition
) {
      const Challenge& alpha = perm_challenges[0];
      const Challenge& beta = perm_challenges[1];

      int perm_width_ef = perm_width / 4;

      // The number of chunks is determined by the number of interactions.
      int num_chunks = (num_interaction + batch_size - 1) / batch_size;

      // The number of entries available for batch sum verification in the permutation trace.
      // It's the total width of the permutation trace minus the last column (which is phi).
      int num_entries_available = perm_width_ef > 0 ? perm_width_ef - 1 : 0;


      // The loop must run for the minimum of the number of available chunks and the number of available entries.
      // This exactly mimics the behavior of `zip`.
      int loop_iterations = min(num_chunks, num_entries_available);

      // === STAGE 1: BATCHED SUM VERIFICATION ===
      #define MAX_BATCH_SIZE 8 
      Challenge rlcs[MAX_BATCH_SIZE];
      Challenge multiplicities[MAX_BATCH_SIZE];

      for (int i = 0; i < loop_iterations; ++i) {
          const Challenge& entry = perm_local[i];
          
          int start_idx = i * batch_size;
          // chunk_len calculation is safe because we know the interactions exist due to the loop bound.
          int chunk_len = min(batch_size, num_interaction - start_idx);

          
          // 1. Collect RLCs and multiplicities for the current chunk.
          // This part of the logic remains unchanged.
          for (int j = 0; j < chunk_len; ++j) {
              const auto& interaction = interaction_buffer[start_idx + j];
              rlcs[j] = calculate_rlc(interaction, alpha, beta, main_row, prep_row);
              
              Val mult_val = interaction.multiplicity.apply(main_row, prep_row);
              multiplicities[j] = Challenge(mult_val) * (interaction.is_send ? Challenge::one() : Challenge(Val::neg_one()));
          }

          // 2. Calculate the combined constraint.
          // This part of the logic also remains unchanged.
          Challenge product = Challenge::one();
          for (int j = 0; j < chunk_len; ++j) {
              product = product * rlcs[j];
          }

          Challenge numerator = Challenge::zero();
          for (int j = 0; j < chunk_len; ++j) {
              Challenge all_but_current = Challenge::one();
              for (int k = 0; k < chunk_len; ++k) {
                  if (j != k) {
                      all_but_current = all_but_current * rlcs[k];
                  }
              }
              numerator = numerator + multiplicities[j] * all_but_current;
          }

          // 3. Apply the constraint.
          folder_assert_zero_ext(folder, product * entry - numerator);
 
      }
   
      // === STAGE 2: RUNNING SUM VERIFICATION (This part remains the same) ===
      if (perm_width_ef > 0) {
          Challenge sum_local = Challenge::zero();
          for (int i = 0; i < perm_width_ef - 1; ++i) sum_local += perm_local[i];
          
          Challenge sum_next = Challenge::zero();
          for (int i = 0; i < perm_width_ef - 1; ++i) sum_next += perm_next[i];

          const Challenge& phi_local = perm_local[perm_width_ef - 1];
          const Challenge& phi_next = perm_next[perm_width_ef - 1];

          folder_assert_zero_ext(folder, (phi_local - sum_local) * Challenge(is_first_row));
          folder_assert_zero_ext(folder, (phi_next - phi_local - sum_next) * Challenge(is_transition));
          folder_assert_zero_ext(folder, (phi_local - local_cumulative_sum) * Challenge(is_last_row));
      }
}

__host__  __device__ inline int get_base_alu_interactions(Interaction* interaction_buffer)
{
    int interaction_send_count = 0;
    int interaction_receive_idx = 4;

    // For the preprocessed trace
    constexpr size_t prep_accesses_base_offset = offsetof(BaseAluPreprocessedCols<Val>, accesses);
    constexpr size_t prep_access_struct_size = sizeof(BaseAluAccessCols<Val>);
    
    // For the main trace
    constexpr size_t main_values_base_offset = offsetof(BaseAluCols<Val>, values);
    constexpr size_t main_value_struct_size = sizeof(BaseAluValueCols<Val>);

    // Offsets of members *within* their own small struct
    constexpr size_t is_add_member_offset = offsetof(BaseAluAccessCols<Val>, is_add);
    constexpr size_t is_sub_member_offset = offsetof(BaseAluAccessCols<Val>, is_sub);
    constexpr size_t is_mul_member_offset = offsetof(BaseAluAccessCols<Val>, is_mul);
    constexpr size_t is_div_member_offset = offsetof(BaseAluAccessCols<Val>, is_div);
    constexpr size_t mult_member_offset = offsetof(BaseAluAccessCols<Val>, mult);
    constexpr size_t addr_out_member_offset = offsetof(BaseAluAccessCols<Val>, addrs.out.val);
    constexpr size_t addr_in1_member_offset = offsetof(BaseAluAccessCols<Val>, addrs.in1.val);
    constexpr size_t addr_in2_member_offset = offsetof(BaseAluAccessCols<Val>, addrs.in2.val);
    constexpr size_t val_out_member_offset = offsetof(BaseAluValueCols<Val>, vals.out);
    constexpr size_t val_in1_member_offset = offsetof(BaseAluValueCols<Val>, vals.in1);
    constexpr size_t val_in2_member_offset = offsetof(BaseAluValueCols<Val>, vals.in2);

    for (int i = 0; i < 4; ++i) { 
        size_t current_prep_op_base = prep_accesses_base_offset + i * prep_access_struct_size;
        size_t current_main_op_base = main_values_base_offset + i * main_value_struct_size;         
        
        VirtualPairCol vpc_is_add = VirtualPairCol::single_preprocessed((current_prep_op_base + is_add_member_offset) / sizeof(Val));
        VirtualPairCol vpc_is_sub = VirtualPairCol::single_preprocessed((current_prep_op_base + is_sub_member_offset) / sizeof(Val));
        VirtualPairCol vpc_is_mul = VirtualPairCol::single_preprocessed((current_prep_op_base + is_mul_member_offset) / sizeof(Val));
        VirtualPairCol vpc_is_div = VirtualPairCol::single_preprocessed((current_prep_op_base + is_div_member_offset) / sizeof(Val));

        VirtualPairCol vpc_is_real = vpc_is_add + vpc_is_sub + vpc_is_mul + vpc_is_div;
              
        //RLC: first computes the send intentions!!
        interaction_buffer[interaction_send_count++] = { // send(out) 
            .values = {
                VirtualPairCol::single_preprocessed((current_prep_op_base + addr_out_member_offset) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_out_member_offset) / sizeof(Val))
              }, 
            .num_values = 2, 
            //.multiplicity = VirtualPairCol::single_preprocessed(prep_base_idx + 7), 
            .multiplicity = VirtualPairCol::single_preprocessed((current_prep_op_base + mult_member_offset) / sizeof(Val)),
            .kind = InteractionKind::Memory,
            .scope = InteractionScope::Local,
            .is_send = true
          };

        interaction_buffer[interaction_receive_idx++] = { // receive(in1) 
                    .values = {
                        VirtualPairCol::single_preprocessed((current_prep_op_base + addr_in1_member_offset) / sizeof(Val)),
                        VirtualPairCol::single_main((current_main_op_base + val_in1_member_offset) / sizeof(Val))
                    }, 
                    .num_values = 2,
                    .multiplicity = vpc_is_real, // multiplicity for receive is `is_real`
                    .kind = InteractionKind::Memory,
                    .scope = InteractionScope::Local,
                    .is_send = false
          };

        interaction_buffer[interaction_receive_idx++] = { // receive(in2) 
                    .values = {
                        VirtualPairCol::single_preprocessed((current_prep_op_base + addr_in2_member_offset) / sizeof(Val)),
                        VirtualPairCol::single_main((current_main_op_base + val_in2_member_offset) / sizeof(Val))
                    }, 
                    .num_values = 2,
                    .multiplicity = vpc_is_real, // multiplicity for receive is `is_real`
                    .kind = InteractionKind::Memory,
                    .scope = InteractionScope::Local,
                    .is_send = false
                };
    }

    return interaction_receive_idx;
}

__host__  __device__ inline int get_ext_alu_interactions(Interaction* interaction_buffer) 
{
    int interaction_send_idx = 0;
    int interaction_receive_idx = 4;

    // For the preprocessed trace
    constexpr size_t prep_accesses_base_offset = offsetof(ExtAluPreprocessedCols<Val>, accesses);
    constexpr size_t prep_access_struct_size = sizeof(ExtAluAccessCols<Val>);
    
    // For the main trace
    constexpr size_t main_values_base_offset = offsetof(ExtAluCols<Val>, values);
    constexpr size_t main_value_struct_size = sizeof(ExtAluValueCols<Val>);

    // Offsets of members *within* their own small struct
    constexpr size_t is_add_member_offset = offsetof(ExtAluAccessCols<Val>, is_add);
    constexpr size_t is_sub_member_offset = offsetof(ExtAluAccessCols<Val>, is_sub);
    constexpr size_t is_mul_member_offset = offsetof(ExtAluAccessCols<Val>, is_mul);
    constexpr size_t is_div_member_offset = offsetof(ExtAluAccessCols<Val>, is_div);
    constexpr size_t mult_member_offset = offsetof(ExtAluAccessCols<Val>, mult);
    
    // Base offsets for addresses and values within their containers
    constexpr size_t addr_out_base_offset = offsetof(ExtAluAccessCols<Val>, addrs.out);
    constexpr size_t addr_in1_base_offset = offsetof(ExtAluAccessCols<Val>, addrs.in1);
    constexpr size_t addr_in2_base_offset = offsetof(ExtAluAccessCols<Val>, addrs.in2);
    constexpr size_t val_out_base_offset = offsetof(ExtAluValueCols<Val>, vals.out);
    constexpr size_t val_in1_base_offset = offsetof(ExtAluValueCols<Val>, vals.in1);
    constexpr size_t val_in2_base_offset = offsetof(ExtAluValueCols<Val>, vals.in2);

    for (int i = 0; i < 4; ++i) {
        // --- Calculate the dynamic base offsets for the current operation `i` ---
        size_t current_prep_op_base = prep_accesses_base_offset + i * prep_access_struct_size;
        size_t current_main_op_base = main_values_base_offset + i * main_value_struct_size;    
        
        VirtualPairCol vpc_is_add = VirtualPairCol::single_preprocessed((current_prep_op_base + is_add_member_offset) / sizeof(Val));
        VirtualPairCol vpc_is_sub = VirtualPairCol::single_preprocessed((current_prep_op_base + is_sub_member_offset) / sizeof(Val));
        VirtualPairCol vpc_is_mul = VirtualPairCol::single_preprocessed((current_prep_op_base + is_mul_member_offset) / sizeof(Val));
        VirtualPairCol vpc_is_div = VirtualPairCol::single_preprocessed((current_prep_op_base + is_div_member_offset) / sizeof(Val));
        
        
        VirtualPairCol vpc_is_real = vpc_is_add + vpc_is_sub + vpc_is_mul + vpc_is_div;

          //RLC: first computes the send intentions!!
          interaction_buffer[interaction_send_idx++] = { // send(out) 
              .values = {
                  // Address<T> has 1 element, Block<T> has 4 elements
                VirtualPairCol::single_preprocessed((current_prep_op_base + addr_out_base_offset) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_out_base_offset + 0 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_out_base_offset + 1 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_out_base_offset + 2 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_out_base_offset + 3 * sizeof(Val)) / sizeof(Val))
              }, 
              .num_values = 5, 
              //.multiplicity = VirtualPairCol::single_preprocessed(prep_base_idx + 7), 
              .multiplicity = VirtualPairCol::single_preprocessed((current_prep_op_base + mult_member_offset) / sizeof(Val)),
              .kind = InteractionKind::Memory,
              .scope = InteractionScope::Local,
              .is_send = true
          };

          interaction_buffer[interaction_receive_idx++] = { // receive(in1) 
              .values = {
                VirtualPairCol::single_preprocessed((current_prep_op_base + addr_in1_base_offset) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_in1_base_offset + 0 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_in1_base_offset + 1 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_in1_base_offset + 2 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_in1_base_offset + 3 * sizeof(Val)) / sizeof(Val))
              }, 
              .num_values = 5,
              .multiplicity = vpc_is_real, 
              .kind = InteractionKind::Memory,
              .scope = InteractionScope::Local,
              .is_send = false
          };
          interaction_buffer[interaction_receive_idx++] = { // receive(in2) 
              .values = {
                VirtualPairCol::single_preprocessed((current_prep_op_base + addr_in2_base_offset) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_in2_base_offset + 0 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_in2_base_offset + 1 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_in2_base_offset + 2 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((current_main_op_base + val_in2_base_offset + 3 * sizeof(Val)) / sizeof(Val))
              }, 
              .num_values = 5,
              .multiplicity = vpc_is_real, // multiplicity for receive is `is_real`
              .kind = InteractionKind::Memory,
              .scope = InteractionScope::Local,
              .is_send = false
          };
    }

    return interaction_receive_idx;
}

__host__ __device__ inline int get_p2_wide_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;

    // a) Input memory interactions
    constexpr size_t input_base_offset = offsetof(Poseidon2PreprocessedColsWide<Val>, input);
    // The size of one element in the `input` array.
    // In this case, it's an Address<Val>.
    constexpr size_t input_element_size = sizeof(Address<Val>);
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
         // --- Calculate the dynamic part of the offset at runtime ---
        size_t current_element_offset = input_base_offset + i * input_element_size;

        interaction_buffer[interaction_count++] = {
            .values = {
                VirtualPairCol::single_preprocessed(current_element_offset / sizeof(Val)),
                VirtualPairCol::single_main(offsetof(Poseidon2Degree3Cols<Val>, state.external_rounds_state) / sizeof(Val) + i)
                //VirtualPairCol::from_constant(Val::zero()),
                //VirtualPairCol::from_constant(Val::zero()),
                //VirtualPairCol::from_constant(Val::zero()) 
            }, 
            .num_values = 2,
            .multiplicity = VirtualPairCol::single_preprocessed(offsetof(Poseidon2PreprocessedColsWide<Val>, is_real_neg) / sizeof(Val)),
            .kind = InteractionKind::Memory, 
            .scope = InteractionScope::Local, 
            .is_send = true
        };
    }

    // b) Output memory interactions
    constexpr size_t output_base_offset = offsetof(Poseidon2PreprocessedColsWide<Val>, output);
    constexpr size_t output_element_size = sizeof(MemoryAccessCols<Val>);
    for (int i = 0; i < POSEIDON2_STATE_WIDTH; ++i) {
        size_t current_addr_offset = output_base_offset + i * output_element_size + offsetof(MemoryAccessCols<Val>, addr);
        size_t current_mult_offset = output_base_offset + i * output_element_size + offsetof(MemoryAccessCols<Val>, mult);
        interaction_buffer[interaction_count++] = {
            .values = {
                VirtualPairCol::single_preprocessed(current_addr_offset / sizeof(Val)),
                VirtualPairCol::single_main(offsetof(Poseidon2Degree3Cols<Val>, state.output_state) / sizeof(Val) + i) // Need correct offset for output
                //VirtualPairCol::from_constant(Val::zero()),
               // VirtualPairCol::from_constant(Val::zero()),
                //VirtualPairCol::from_constant(Val::zero()) 
            }, 
            .num_values = 2,
            .multiplicity = VirtualPairCol::single_preprocessed(current_mult_offset / sizeof(Val)),
            .kind = InteractionKind::Memory, 
            .scope = InteractionScope::Local, 
            .is_send = true
        };
    }

    return interaction_count;
}

__host__ __device__ inline int get_p2_skinny_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;
    // Base offset of the `memory_preprocessed` array within the preprocessed struct.
    constexpr size_t mem_prep_base_offset = offsetof(Poseidon2PreprocessedColsSkinny<Val>, memory_preprocessed);
    
    // Size of a single `MemoryAccessCols<Val>` element in that array.
    constexpr size_t mem_access_struct_size = sizeof(MemoryAccessCols<Val>);

    // Constant offsets of the `addr` and `mult` members within the `MemoryAccessCols` struct.
    constexpr size_t addr_member_offset = offsetof(MemoryAccessCols<Val>, addr);
    constexpr size_t mult_member_offset = offsetof(MemoryAccessCols<Val>, mult);
    
    // Base offset of the `state_var` array within the main trace struct.
    constexpr size_t state_var_base_offset = offsetof(Poseidon2SkinnyCols<Val>, state_var);
    
    // Size of a single `Val` element in the `state_var` array.
    constexpr size_t state_var_element_size = sizeof(Val);

    for (int i=0; i < POSEIDON2_STATE_WIDTH; ++i) {
        // Offset for the i-th `MemoryAccessCols` struct.
        size_t current_mem_prep_base = mem_prep_base_offset + i * mem_access_struct_size;
        
        // Total offset for the `addr` field of the i-th struct.
        size_t current_addr_offset = current_mem_prep_base + addr_member_offset;
        
        // Total offset for the `mult` field of the i-th struct.
        size_t current_mult_offset = current_mem_prep_base + mult_member_offset;
        
        // Total offset for the i-th element of the `state_var` array.
        size_t current_state_var_offset = state_var_base_offset + i * state_var_element_size;

        interaction_buffer[interaction_count++] = {
            .values = {
                VirtualPairCol::single_preprocessed(current_addr_offset / sizeof(Val)),
                VirtualPairCol::single_main(current_state_var_offset / sizeof(Val))
            }, 
            .num_values = 2,
            .multiplicity = VirtualPairCol::single_preprocessed(current_mult_offset / sizeof(Val)),
            .kind = InteractionKind::Memory, 
            .scope = InteractionScope::Local, 
            .is_send = true
        };
    }
    return interaction_count;
}


__host__  __device__ inline int get_select_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;

    //first count the send.
    // builder.send_single(prep_local.addrs.out1, local.vals.out1, prep_local.mult1);
    interaction_buffer[interaction_count++] = {
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, addrs.out1) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(SelectCols<Val>, vals.out1) / sizeof(Val))
           
        },
        .num_values = 2, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, mult1) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true // send
    };
    // builder.send_single(prep_local.addrs.out2, local.vals.out2, prep_local.mult2);
    interaction_buffer[interaction_count++] = {
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, addrs.out2) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(SelectCols<Val>, vals.out2)/ sizeof(Val))
            
        },
        .num_values = 2, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, mult2) / sizeof(Val)),
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true // send
    };

    interaction_buffer[interaction_count++] = {
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, addrs.bit) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(SelectCols<Val>, vals.bit) / sizeof(Val))
           
        },
        .num_values = 2, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, is_real) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = false 
    };
    // builder.receive_single(prep_local.addrs.in1, local.vals.in1, prep_local.is_real);
    interaction_buffer[interaction_count++] = {
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, addrs.in1) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(SelectCols<Val>, vals.in1) / sizeof(Val))
            
        },
        .num_values = 2, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, is_real) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = false // receive
    };
    // builder.receive_single(prep_local.addrs.in2, local.vals.in2, prep_local.is_real);
    interaction_buffer[interaction_count++] = {
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, addrs.in2) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(SelectCols<Val>, vals.in2) / sizeof(Val)) 
            
        },
        .num_values = 2, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(SelectPreprocessedCols<Val>, is_real) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = false // receive
    };
    
    return interaction_count;
}

__host__  __device__ inline int get_batch_fri_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;
    // Send `acc` with multiplicity `is_end`
    interaction_buffer[interaction_count++] = {
        .values = {
            VirtualPairCol::single_preprocessed(offsetof(BatchFRIPreprocessedCols<Val>, acc_addr) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, acc._0[0]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, acc._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, acc._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, acc._0[3]) / sizeof(Val))
         
        },
        .num_values = 5,
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(BatchFRIPreprocessedCols<Val>, is_end) / sizeof(Val)),
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true
    };
    
    // Receive `alpha_pow` with multiplicity `is_real`
    interaction_buffer[interaction_count++] = {
        .values = {
            VirtualPairCol::single_preprocessed(offsetof(BatchFRIPreprocessedCols<Val>, alpha_pow_addr) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, alpha_pow._0[0]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, alpha_pow._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, alpha_pow._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, alpha_pow._0[3]) / sizeof(Val))
        },
        .num_values = 5,
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(BatchFRIPreprocessedCols<Val>, is_real) / sizeof(Val)),
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = false
    };

    // Receive `p_at_z` with multiplicity `is_real`
    interaction_buffer[interaction_count++] = {
        .values = {
            VirtualPairCol::single_preprocessed(offsetof(BatchFRIPreprocessedCols<Val>, p_at_z_addr) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, p_at_z._0[0]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, p_at_z._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, p_at_z._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, p_at_z._0[3]) / sizeof(Val))
             
        },
        .num_values = 5,
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(BatchFRIPreprocessedCols<Val>, is_real) / sizeof(Val)),
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = false
    };

    // Receive `p_at_x` with multiplicity `is_real`
    interaction_buffer[interaction_count++] = {
        .values = {
            VirtualPairCol::single_preprocessed(offsetof(BatchFRIPreprocessedCols<Val>, p_at_x_addr) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(BatchFRICols<Val>, p_at_x) / sizeof(Val))
      
        },
        .num_values = 2, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(BatchFRIPreprocessedCols<Val>, is_real) / sizeof(Val)),
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = false
    };

    return interaction_count;
}


__host__  __device__ inline int get_public_values_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;
    // Constrain mem read for the public value element.
    // builder.send_single(local_prepr.pv_mem.addr, local.pv_element, local_prepr.pv_mem.mult);
    interaction_buffer[interaction_count++] = {
        .values = {
            VirtualPairCol::single_preprocessed(offsetof(PublicValuesPreprocessedCols<Val>, pv_mem.addr) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(PublicValuesCols<Val>, pv_element) / sizeof(Val))
            
        },
        .num_values = 2,
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(PublicValuesPreprocessedCols<Val>, pv_mem.mult) / sizeof(Val)),
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true
    };

    return interaction_count;
}


__host__  __device__ inline int get_exp_reverse_fri_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;
    // a) builder.send_single(local_prepr.x_mem.addr, local.x, local_prepr.x_mem.mult);
    //Send x (base) value. Multiplicity is -1 on the first row, 0 otherwise.
    interaction_buffer[interaction_count++] = {
        .values = {
            VirtualPairCol::single_preprocessed(offsetof(ExpReverseBitsLenPreprocessedCols<Val>, x_mem.addr) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(ExpReverseBitsLenCols<Val>, x) / sizeof(Val))
        }, 
        .num_values = 2,
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(ExpReverseBitsLenPreprocessedCols<Val>, x_mem.mult) / sizeof(Val)),
        .kind = InteractionKind::Memory, 
        .scope = InteractionScope::Local, 
        .is_send = true
    };
    
    // b) Send current_bit (from exponent). Multiplicity is -1 for all real rows.
    interaction_buffer[interaction_count++] = {
        .values = {
            VirtualPairCol::single_preprocessed(offsetof(ExpReverseBitsLenPreprocessedCols<Val>, exponent_mem.addr) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(ExpReverseBitsLenCols<Val>, current_bit) / sizeof(Val))

        }, 
        .num_values = 2,
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(ExpReverseBitsLenPreprocessedCols<Val>, exponent_mem.mult) / sizeof(Val)),
        .kind = InteractionKind::Memory, 
        .scope = InteractionScope::Local, 
        .is_send = true
    };
    
    // c) Send accum (result). Multiplicity is `mult` on the last row, 0 otherwise.
    interaction_buffer[interaction_count++] = {
        .values = {
            VirtualPairCol::single_preprocessed(offsetof(ExpReverseBitsLenPreprocessedCols<Val>, result_mem.addr) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(ExpReverseBitsLenCols<Val>, accum) / sizeof(Val))
  
        }, 
        .num_values = 2,
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(ExpReverseBitsLenPreprocessedCols<Val>, result_mem.mult) / sizeof(Val)),
        .kind = InteractionKind::Memory, 
        .scope = InteractionScope::Local, 
        .is_send = true
    };
    return interaction_count;
}

__host__  __device__ inline int get_fri_fold_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;
        // Reads for x , z, alpha
    interaction_buffer[interaction_count++] = { 
        .values = { 
                VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, x_mem.addr) / sizeof(Val)),  
                VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, x) / sizeof(Val))
                //VirtualPairCol::from_constant(Val::zero()),
                //VirtualPairCol::from_constant(Val::zero()),
                //VirtualPairCol::from_constant(Val::zero()) 
        }, 
        .num_values = 2, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, x_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true 
    };

    interaction_buffer[interaction_count++] = { 
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, z_mem.addr) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, z._0) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, z._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, z._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, z._0[3]) / sizeof(Val))  
        }, 
        .num_values = 5, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, z_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true
    };

    interaction_buffer[interaction_count++] = { 
        .values = { 
                VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, alpha_mem.addr) / sizeof(Val)),       
                VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha._0) / sizeof(Val)), 
                VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha._0[1]) / sizeof(Val)),
                VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha._0[2]) / sizeof(Val)),
                VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha._0[3]) / sizeof(Val)) 
            }, 
        .num_values = 5, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, alpha_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true 
    };
    

    // Constrain write  alpha_pow_input.
    interaction_buffer[interaction_count++] = { 
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, alpha_pow_input_mem.addr) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha_pow_input._0) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha_pow_input._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha_pow_input._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha_pow_input._0[3]) / sizeof(Val))
        }, 
        .num_values = 5, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, alpha_pow_input_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true 
    };

    //Constrain write ro_input
    interaction_buffer[interaction_count++] = { 
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, ro_input_mem.addr) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, ro_input._0[0]) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, ro_input._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, ro_input._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, ro_input._0[3]) / sizeof(Val))
        }, 
        .num_values = 5, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, ro_input_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true 
    };

    //Constrain write  p_at_z
    interaction_buffer[interaction_count++] = { 
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, p_at_z_mem.addr) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, p_at_z._0[0]) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, p_at_z._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, p_at_z._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, p_at_z._0[3]) / sizeof(Val))
        }, 
        .num_values = 5, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, p_at_z_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true 
    };

    // Constrain write  p_at_x
    interaction_buffer[interaction_count++] = { 
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, p_at_x_mem.addr) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, p_at_x._0[0]) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, p_at_x._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, p_at_x._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, p_at_x._0[3]) / sizeof(Val)) 
        }, 
        .num_values = 5, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, p_at_x_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true 
    };
    
    
    // Writes for output vectors
    interaction_buffer[interaction_count++] = { 
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, alpha_pow_output_mem.addr) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha_pow_output._0[0]) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha_pow_output._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha_pow_output._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, alpha_pow_output._0[3]) / sizeof(Val))
        }, 
        .num_values = 5, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, alpha_pow_output_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true 
    };

    interaction_buffer[interaction_count++] = { 
        .values = { 
            VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, ro_output_mem.addr) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, ro_output._0[0]) / sizeof(Val)), 
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, ro_output._0[1]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, ro_output._0[2]) / sizeof(Val)),
            VirtualPairCol::single_main(offsetof(FriFoldCols<Val>, ro_output._0[3]) / sizeof(Val))
        }, 
        .num_values = 5, 
        .multiplicity = VirtualPairCol::single_preprocessed(offsetof(FriFoldPreprocessedCols<Val>, ro_output_mem.mult) / sizeof(Val)), 
        .kind = InteractionKind::Memory,
        .scope = InteractionScope::Local,
        .is_send = true 
    };

    return interaction_count;
}

// Helper function to create interactions for MemoryConst
__host__ __device__ inline int get_memory_const_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;

    // The chip has two interaction slots per row.
    for (int i = 0; i < NUM_CONST_MEM_ENTRIES_PER_ROW; ++i) {
        // --- Pre-calculate constant base offsets and sizes ---
        constexpr size_t base_offset = offsetof(MemoryPreprocessedCols<Val>, values_and_accesses);
        constexpr size_t pair_size = sizeof(ValueAccessPair<Val>);
        
        // --- Calculate dynamic offsets for the current slot `i` ---
        size_t current_pair_offset = base_offset + i * pair_size;
        
        size_t addr_offset = current_pair_offset + offsetof(ValueAccessPair<Val>, access.addr);
        size_t mult_offset = current_pair_offset + offsetof(ValueAccessPair<Val>, access.mult);
        size_t value_offset = current_pair_offset + offsetof(ValueAccessPair<Val>, value);

        // builder.send_block(access.addr, value, access.mult);
        interaction_buffer[interaction_count++] = {
            .values = {
                // This interaction is for a Block<T>, which is 4 Vals + 1 Address Val.
                VirtualPairCol::single_preprocessed(addr_offset / sizeof(Val)),
                VirtualPairCol::single_preprocessed((value_offset + 0 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_preprocessed((value_offset + 1 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_preprocessed((value_offset + 2 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_preprocessed((value_offset + 3 * sizeof(Val)) / sizeof(Val)),
            },
            .num_values = 5,
            .multiplicity = VirtualPairCol::single_preprocessed(mult_offset / sizeof(Val)),
            .kind = InteractionKind::Memory,
            .scope = InteractionScope::Local,
            .is_send = true
        };
    }

    return interaction_count;
}

// Helper function to create interactions for MemoryVar
__host__ __device__ inline int get_memory_var_interactions(Interaction* interaction_buffer) {
    int interaction_count = 0;

    // The chip has two interaction slots per row.
    for (int i = 0; i < NUM_VAR_MEM_ENTRIES_PER_ROW; ++i) {
        // --- Pre-calculate constant base offsets and struct sizes ---
        constexpr size_t prep_base_offset = offsetof(MemoryVarPreprocessedCols<Val>, accesses);
        constexpr size_t prep_element_size = sizeof(MemoryAccessCols<Val>);
        
        constexpr size_t main_base_offset = offsetof(MemoryVarCols<Val>, values);
        constexpr size_t main_element_size = sizeof(Block<Val>);

        // --- Calculate dynamic offsets for the current slot `i` ---
        size_t current_prep_base = prep_base_offset + i * prep_element_size;
        size_t current_main_base = main_base_offset + i * main_element_size;
        
        size_t addr_offset = current_prep_base + offsetof(MemoryAccessCols<Val>, addr);
        size_t mult_offset = current_prep_base + offsetof(MemoryAccessCols<Val>, mult);
        size_t value_offset = current_main_base; // `value` is the start of the Block<T>

        // builder.send_block(access.addr, value, access.mult);
        interaction_buffer[interaction_count++] = {
            .values = {
                // This interaction is for a Block<T>, which is 4 Vals + 1 Address Val.
                VirtualPairCol::single_preprocessed(addr_offset / sizeof(Val)),
                VirtualPairCol::single_main((value_offset + 0 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((value_offset + 1 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((value_offset + 2 * sizeof(Val)) / sizeof(Val)),
                VirtualPairCol::single_main((value_offset + 3 * sizeof(Val)) / sizeof(Val)),
            },
            .num_values = 5,
            .multiplicity = VirtualPairCol::single_preprocessed(mult_offset / sizeof(Val)),
            .kind = InteractionKind::Memory,
            .scope = InteractionScope::Local,
            .is_send = true
        };
    }

    return interaction_count;
}
