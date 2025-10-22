#![allow(clippy::needless_range_loop)]

use crate::{
    Address, BatchFRIEvent, BatchFRIInstr, ExecutionRecord, Instruction, air::Block,
    builder::SP1RecursionAirBuilder,
};
use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use sp1_core_machine::utils::next_power_of_two;
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{BaseAirBuilder, BinomialExtension, ExtensionAirBuilder, MachineAir};

use std::borrow::BorrowMut;
use tracing::instrument;

use itertools::Itertools;
//use sp1_stark::GpuMatrix;

pub const NUM_BATCH_FRI_COLS: usize = core::mem::size_of::<BatchFRICols<u8>>();
pub const NUM_BATCH_FRI_PREPROCESSED_COLS: usize =
    core::mem::size_of::<BatchFRIPreprocessedCols<u8>>();

#[derive(Clone, Debug, Copy, Default)]
pub struct BatchFRIChip<const DEGREE: usize>;

/// The preprocessed columns for a batch FRI invocation.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BatchFRIPreprocessedCols<T: Copy> {
    pub is_real: T,
    pub is_end: T,
    pub acc_addr: Address<T>,
    pub alpha_pow_addr: Address<T>,
    pub p_at_z_addr: Address<T>,
    pub p_at_x_addr: Address<T>,
}

/// The main columns for a batch FRI invocation.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct BatchFRICols<T: Copy> {
    pub acc: Block<T>,
    pub alpha_pow: Block<T>,
    pub p_at_z: Block<T>,
    pub p_at_x: T,
}

impl<F, const DEGREE: usize> BaseAir<F> for BatchFRIChip<DEGREE> {
    fn width(&self) -> usize {
        NUM_BATCH_FRI_COLS
    }
}

impl<F: PrimeField32, const DEGREE: usize> MachineAir<F> for BatchFRIChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "BatchFRI".to_string()
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn preprocessed_width(&self) -> usize {
        NUM_BATCH_FRI_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_preprocessed_trace only supports BabyBear field"
        );

        let instrs: Vec<&BatchFRIInstr<BabyBear>> = program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::BatchFRI(x) => Some(unsafe {
                    // Transmute Box<BatchFRIInstr<F>> to Box<BatchFRIInstr<BabyBear>>
                    // Then get reference from Box to FriFoldInstr
                    std::mem::transmute::<&BatchFRIInstr<F>, &BatchFRIInstr<BabyBear>>(x.as_ref())
                }),
                _ => None,
            })
            .collect_vec();

        if instrs.is_empty() {
            let values = vec![BabyBear::ZERO; NUM_BATCH_FRI_PREPROCESSED_COLS];
            return Some(RowMajorMatrix::new(
                unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
                NUM_BATCH_FRI_PREPROCESSED_COLS,
            ));
        }

        let mut values: Vec<BabyBear>;

        let mut cpu_rows: Vec<[BabyBear; NUM_BATCH_FRI_PREPROCESSED_COLS]> = Vec::new();
        instrs.iter().for_each(|instruction| {
            let BatchFRIInstr { base_vec_addrs: _, ext_single_addrs: _, ext_vec_addrs, acc_mult } =
                *instruction;
            let len: usize = ext_vec_addrs.p_at_z.len();
            let mut row_add = vec![[BabyBear::ZERO; NUM_BATCH_FRI_PREPROCESSED_COLS]; len];
            debug_assert_eq!(*acc_mult, BabyBear::ONE);

            row_add.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut BatchFRIPreprocessedCols<BabyBear> = row.as_mut_slice().borrow_mut();
                unsafe {
                    crate::sys::batch_fri_instr_to_row_babybear(
                        &(&(*(*instruction))).into(),
                        cols,
                        i,
                    );
                }
            });
            cpu_rows.extend(row_add);
        });
        values = cpu_rows.into_iter().flatten().collect::<Vec<BabyBear>>();

        // Pad the trace to a power of two.
        if program.fixed_log2_rows(self).is_some() || values.len() > 0 {
            let current_num_rows = values.len() / NUM_BATCH_FRI_PREPROCESSED_COLS;
            let padded_num_rows =
                next_power_of_two(current_num_rows, program.fixed_log2_rows(self));
            let target_total_elements = padded_num_rows * NUM_BATCH_FRI_PREPROCESSED_COLS;
            values.resize(target_total_elements, BabyBear::ZERO);
        }

        let trace = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_BATCH_FRI_PREPROCESSED_COLS,
        );
        Some(trace)
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.batch_fri_events;
        Some(next_power_of_two(events.len(), input.fixed_log2_rows(self)))
    }

    #[instrument(name = "generate batch fri trace", level = "debug", skip_all, fields(rows = input.batch_fri_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _: &mut ExecutionRecord<F>,
    ) -> RowMajorMatrix<F> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_trace only supports BabyBear field"
        );

        let events = unsafe {
            std::mem::transmute::<&Vec<BatchFRIEvent<F>>, &Vec<BatchFRIEvent<BabyBear>>>(
                &input.batch_fri_events,
            )
        };

        let mut values: Vec<BabyBear>;

        let mut cpu_rows = Vec::new();
        events.iter().for_each(|bb_event| {
            let mut row = [BabyBear::ZERO; NUM_BATCH_FRI_COLS];
            let cols: &mut BatchFRICols<BabyBear> = row.as_mut_slice().borrow_mut();
            cols.acc = bb_event.ext_single.acc;
            cols.alpha_pow = bb_event.ext_vec.alpha_pow;
            cols.p_at_z = bb_event.ext_vec.p_at_z;
            cols.p_at_x = bb_event.base_vec.p_at_x;
            cpu_rows.push(row);
        });
        values = cpu_rows.into_iter().flatten().collect::<Vec<BabyBear>>();

        // Pad the trace to a power of two.
        let padded_num_rows = self.num_rows(input).unwrap();
        let target_total_elements = padded_num_rows * NUM_BATCH_FRI_COLS;
        values.resize(target_total_elements, BabyBear::ZERO);

        // Convert the trace to a row major matrix.
        let trace = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_BATCH_FRI_COLS,
        );

        #[cfg(debug_assertions)]
        eprintln!(
            "batch fri trace dims is width: {:?}, height: {:?}",
            trace.width(),
            trace.height()
        );

        trace
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    /*
    fn generate_trace_gpu(&self, input: &Self::Record, _: &mut Self::Record) -> GpuMatrix<F> {
        let events  = unsafe {
                std::mem::transmute::<&Vec<BatchFRIEvent<F>>, &Vec<BatchFRIEvent<BabyBear>>>(
                    &input.batch_fri_events,
            )
        };
        let padded_nb_rows = self.num_rows(input).unwrap();
        let num_cols = <Self as BaseAir<F>>::width(self);

        // 1. Allocate the matrix directly on the GPU.
        let mut gpu_matrix = GpuMatrix::<F>::new(padded_nb_rows, num_cols);

        if !events.is_empty() {
            unsafe {
                crate::sys::process_batch_fri_events_gpu(
                    events.as_ptr(),
                    events.len(),
                    gpu_matrix.as_mut_ptr() as *mut BabyBear, // Pass the device pointer
                    gpu_matrix.height * gpu_matrix.width, // Pass total elements
                    NUM_BATCH_FRI_COLS,
                );
            }
        }

        // 3. Return the GpuMatrix handle.
        gpu_matrix
    }

    fn generate_preprocessed_trace_gpu(
        &self,
        program: &Self::Program,
    ) -> Option<GpuMatrix<F>> {
        let instrs: Vec<&BatchFRIInstr<BabyBear>> = program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::BatchFRI(x) => Some(unsafe {
                    // Transmute Box<BatchFRIInstr<F>> to Box<BatchFRIInstr<BabyBear>>
                    // Then get reference from Box to FriFoldInstr
                    std::mem::transmute::<&BatchFRIInstr<F>, &BatchFRIInstr<BabyBear>>(x.as_ref())
                }),
                _ => None,
            })
            .collect_vec();
        //let mut values: Vec<BabyBear>;
        if !instrs.is_empty() {
            let mut all_base_p_at_x: Vec<Address<BabyBear>> = Vec::new();
            let mut all_ext_p_at_z: Vec<Address<BabyBear>> = Vec::new();
            let mut all_ext_alpha_pow: Vec<Address<BabyBear>> = Vec::new();
            let mut instrs_values: Vec<crate::BatchFRIInstrFlat<BabyBear>> = Vec::with_capacity(instrs.len());

            let mut current_base_p_at_x_offset = 0;
            let mut current_ext_p_at_z_offset = 0;
            let mut current_ext_alpha_pow_offset = 0;

            let mut instrs_index_info: Vec<crate::InstrsFlatIdex> = Vec::new();

            let mut num_total_output_rows = 0;
            for (instr_idx, &instr_ref) in instrs.iter().enumerate() {
                let instr_data = instr_ref;
                let p_at_z_len = instr_data.ext_vec_addrs.p_at_z.len();
                num_total_output_rows += p_at_z_len;
                for j in 0..p_at_z_len {
                     instrs_index_info.push(InstrsFlatIdex {
                            instr_idx: instr_idx,
                            arr_idx: j ,
                           // last_row: p_at_z_len,
                        });
                }

                all_base_p_at_x.extend_from_slice(&instr_data.base_vec_addrs.p_at_x);
                all_ext_p_at_z.extend_from_slice(&instr_data.ext_vec_addrs.p_at_z);
                all_ext_alpha_pow.extend_from_slice(&instr_data.ext_vec_addrs.alpha_pow);

                instrs_values.push(crate::BatchFRIInstrFlat {
                    base_p_at_x_offset: current_base_p_at_x_offset ,
                    base_p_at_x_len: instr_data.base_vec_addrs.p_at_x.len() ,
                    ext_single_addrs_acc_val: instr_data.ext_single_addrs,
                    ext_p_at_z_offset: current_ext_p_at_z_offset ,
                    ext_p_at_z_len: instr_data.ext_vec_addrs.p_at_z.len() ,
                    ext_alpha_pow_offset: current_ext_alpha_pow_offset ,
                    ext_alpha_pow_len: instr_data.ext_vec_addrs.alpha_pow.len() ,
                    acc_mult_val: instr_data.acc_mult,
                });

                current_base_p_at_x_offset += instr_data.base_vec_addrs.p_at_x.len();
                current_ext_p_at_z_offset += instr_data.ext_vec_addrs.p_at_z.len();
                current_ext_alpha_pow_offset += instr_data.ext_vec_addrs.alpha_pow.len();
            }

            let padded_num_rows = next_power_of_two(num_total_output_rows, program.fixed_log2_rows(self));

            let num_cols = NUM_BATCH_FRI_PREPROCESSED_COLS;

            // 1. Allocate the matrix directly on the GPU.
            let mut gpu_matrix = GpuMatrix::<F>::new(padded_num_rows, num_cols);

            unsafe {
                crate::sys::process_batch_fri_instructions_gpu(
                    instrs_values.as_ptr(),
                    instrs_values.len(),
                    instrs_index_info.as_ptr(),
                    instrs_index_info.len(),
                    all_base_p_at_x.as_ptr(),
                    all_base_p_at_x.len(),
                    all_ext_p_at_z.as_ptr(),
                    all_ext_p_at_z.len(),
                    all_ext_alpha_pow.as_ptr(),
                    all_ext_alpha_pow.len(),
                    instrs.len(),          //=instrucctions before extending
                    num_total_output_rows, //= instrucctions after extending
                    gpu_matrix.as_mut_ptr() as *mut BabyBear,
                    gpu_matrix.height * gpu_matrix.width,
                    num_cols,
                );
            }

            Some(gpu_matrix)
        } else {
            None
        }

    } */
}

impl<const DEGREE: usize> BatchFRIChip<DEGREE> {
    pub fn eval_batch_fri<AB: SP1RecursionAirBuilder>(
        &self,
        builder: &mut AB,
        local: &BatchFRICols<AB::Var>,
        next: &BatchFRICols<AB::Var>,
        local_prepr: &BatchFRIPreprocessedCols<AB::Var>,
        _next_prepr: &BatchFRIPreprocessedCols<AB::Var>,
    ) {
        // Constrain memory read for alpha_pow, p_at_z, and p_at_x.
        builder.receive_block(local_prepr.alpha_pow_addr, local.alpha_pow, local_prepr.is_real);
        builder.receive_block(local_prepr.p_at_z_addr, local.p_at_z, local_prepr.is_real);
        builder.receive_single(local_prepr.p_at_x_addr, local.p_at_x, local_prepr.is_real);

        // Constrain memory write for the accumulator.
        // Note that we write with multiplicity 1, when `is_end` is true.
        builder.send_block(local_prepr.acc_addr, local.acc, local_prepr.is_end);

        // Constrain the accumulator value of the first row.
        builder.when_first_row().assert_ext_eq(
            local.acc.as_extension::<AB>(),
            local.alpha_pow.as_extension::<AB>()
                * (local.p_at_z.as_extension::<AB>()
                    - BinomialExtension::from_base(local.p_at_x.into())),
        );

        // Constrain the accumulator of the next row when the current row is the end of loop.
        builder.when_transition().when(local_prepr.is_end).assert_ext_eq(
            next.acc.as_extension::<AB>(),
            next.alpha_pow.as_extension::<AB>()
                * (next.p_at_z.as_extension::<AB>()
                    - BinomialExtension::from_base(next.p_at_x.into())),
        );

        // Constrain the accumulator of the next row when the current row is not the end of loop.
        builder.when_transition().when_not(local_prepr.is_end).assert_ext_eq(
            next.acc.as_extension::<AB>(),
            local.acc.as_extension::<AB>()
                + next.alpha_pow.as_extension::<AB>()
                    * (next.p_at_z.as_extension::<AB>()
                        - BinomialExtension::from_base(next.p_at_x.into())),
        );
    }

    pub const fn do_memory_access<T: Copy>(local: &BatchFRIPreprocessedCols<T>) -> T {
        local.is_real
    }
}

impl<AB, const DEGREE: usize> Air<AB> for BatchFRIChip<DEGREE>
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0).unwrap(), main.row_slice(1).unwrap());
        let local: &BatchFRICols<AB::Var> = (*local).borrow();
        let next: &BatchFRICols<AB::Var> = (*next).borrow();
        let prepr = builder.preprocessed();
        let (prepr_local, prepr_next) = (prepr.row_slice(0).unwrap(), prepr.row_slice(1).unwrap());
        let prepr_local: &BatchFRIPreprocessedCols<AB::Var> = (*prepr_local).borrow();
        let prepr_next: &BatchFRIPreprocessedCols<AB::Var> = (*prepr_next).borrow();

        // Dummy constraints to normalize to DEGREE.
        let lhs = (0..DEGREE).map(|_| prepr_local.is_real.into()).product::<AB::Expr>();
        let rhs = (0..DEGREE).map(|_| prepr_local.is_real.into()).product::<AB::Expr>();
        builder.assert_eq(lhs, rhs);

        self.eval_batch_fri::<AB>(builder, local, next, prepr_local, prepr_next);
    }
}

#[cfg(test)]
mod tests {
    use crate::{Instruction, RecursionProgram, chips::test_fixtures};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use sp1_core_machine::utils::pad_rows_fixed;

    use super::*;

    const DEGREE: usize = 2;

    fn generate_trace_reference<const DEGREE: usize>(
        input: &ExecutionRecord<BabyBear>,
        _: &mut ExecutionRecord<BabyBear>,
    ) -> RowMajorMatrix<BabyBear> {
        type F = BabyBear;

        let mut rows = input
            .batch_fri_events
            .iter()
            .map(|event| {
                let mut row = [F::ZERO; NUM_BATCH_FRI_COLS];
                let cols: &mut BatchFRICols<F> = row.as_mut_slice().borrow_mut();
                cols.acc = event.ext_single.acc;
                cols.alpha_pow = event.ext_vec.alpha_pow;
                cols.p_at_z = event.ext_vec.p_at_z;
                cols.p_at_x = event.base_vec.p_at_x;
                row
            })
            .collect_vec();

        rows.resize(BatchFRIChip::<DEGREE>.num_rows(input).unwrap(), [F::ZERO; NUM_BATCH_FRI_COLS]);

        RowMajorMatrix::new(rows.into_iter().flatten().collect(), NUM_BATCH_FRI_COLS)
    }

    //use crate::gpu::init_gpu_context;
    #[test]
    fn generate_trace() {
        //init_gpu_context();
        let shard = test_fixtures::shard();
        let mut execution_record = test_fixtures::default_execution_record();
        println!("batch_fri::tests:generate_trace ");
        let trace = BatchFRIChip::<DEGREE>.generate_trace(&shard, &mut execution_record);
        assert!(trace.height() >= test_fixtures::MIN_TEST_CASES);

        assert_eq!(trace, generate_trace_reference::<DEGREE>(&shard, &mut execution_record));
    }

    fn generate_preprocessed_trace_reference<const DEGREE: usize>(
        program: &RecursionProgram<BabyBear>,
    ) -> RowMajorMatrix<BabyBear> {
        type F = BabyBear;

        let mut rows: Vec<[F; NUM_BATCH_FRI_PREPROCESSED_COLS]> = Vec::new();
        program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::BatchFRI(instr) => Some(instr),
                _ => None,
            })
            .for_each(|instruction| {
                let BatchFRIInstr { base_vec_addrs, ext_single_addrs, ext_vec_addrs, acc_mult } =
                    instruction.as_ref();
                let len = ext_vec_addrs.p_at_z.len();
                let mut row_add = vec![[F::ZERO; NUM_BATCH_FRI_PREPROCESSED_COLS]; len];
                debug_assert_eq!(*acc_mult, F::ONE);

                row_add.iter_mut().enumerate().for_each(|(_i, row)| {
                    let row: &mut BatchFRIPreprocessedCols<F> = row.as_mut_slice().borrow_mut();
                    row.is_real = F::ONE;
                    row.is_end = F::from_bool(_i == len - 1);
                    row.acc_addr = ext_single_addrs.acc;
                    row.alpha_pow_addr = ext_vec_addrs.alpha_pow[_i];
                    row.p_at_z_addr = ext_vec_addrs.p_at_z[_i];
                    row.p_at_x_addr = base_vec_addrs.p_at_x[_i];
                });
                rows.extend(row_add);
            });

        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_BATCH_FRI_PREPROCESSED_COLS],
            program.fixed_log2_rows(&BatchFRIChip::<DEGREE>),
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect(), NUM_BATCH_FRI_PREPROCESSED_COLS)
    }

    #[test]
    #[ignore = "Failing due to merge conflicts. Will be fixed shortly."]
    fn generate_preprocessed_trace() {
        //init_gpu_context();
        let program = test_fixtures::program();
        let trace = BatchFRIChip::<DEGREE>.generate_preprocessed_trace(&program).unwrap();
        assert!(trace.height() >= test_fixtures::MIN_TEST_CASES);

        assert_eq!(trace, generate_preprocessed_trace_reference::<DEGREE>(&program));
    }
}
