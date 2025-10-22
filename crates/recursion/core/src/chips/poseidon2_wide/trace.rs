use crate::{
    ExecutionRecord, Poseidon2Io, Poseidon2SkinnyInstr, instruction::Instruction::Poseidon2,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use sp1_core_machine::{operations::poseidon2::WIDTH, utils::next_power_of_two};
use sp1_stark::air::MachineAir;
use std::{borrow::BorrowMut, mem::size_of};
use tracing::instrument;
//use itertools::Itertools;
use super::{Poseidon2WideChip, columns::preprocessed::Poseidon2PreprocessedColsWide};

//use sp1_stark::GpuMatrix;

pub const PREPROCESSED_POSEIDON2_WIDTH: usize = size_of::<Poseidon2PreprocessedColsWide<u8>>();

impl<F: PrimeField32, const DEGREE: usize> MachineAir<F> for Poseidon2WideChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        format!("Poseidon2WideDeg{}", DEGREE)
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.poseidon2_events;
        match input.fixed_log2_rows(self) {
            Some(log2_rows) => Some(1 << log2_rows),
            None => Some(next_power_of_two(events.len(), None)),
        }
    }

    #[instrument(name = "generate poseidon2 wide trace", level = "debug", skip_all, fields(rows = input.poseidon2_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _output: &mut ExecutionRecord<F>,
    ) -> RowMajorMatrix<F> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_trace only supports BabyBear field"
        );

        let events = unsafe {
            std::mem::transmute::<&Vec<Poseidon2Io<F>>, &Vec<Poseidon2Io<BabyBear>>>(
                &input.poseidon2_events,
            )
        };
        let padded_nb_rows = self.num_rows(input).unwrap();
        let num_columns = <Self as BaseAir<F>>::width(self);
        let mut values = vec![BabyBear::ZERO; padded_nb_rows * num_columns];

        let populate_len = input.poseidon2_events.len() * num_columns;
        let (values_pop, values_dummy) = values.split_at_mut(populate_len);

        let populate_perm_ffi = |input: &[BabyBear; WIDTH], input_row: &mut [BabyBear]| unsafe {
            crate::sys::poseidon2_wide_event_to_row_babybear(
                input.as_ptr(),
                input_row.as_mut_ptr(),
                DEGREE == 3,
            )
        };

        join(
            || {
                values_pop
                    .par_chunks_mut(num_columns)
                    //.chunks_mut(num_columns) //test
                    .zip_eq(events)
                    //.zip(events)
                    .for_each(|(row, event)| populate_perm_ffi(&event.input, row))
            },
            || {
                let mut dummy_row = vec![BabyBear::ZERO; num_columns];
                populate_perm_ffi(&[BabyBear::ZERO; WIDTH], &mut dummy_row);
                values_dummy
                    .par_chunks_mut(num_columns)
                    .for_each(|row| row.copy_from_slice(&dummy_row))
            },
        );

        let trace = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            num_columns,
        );
        trace
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }

    fn preprocessed_width(&self) -> usize {
        PREPROCESSED_POSEIDON2_WIDTH
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        Some(match program.fixed_log2_rows(self) {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(instrs_len, None),
        })
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_preprocessed_trace only supports BabyBear field"
        );
        // Allocating an intermediate `Vec` is faster.
        let instrs: Vec<&Poseidon2SkinnyInstr<BabyBear>> =
            program
                .inner
                .iter() // Faster than using `rayon` for some reason. Maybe vectorization?
                .filter_map(|instruction| match instruction {
                    Poseidon2(instr) => Some(unsafe {
                        std::mem::transmute::<
                            &Poseidon2SkinnyInstr<F>,
                            &Poseidon2SkinnyInstr<BabyBear>,
                        >(instr.as_ref())
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut values = vec![BabyBear::ZERO; padded_nb_rows * PREPROCESSED_POSEIDON2_WIDTH];

        if instrs.is_empty() {
            return Some(RowMajorMatrix::new(
                unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
                PREPROCESSED_POSEIDON2_WIDTH,
            ));
        }

        let populate_len = instrs.len() * PREPROCESSED_POSEIDON2_WIDTH;
        values[..populate_len]
            .par_chunks_mut(PREPROCESSED_POSEIDON2_WIDTH)
            .zip_eq(instrs)
            .for_each(|(row, instr)| {
                let cols: &mut Poseidon2PreprocessedColsWide<_> = row.borrow_mut();
                unsafe {
                    crate::sys::poseidon2_wide_instr_to_row_babybear(instr, cols);
                }
            });

        let trace = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            PREPROCESSED_POSEIDON2_WIDTH,
        );

        Some(trace)
    }

    /*
    fn generate_trace_gpu(&self, input: &Self::Record, _: &mut Self::Record) -> GpuMatrix<F> {
        let events = unsafe {
            std::mem::transmute::<&Vec<Poseidon2Io<F>>, &Vec<Poseidon2Io<BabyBear>>>(
                &input.poseidon2_events,
            )
        };

        // Pad the trace
        let padded_nb_rows = self.num_rows(input).unwrap();
        let num_cols = <Self as BaseAir<F>>::width(self);

        // 1. Allocate the matrix directly on the GPU.
        let mut gpu_matrix = GpuMatrix::<F>::new(padded_nb_rows, num_cols);

        if !events.is_empty() {
            unsafe {
                crate::sys::process_poseidon2_wide_events_gpu(
                    events.as_ptr(),
                    events.len(),
                    gpu_matrix.as_mut_ptr() as *mut BabyBear, // Pass the device pointer
                    gpu_matrix.height * gpu_matrix.width, // Pass total elements
                    padded_nb_rows,
                    num_cols,
                    DEGREE == 3,
                    WIDTH,
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
        let instrs: Vec<&Poseidon2SkinnyInstr<BabyBear>> =
            program
                .inner
                .iter() // Faster than using `rayon` for some reason. Maybe vectorization?
                .filter_map(|instruction| match instruction {
                    Poseidon2(instr) => Some(unsafe {
                        std::mem::transmute::<
                            &Poseidon2SkinnyInstr<F>,
                            &Poseidon2SkinnyInstr<BabyBear>,
                        >(instr.as_ref())
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();

        if !instrs.is_empty()  {
            let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
            let num_cols = PREPROCESSED_POSEIDON2_WIDTH;

            // 1. Allocate the matrix directly on the GPU.
            let mut gpu_matrix = GpuMatrix::<F>::new(padded_nb_rows, num_cols);

            let instrs_for_gpu: Vec<Poseidon2SkinnyInstr<BabyBear>> = instrs
                .iter()
                .map(|&instr_ref| *instr_ref)
                .collect_vec();
            unsafe {
                crate::sys::process_poseidon2_wide_instructions_gpu(
                    instrs_for_gpu.as_ptr(),
                    instrs_for_gpu.len(),
                    gpu_matrix.as_mut_ptr() as *mut BabyBear, // Pass the device pointer
                    gpu_matrix.height * gpu_matrix.width, // Pass total elements
                    PREPROCESSED_POSEIDON2_WIDTH,
                );
            }
            Some(gpu_matrix)
        } else {
            None
        }
    } */
}

#[cfg(test)]
mod tests {
    use crate::{
        ExecutionRecord, RecursionProgram,
        chips::{mem::MemoryAccessCols, poseidon2_wide::Poseidon2WideChip, test_fixtures},
    };
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::{Matrix, dense::RowMajorMatrix};
    use sp1_core_machine::operations::poseidon2::{WIDTH, trace::populate_perm};
    use sp1_stark::air::MachineAir;

    use super::*;

    const DEGREE_3: usize = 3;
    const DEGREE_9: usize = 9;

    fn generate_trace_reference<const DEGREE: usize>(
        input: &ExecutionRecord<BabyBear>,
        _: &mut ExecutionRecord<BabyBear>,
    ) -> RowMajorMatrix<BabyBear> {
        type F = BabyBear;

        let events = &input.poseidon2_events;
        let chip = Poseidon2WideChip::<DEGREE>;
        let padded_nb_rows = chip.num_rows(input).unwrap();
        let num_columns = <Poseidon2WideChip<DEGREE> as BaseAir<F>>::width(&chip);
        let mut values = vec![F::ZERO; padded_nb_rows * num_columns];

        let populate_len = events.len() * num_columns;
        let (values_pop, values_dummy) = values.split_at_mut(populate_len);
        join(
            || {
                values_pop.par_chunks_mut(num_columns).zip_eq(&input.poseidon2_events).for_each(
                    |(row, &event)| {
                        populate_perm::<F, DEGREE>(event.input, Some(event.output), row);
                    },
                )
            },
            || {
                let mut dummy_row = vec![F::ZERO; num_columns];
                populate_perm::<F, DEGREE>([F::ZERO; WIDTH], None, &mut dummy_row);
                values_dummy
                    .par_chunks_mut(num_columns)
                    .for_each(|row| row.copy_from_slice(&dummy_row))
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, num_columns)
    }

    use p3_symmetric::Permutation;
    use rand::{Rng, SeedableRng, prelude::SliceRandom, rngs::StdRng};
    use sp1_stark::inner_perm;
    use std::array;
    #[test]
    fn test_cpp_poseidon2() {
        let mut rng = StdRng::seed_from_u64(12345);
        let input = array::from_fn(|_| BabyBear::from_u32(rng.r#gen()));
        println!("---###########  permute_input:{:?}", input);
        let permuter = inner_perm();
        let output = permuter.permute(input.clone());

        let populate_perm_ffi = |input: &[BabyBear; WIDTH], input_row: &mut [BabyBear]| unsafe {
            crate::sys::poseidon2_wide_event_to_row_babybear(
                input.as_ptr(),
                input_row.as_mut_ptr(),
                3 == 3,
            )
        };

        let mut values = vec![BabyBear::ZERO; 320];
        populate_perm_ffi(&input, &mut values);

        println!("c++  out:{:?}", values);
        println!("rust out:{:?}", output);

        //assert_eq!(values, output);
    }

    //use crate::gpu::init_gpu_context;
    #[test]
    fn test_generate_trace_deg_3() {
        // init_gpu_context();
        let shard = test_fixtures::shard();
        let mut execution_record = test_fixtures::default_execution_record();
        let chip = Poseidon2WideChip::<DEGREE_3>;

        //recursion trace is generated by c++ code!
        let trace = chip.generate_trace(&shard, &mut execution_record);
        assert!(trace.height() >= test_fixtures::MIN_TEST_CASES);

        assert_eq!(trace, generate_trace_reference::<DEGREE_3>(&shard, &mut execution_record));
    }

    #[test]
    fn generate_trace_gpu() {
        let shard = test_fixtures::shard();
        let mut execution_record = test_fixtures::default_execution_record();
        let chip = Poseidon2WideChip::<DEGREE_3>;

        let trace_gpu_ptr = chip.generate_trace_gpu(&shard, &mut execution_record);

        let trace_values = trace_gpu_ptr.to_host();
        let num_columns = <Poseidon2WideChip<DEGREE_3> as BaseAir<BabyBear>>::width(&chip);
        let trace = RowMajorMatrix::new(trace_values, num_columns);

        assert_eq!(trace, generate_trace_reference::<DEGREE_3>(&shard, &mut execution_record));
    }

    #[test]
    fn test_generate_trace_deg_9() {
        //init_gpu_context();
        let shard = test_fixtures::shard();
        let mut execution_record = test_fixtures::default_execution_record();
        let chip = Poseidon2WideChip::<DEGREE_9>;
        let trace = chip.generate_trace(&shard, &mut execution_record);
        assert!(trace.height() >= test_fixtures::MIN_TEST_CASES);

        assert_eq!(trace, generate_trace_reference::<DEGREE_9>(&shard, &mut execution_record));
    }

    fn generate_preprocessed_trace_ffi<const DEGREE: usize>(
        program: &RecursionProgram<BabyBear>,
    ) -> RowMajorMatrix<BabyBear> {
        type F = BabyBear;

        let instrs = program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Poseidon2(instr) => Some(instr.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let padded_nb_rows = Poseidon2WideChip::<DEGREE>::preprocessed_num_rows(
            &Poseidon2WideChip::<DEGREE>,
            program,
            instrs.len(),
        )
        .unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * PREPROCESSED_POSEIDON2_WIDTH];

        let populate_len = instrs.len() * PREPROCESSED_POSEIDON2_WIDTH;
        values[..populate_len]
            .par_chunks_mut(PREPROCESSED_POSEIDON2_WIDTH)
            .zip_eq(instrs)
            .for_each(|(row, instr)| {
                // Set the memory columns. We read once, at the first iteration,
                // and write once, at the last iteration.
                *row.borrow_mut() = Poseidon2PreprocessedColsWide {
                    input: instr.addrs.input,
                    output: std::array::from_fn(|j| MemoryAccessCols {
                        addr: instr.addrs.output[j],
                        mult: instr.mults[j],
                    }),
                    is_real_neg: F::NEG_ONE,
                }
            });

        RowMajorMatrix::new(values, PREPROCESSED_POSEIDON2_WIDTH)
    }

    #[test]
    #[ignore = "Failing due to merge conflicts. Will be fixed shortly."]
    fn test_generate_preprocessed_trace_deg_3() {
        //init_gpu_context();
        let program = test_fixtures::program();
        let chip = Poseidon2WideChip::<DEGREE_3>;
        let trace = chip.generate_preprocessed_trace(&program).unwrap();
        assert!(trace.height() >= test_fixtures::MIN_TEST_CASES);

        assert_eq!(trace, generate_preprocessed_trace_ffi::<DEGREE_3>(&program));
    }

    #[test]
    #[ignore = "Failing due to merge conflicts. Will be fixed shortly."]
    fn test_generate_preprocessed_trace_deg_9() {
        let program = test_fixtures::program();
        let chip = Poseidon2WideChip::<DEGREE_9>;
        let trace = chip.generate_preprocessed_trace(&program).unwrap();
        assert!(trace.height() >= test_fixtures::MIN_TEST_CASES);

        assert_eq!(trace, generate_preprocessed_trace_ffi::<DEGREE_9>(&program));
    }
}
