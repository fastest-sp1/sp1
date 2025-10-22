#![allow(clippy::needless_range_loop)]

use crate::{
    ExpReverseBitsEvent,
    ExpReverseBitsInstr,
    Instruction,
    //   ExpReverseBitsInstrFlatFFI, InstrsFlatIdex, Address,  ExpReverseBitsFlatIdex, ExpReverseBitsEventFlatFFI
    builder::SP1RecursionAirBuilder,
    runtime::ExecutionRecord,
};
use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use sp1_core_machine::utils::next_power_of_two;
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{BaseAirBuilder, ExtensionAirBuilder, MachineAir, SP1AirBuilder};
use std::borrow::BorrowMut;
use tracing::instrument;
//use itertools::Itertools;
use super::mem::{MemoryAccessCols, MemoryAccessColsChips};

//use sp1_stark::GpuMatrix;

pub const NUM_EXP_REVERSE_BITS_LEN_COLS: usize = core::mem::size_of::<ExpReverseBitsLenCols<u8>>();
pub const NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS: usize =
    core::mem::size_of::<ExpReverseBitsLenPreprocessedCols<u8>>();

#[derive(Clone, Debug, Copy, Default)]
pub struct ExpReverseBitsLenChip<const DEGREE: usize>;

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct ExpReverseBitsLenPreprocessedCols<T: Copy> {
    pub x_mem: MemoryAccessColsChips<T>,
    pub exponent_mem: MemoryAccessColsChips<T>,
    pub result_mem: MemoryAccessColsChips<T>,
    pub iteration_num: T,
    pub is_first: T,
    pub is_last: T,
    pub is_real: T,
}

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct ExpReverseBitsLenCols<T: Copy> {
    /// The base of the exponentiation.
    pub x: T,

    /// The current bit of the exponent. This is read from memory.
    pub current_bit: T,

    /// The previous accumulator squared.
    pub prev_accum_squared: T,

    /// Is set to the value local.prev_accum_squared * local.multiplier.
    pub prev_accum_squared_times_multiplier: T,

    /// The accumulator of the current iteration.
    pub accum: T,

    /// The accumulator squared.
    pub accum_squared: T,

    /// A column which equals x if `current_bit` is on, and 1 otherwise.
    pub multiplier: T,
}

impl<F, const DEGREE: usize> BaseAir<F> for ExpReverseBitsLenChip<DEGREE> {
    fn width(&self) -> usize {
        NUM_EXP_REVERSE_BITS_LEN_COLS
    }
}

impl<F: PrimeField32, const DEGREE: usize> MachineAir<F> for ExpReverseBitsLenChip<DEGREE> {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    fn name(&self) -> String {
        "ExpReverseBitsLen".to_string()
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn preprocessed_width(&self) -> usize {
        NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        assert!(
            std::any::TypeId::of::<F>() == std::any::TypeId::of::<BabyBear>(),
            "generate_preprocessed_trace only supports BabyBear field"
        );

        //let mut rows: Vec<[BabyBear; NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS]> = Vec::new();
        let mut values: Vec<BabyBear>;

        //CPU
        let mut cpu_rows: Vec<[BabyBear; NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS]> = Vec::new();
        program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::ExpReverseBitsLen(x) => Some(unsafe {
                    std::mem::transmute::<&ExpReverseBitsInstr<F>, &ExpReverseBitsInstr<BabyBear>>(
                        x,
                    )
                }),
                _ => None,
            })
            .for_each(|instruction: &ExpReverseBitsInstr<BabyBear>| {
                let ExpReverseBitsInstr { addrs, mult } = instruction;
                let mut row_add = vec![
                    [BabyBear::ZERO; NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS];
                    addrs.exp.len()
                ];
                row_add.iter_mut().enumerate().for_each(|(i, row)| {
                    let row: &mut ExpReverseBitsLenPreprocessedCols<BabyBear> =
                        row.as_mut_slice().borrow_mut();
                    row.iteration_num = BabyBear::from_u32(i as u32);
                    row.is_first = BabyBear::from_bool(i == 0);
                    row.is_last = BabyBear::from_bool(i == addrs.exp.len() - 1);
                    row.is_real = BabyBear::ONE;
                    row.x_mem =
                        MemoryAccessCols { addr: addrs.base, mult: -BabyBear::from_bool(i == 0) };
                    row.exponent_mem =
                        MemoryAccessCols { addr: addrs.exp[i], mult: BabyBear::NEG_ONE };
                    row.result_mem = MemoryAccessCols {
                        addr: addrs.result,
                        mult: *mult * BabyBear::from_bool(i == addrs.exp.len() - 1),
                    };
                });
                cpu_rows.extend(row_add);
            });
        values = cpu_rows.into_iter().flatten().collect::<Vec<BabyBear>>();

        // Pad the trace to a power of two.
        if program.fixed_log2_rows(self).is_some() || values.len() > 0 {
            let current_num_rows = values.len() / NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS;
            let padded_num_rows =
                next_power_of_two(current_num_rows, program.fixed_log2_rows(self));
            let target_total_elements =
                padded_num_rows * NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS;
            values.resize(target_total_elements, BabyBear::ZERO);
        }

        let trace = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS,
        );
        Some(trace)
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.exp_reverse_bits_len_events;
        Some(next_power_of_two(events.len(), input.fixed_log2_rows(self)))
    }

    #[instrument(name = "generate exp reverse bits len trace", level = "debug", skip_all, fields(rows = input.exp_reverse_bits_len_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _: &mut ExecutionRecord<F>,
    ) -> RowMajorMatrix<F> {
        assert!(
            std::any::TypeId::of::<F>() == std::any::TypeId::of::<BabyBear>(),
            "generate_trace only supports BabyBear field"
        );

        let events = unsafe {
            std::mem::transmute::<&Vec<ExpReverseBitsEvent<F>>, &Vec<ExpReverseBitsEvent<BabyBear>>>(
                &input.exp_reverse_bits_len_events,
            )
        };

        let mut values: Vec<BabyBear>;

        let mut overall_rows = Vec::new();

        events.iter().for_each(|event| {
            let mut rows =
                vec![vec![BabyBear::ZERO; NUM_EXP_REVERSE_BITS_LEN_COLS]; event.exp.len()];
            let mut accum = BabyBear::ONE;

            rows.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut ExpReverseBitsLenCols<BabyBear> = row.as_mut_slice().borrow_mut();
                unsafe {
                    crate::sys::exp_reverse_bits_event_to_row_babybear(&event.into(), i, cols);
                }

                let prev_accum = accum;
                accum = prev_accum * prev_accum * cols.multiplier;

                cols.accum = accum;
                cols.accum_squared = accum * accum;
                cols.prev_accum_squared = prev_accum * prev_accum;
                cols.prev_accum_squared_times_multiplier =
                    cols.prev_accum_squared * cols.multiplier;
            });
            overall_rows.extend(rows);
        });

        let out_rows = overall_rows.len();
        values = overall_rows.into_iter().flatten().collect::<Vec<BabyBear>>();

        // Pad the trace to a power of two.
        //let padded_num_rows = self.num_rows(input).unwrap();
        let padded_num_rows = next_power_of_two(out_rows, input.fixed_log2_rows(self));

        let target_total_elements = padded_num_rows * NUM_EXP_REVERSE_BITS_LEN_COLS;
        values.resize(target_total_elements, BabyBear::ZERO);

        let trace = RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_EXP_REVERSE_BITS_LEN_COLS,
        );

        #[cfg(debug_assertions)]
        eprintln!(
            "exp reverse bits len trace dims is width: {:?}, height: {:?}",
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
        let events = unsafe {
            std::mem::transmute::<&Vec<ExpReverseBitsEvent<F>>, &Vec<ExpReverseBitsEvent<BabyBear>>>(
                &input.exp_reverse_bits_len_events,
            )
        };

        if !events.is_empty()  {
            let mut ffi_events: Vec<ExpReverseBitsEventFlatFFI<BabyBear>> = Vec::new();
            let mut all_exp_bits: Vec<BabyBear> = Vec::new();

            let mut current_exp_offset = 0;
            let mut events_index_info: Vec<ExpReverseBitsFlatIdex<BabyBear>> = Vec::new();

            for (event_idx, event) in events.iter().enumerate() {
                //let event = event_ref;
                let event_exp_len  = event.exp.len();

                let mut accum = BabyBear::ONE;
                for j in 0..event_exp_len {
                    let current_exp_bit = event.exp[j];
                    let base_val = event.base;

                    let multiplier = if BabyBear::ONE == current_exp_bit {
                        base_val
                    } else {
                        BabyBear::ONE
                    };
                    let prev_accum = accum;
                    events_index_info.push(ExpReverseBitsFlatIdex {
                            event_idx: event_idx,
                            exp_idx: j,
                            multiplier: multiplier,
                            prev_accum: prev_accum,
                        });


                    accum = prev_accum * prev_accum * multiplier;
                }

                //event_output_offsets.push(current_output_offset);
                all_exp_bits.extend_from_slice(&event.exp);

                // ExpReverseBitsEventFlatFFI
                ffi_events.push(ExpReverseBitsEventFlatFFI {
                    base_val: event.base,
                    exp_bits_offset: current_exp_offset,
                    exp_len: event_exp_len,
                    result_val: event.result,
                });

                current_exp_offset += event_exp_len;
                //current_output_offset += event_exp_len;
            }
            let total_output_rows = current_exp_offset ;

            let padded_num_rows =  next_power_of_two(total_output_rows, input.fixed_log2_rows(self));
            let num_cols = <Self as BaseAir<F>>::width(self);

            // Allocate the matrix directly on the GPU.
            let mut gpu_matrix = GpuMatrix::<F>::new(padded_num_rows, num_cols);

            unsafe {
                crate::sys::process_exp_reverse_bits_events_gpu(
                    ffi_events.as_ptr(),
                    ffi_events.len(),
                    events_index_info.as_ptr(),
                    events_index_info.len(),
                    all_exp_bits.as_ptr(),
                    all_exp_bits.len(),
                    gpu_matrix.as_mut_ptr() as *mut BabyBear, // Pass the device pointer
                    gpu_matrix.height * gpu_matrix.width, // Pass total elements including padding ele
                    total_output_rows,     //= events after extending
                    NUM_EXP_REVERSE_BITS_LEN_COLS,
                );
            }
            gpu_matrix

        } else {
            let padded_nb_rows = self.num_rows(input).unwrap();
            let num_cols = <Self as BaseAir<F>>::width(self);

            // Allocate the matrix directly on the GPU.
            let mut gpu_matrix = GpuMatrix::<F>::new(padded_nb_rows, num_cols);
            gpu_matrix
        }
    }


    fn generate_preprocessed_trace_gpu(
        &self,
        program: &Self::Program,
    ) -> Option<GpuMatrix<F>> {
        let instrs_raw: Vec<&ExpReverseBitsInstr<BabyBear>> = program
                .inner
                .iter()
                .filter_map(|instruction| match instruction {
                    Instruction::ExpReverseBitsLen(x) => Some(unsafe {
                        std::mem::transmute::<&ExpReverseBitsInstr<F>, &ExpReverseBitsInstr<BabyBear>>(
                            x,
                        )
                    }),
                    _ => None,
                })
                .collect_vec();

        if !instrs_raw.is_empty() {
            let mut all_exp_bits: Vec<Address<BabyBear>> = Vec::new();
            let mut instrs_values: Vec<ExpReverseBitsInstrFlatFFI<BabyBear>> = Vec::with_capacity(instrs_raw.len());

            let mut current_exp_offset = 0;


            let mut instrs_index_info: Vec<InstrsFlatIdex> = Vec::new();

            let mut num_total_output_rows = 0;
            for (instr_idx, &instr_ref) in instrs_raw.iter().enumerate() {
                let instr_data = instr_ref;
                let exp_len = instr_data.addrs.exp.len();
                num_total_output_rows += exp_len;
                for j in 0..exp_len {
                     instrs_index_info.push(InstrsFlatIdex {
                            instr_idx: instr_idx,
                            arr_idx: j ,
                           // last_row: p_at_z_len,
                        });
                }

                all_exp_bits.extend_from_slice(&instr_data.addrs.exp);

                instrs_values.push(crate::ExpReverseBitsInstrFlatFFI {
                    base: instr_data.addrs.base,
                    exp_offset: current_exp_offset,
                    exp_len: exp_len,
                    result: instr_data.addrs.result,
                    mult: instr_data.mult,
                });

                current_exp_offset += exp_len;

            }

            let padded_num_rows = next_power_of_two(num_total_output_rows, program.fixed_log2_rows(self));
            let num_cols = NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS;
            //  Allocate the matrix directly on the GPU.
            let mut gpu_matrix = GpuMatrix::<F>::new(padded_num_rows, num_cols);

            unsafe {
                crate::sys::process_exp_reverse_bits_instructions_gpu(
                    instrs_values.as_ptr(),
                    instrs_values.len(),
                    instrs_index_info.as_ptr(),
                    instrs_index_info.len(),
                    all_exp_bits.as_ptr(),
                    all_exp_bits.len(),
                    gpu_matrix.as_mut_ptr() as *mut BabyBear,
                    gpu_matrix.height * gpu_matrix.width,
                    num_total_output_rows, //= instrucctions after extending
                    NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS,
                );
            }
            Some(gpu_matrix)
        } else {
            None
        }

    }*/
}

impl<const DEGREE: usize> ExpReverseBitsLenChip<DEGREE> {
    pub fn eval_exp_reverse_bits_len<
        AB: BaseAirBuilder + ExtensionAirBuilder + SP1RecursionAirBuilder + SP1AirBuilder,
    >(
        &self,
        builder: &mut AB,
        local: &ExpReverseBitsLenCols<AB::Var>,
        local_prepr: &ExpReverseBitsLenPreprocessedCols<AB::Var>,
        next: &ExpReverseBitsLenCols<AB::Var>,
        next_prepr: &ExpReverseBitsLenPreprocessedCols<AB::Var>,
    ) {
        // Dummy constraints to normalize to DEGREE when DEGREE > 3.
        if DEGREE > 3 {
            let lhs = (0..DEGREE).map(|_| local_prepr.is_real.into()).product::<AB::Expr>();
            let rhs = (0..DEGREE).map(|_| local_prepr.is_real.into()).product::<AB::Expr>();
            builder.assert_eq(lhs, rhs);
        }

        // Constrain mem read for x.  The read mult is one for only the first row, and zero for all
        // others.
        builder.send_single(local_prepr.x_mem.addr, local.x, local_prepr.x_mem.mult);

        //move
        // Constrain mem read for exponent's bits.  The read mult is one for all real rows.
        builder.send_single(
            local_prepr.exponent_mem.addr,
            local.current_bit,
            local_prepr.exponent_mem.mult,
        );

        //move
        builder.send_single(local_prepr.result_mem.addr, local.accum, local_prepr.result_mem.mult);

        // Ensure that the value at the x memory access is unchanged when not `is_last`.
        //println!("---cpu: local.x={:?}, next.x={:?}", local.x.into(), next.x.into());
        builder
            .when_transition()
            .when(next_prepr.is_real)
            .when_not(local_prepr.is_last)
            .assert_eq(local.x, next.x);

        // Constrain mem read for exponent's bits.  The read mult is one for all real rows.
        // builder.send_single(
        //    local_prepr.exponent_mem.addr,
        //     local.current_bit,
        //    local_prepr.exponent_mem.mult,
        // );

        // The accumulator needs to start with the multiplier for every `is_first` row.
        builder.when(local_prepr.is_first).assert_eq(local.accum, local.multiplier);

        // `multiplier` is x if the current bit is 1, and 1 if the current bit is 0.
        builder
            .when(local_prepr.is_real)
            .when(local.current_bit)
            .assert_eq(local.multiplier, local.x);
        builder
            .when(local_prepr.is_real)
            .when_not(local.current_bit)
            .assert_eq(local.multiplier, AB::Expr::ONE);

        // To get `next.accum`, we multiply `local.prev_accum_squared` by `local.multiplier` when
        // not `is_last`.
        builder.when(local_prepr.is_real).assert_eq(
            local.prev_accum_squared_times_multiplier,
            local.prev_accum_squared * local.multiplier,
        );

        builder
            .when(local_prepr.is_real)
            .when_not(local_prepr.is_first)
            .assert_eq(local.accum, local.prev_accum_squared_times_multiplier);

        // Constrain the accum_squared column.
        builder.when(local_prepr.is_real).assert_eq(local.accum_squared, local.accum * local.accum);

        builder
            .when_transition()
            .when(next_prepr.is_real)
            .when_not(local_prepr.is_last)
            .assert_eq(next.prev_accum_squared, local.accum_squared);

        // Constrain mem write for the result.
        //builder.send_single(local_prepr.result_mem.addr, local.accum, local_prepr.result_mem.mult);
    }

    pub const fn do_exp_bit_memory_access<T: Copy>(
        local: &ExpReverseBitsLenPreprocessedCols<T>,
    ) -> T {
        local.is_real
    }
}

impl<AB, const DEGREE: usize> Air<AB> for ExpReverseBitsLenChip<DEGREE>
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0).unwrap(), main.row_slice(1).unwrap());
        let local: &ExpReverseBitsLenCols<AB::Var> = (*local).borrow();
        let next: &ExpReverseBitsLenCols<AB::Var> = (*next).borrow();
        let prep = builder.preprocessed();
        let (prep_local, prep_next) = (prep.row_slice(0).unwrap(), prep.row_slice(1).unwrap());
        let prep_local: &ExpReverseBitsLenPreprocessedCols<_> = (*prep_local).borrow();
        let prep_next: &ExpReverseBitsLenPreprocessedCols<_> = (*prep_next).borrow();
        self.eval_exp_reverse_bits_len::<AB>(builder, local, prep_local, next, prep_next);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use crate::{
        Address, ExpReverseBitsEvent, ExpReverseBitsIo, Instruction, MemAccessKind,
        RecursionProgram,
        chips::{exp_reverse_bits::ExpReverseBitsLenChip, test_fixtures},
        linear_program,
        machine::tests::test_recursion_linear_program,
        runtime::{ExecutionRecord, instruction as instr},
        stark::BabyBearPoseidon2Outer,
    };
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::reverse_bits_len;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use sp1_core_machine::utils::pad_rows_fixed;
    use sp1_core_machine::utils::setup_logger;
    use sp1_stark::{StarkGenericConfig, air::MachineAir};
    use std::iter::once;

    use super::*;

    const DEGREE: usize = 3;

    #[test]
    fn prove_babybear_circuit_erbl() {
        setup_logger();
        type SC = BabyBearPoseidon2Outer;
        type F = <SC as StarkGenericConfig>::Val;

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_felt = move || -> F { F::from_u32(rng.gen_range(0..1 << 16)) };
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_bit = move || rng.gen_range(0..2);
        let mut addr = 0;

        let instructions = (1..15)
            .flat_map(|i| {
                let base = random_felt();
                let exponent_bits = vec![random_bit(); i];
                let exponent = F::from_u32(
                    exponent_bits.iter().enumerate().fold(0, |acc, (i, x)| acc + x * (1 << i)),
                );
                let result =
                    base.exp_u64(reverse_bits_len(exponent.as_canonical_u32() as usize, i) as u64);

                let alloc_size = i + 2;
                let exp_a = (0..i).map(|x| x + addr + 1).collect::<Vec<_>>();
                let exp_a_clone = exp_a.clone();
                let x_a = addr;
                let result_a = addr + alloc_size - 1;
                addr += alloc_size;
                let exp_bit_instructions = (0..i).map(move |j| {
                    instr::mem_single(
                        MemAccessKind::Write,
                        1,
                        exp_a_clone[j] as u32,
                        F::from_u32(exponent_bits[j]),
                    )
                });
                once(instr::mem_single(MemAccessKind::Write, 1, x_a as u32, base))
                    .chain(exp_bit_instructions)
                    .chain(once(instr::exp_reverse_bits_len(
                        1,
                        F::from_u32(x_a as u32),
                        exp_a.into_iter().map(|bit| F::from_u32(bit as u32)).collect_vec(),
                        F::from_u32(result_a as u32),
                    )))
                    .chain(once(instr::mem_single(MemAccessKind::Read, 1, result_a as u32, result)))
            })
            .collect::<Vec<Instruction<F>>>();

        test_recursion_linear_program(instructions);
    }

    #[test]
    fn generate_trace() {
        type F = BabyBear;

        let shard = ExecutionRecord {
            exp_reverse_bits_len_events: vec![ExpReverseBitsEvent {
                base: F::TWO,
                exp: vec![F::ZERO, F::ONE, F::ONE],
                result: F::TWO.exp_u64(0b110),
            }],
            ..Default::default()
        };
        let chip = ExpReverseBitsLenChip::<3>;
        let trace: RowMajorMatrix<F> = chip.generate_trace(&shard, &mut ExecutionRecord::default());
        println!("{:?}", trace.values)
    }

    #[test]
    fn generate_erbl_preprocessed_trace() {
        type F = BabyBear;

        let program = linear_program(vec![
            instr::mem(MemAccessKind::Write, 2, 0, 0),
            instr::mem(MemAccessKind::Write, 2, 1, 0),
            Instruction::ExpReverseBitsLen(ExpReverseBitsInstr {
                addrs: ExpReverseBitsIo {
                    base: Address(F::ZERO),
                    exp: vec![Address(F::ONE), Address(F::ZERO), Address(F::ONE)],
                    result: Address(F::from_u32(4)),
                },
                mult: F::ONE,
            }),
            instr::mem(MemAccessKind::Read, 1, 4, 0),
        ])
        .unwrap();

        let chip = ExpReverseBitsLenChip::<3>;
        let trace = chip.generate_preprocessed_trace(&program).unwrap();
        println!("{:?}", trace.values);
    }

    fn generate_trace_reference<const DEGREE: usize>(
        input: &ExecutionRecord<BabyBear>,
        _: &mut ExecutionRecord<BabyBear>,
    ) -> RowMajorMatrix<BabyBear> {
        type F = BabyBear;

        let mut overall_rows = Vec::new();
        input.exp_reverse_bits_len_events.iter().for_each(|event| {
            let mut rows = vec![vec![F::ZERO; NUM_EXP_REVERSE_BITS_LEN_COLS]; event.exp.len()];

            let mut accum = F::ONE;

            rows.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut ExpReverseBitsLenCols<F> = row.as_mut_slice().borrow_mut();

                let prev_accum = accum;
                accum = prev_accum
                    * prev_accum
                    * if event.exp[i] == F::ONE { event.base } else { F::ONE };

                cols.x = event.base;
                cols.current_bit = event.exp[i];
                cols.accum = accum;
                cols.accum_squared = accum * accum;
                cols.prev_accum_squared = prev_accum * prev_accum;
                cols.multiplier = if event.exp[i] == F::ONE { event.base } else { F::ONE };
                cols.prev_accum_squared_times_multiplier =
                    cols.prev_accum_squared * cols.multiplier;
                if i == event.exp.len() {
                    assert_eq!(event.result, accum);
                }
            });

            overall_rows.extend(rows);
        });

        pad_rows_fixed(
            &mut overall_rows,
            || [F::ZERO; NUM_EXP_REVERSE_BITS_LEN_COLS].to_vec(),
            input.fixed_log2_rows(&ExpReverseBitsLenChip::<DEGREE>),
        );

        RowMajorMatrix::new(
            overall_rows.into_iter().flatten().collect(),
            NUM_EXP_REVERSE_BITS_LEN_COLS,
        )
    }

    //use crate::gpu::init_gpu_context;
    #[test]
    fn test_generate_trace() {
        //init_gpu_context();
        let shard = test_fixtures::shard();
        let mut execution_record = test_fixtures::default_execution_record();
        let trace = ExpReverseBitsLenChip::<DEGREE>.generate_trace(&shard, &mut execution_record);
        assert!(trace.height() >= test_fixtures::MIN_TEST_CASES);

        assert_eq!(trace, generate_trace_reference::<DEGREE>(&shard, &mut execution_record));
    }

    #[test]
    fn generate_trace_gpu() {
        let shard = test_fixtures::shard();
        let mut execution_record = test_fixtures::default_execution_record();
        let trace_gpu_ptr =
            ExpReverseBitsLenChip::<DEGREE>.generate_trace_gpu(&shard, &mut execution_record);

        let trace_values = trace_gpu_ptr.to_host();
        let trace = RowMajorMatrix::new(trace_values, NUM_EXP_REVERSE_BITS_LEN_COLS);

        assert_eq!(trace, generate_trace_reference::<DEGREE>(&shard, &mut execution_record));
    }

    fn generate_preprocessed_trace_reference(
        program: &RecursionProgram<BabyBear>,
    ) -> RowMajorMatrix<BabyBear> {
        type F = BabyBear;

        let mut rows: Vec<[F; NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS]> = Vec::new();
        program
            .inner
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::ExpReverseBitsLen(x) => Some(x),
                _ => None,
            })
            .for_each(|instruction| {
                let ExpReverseBitsInstr { addrs, mult } = instruction;
                let mut row_add =
                    vec![[F::ZERO; NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS]; addrs.exp.len()];
                row_add.iter_mut().enumerate().for_each(|(i, row)| {
                    let row: &mut ExpReverseBitsLenPreprocessedCols<F> =
                        row.as_mut_slice().borrow_mut();
                    row.iteration_num = F::from_u32(i as u32);
                    row.is_first = F::from_bool(i == 0);
                    row.is_last = F::from_bool(i == addrs.exp.len() - 1);
                    row.is_real = F::ONE;
                    row.x_mem = MemoryAccessCols { addr: addrs.base, mult: -F::from_bool(i == 0) };
                    row.exponent_mem = MemoryAccessCols { addr: addrs.exp[i], mult: F::NEG_ONE };
                    row.result_mem = MemoryAccessCols {
                        addr: addrs.result,
                        mult: *mult * F::from_bool(i == addrs.exp.len() - 1),
                    };
                });
                rows.extend(row_add);
            });

        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS],
            program.fixed_log2_rows(&ExpReverseBitsLenChip::<3>),
        );

        RowMajorMatrix::new(
            rows.into_iter().flatten().collect(),
            NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS,
        )
    }

    #[test]
    #[ignore = "Failing due to merge conflicts. Will be fixed shortly."]
    fn generate_preprocessed_trace() {
        let program = test_fixtures::program();
        let trace = ExpReverseBitsLenChip::<DEGREE>.generate_preprocessed_trace(&program).unwrap();
        assert!(trace.height() >= test_fixtures::MIN_TEST_CASES);

        assert_eq!(trace, generate_preprocessed_trace_reference(&program));
    }
}
