use crate::{
    air::Block,
    chips::{
        alu_base::{BaseAluAccessCols, BaseAluValueCols},
        alu_ext::{ExtAluAccessCols, ExtAluValueCols},
        batch_fri::{BatchFRICols, BatchFRIPreprocessedCols},
        exp_reverse_bits::{ExpReverseBitsLenCols, ExpReverseBitsLenPreprocessedCols},
        fri_fold::{FriFoldCols, FriFoldPreprocessedCols},
        poseidon2_skinny::columns::{preprocessed::Poseidon2PreprocessedColsSkinny, Poseidon2},
        poseidon2_wide::columns::preprocessed::Poseidon2PreprocessedColsWide,
        public_values::{PublicValuesCols, PublicValuesPreprocessedCols},
        select::{SelectCols, SelectPreprocessedCols},
    },
    BaseAluInstr, BaseAluIo, BatchFRIEvent, BatchFRIInstrFFI, CommitPublicValuesEvent,
    CommitPublicValuesInstr, ExpReverseBitsEventFFI, ExpReverseBitsInstrFFI, ExtAluInstr, ExtAluIo,
    FriFoldEvent, FriFoldInstrFFI, Poseidon2Event, Poseidon2Instr, SelectEvent, SelectInstr,
    BatchFRIInstrFlat, InstrsFlatIdex, Address, FriFoldInstrFlat, ExpReverseBitsFlatIdex, 
    ExpReverseBitsEventFlatFFI, ExpReverseBitsInstrFlatFFI, Poseidon2SkinnyInstr, 
};
use p3_baby_bear::BabyBear;

#[link(name = "sp1-recursion-core-sys", kind = "static")]
unsafe extern "C-unwind" {
    pub fn alu_base_event_to_row_babybear(
        io: &BaseAluIo<BabyBear>,
        cols: &mut BaseAluValueCols<BabyBear>,
    );
    pub fn alu_base_instr_to_row_babybear(
        instr: &BaseAluInstr<BabyBear>,
        cols: &mut BaseAluAccessCols<BabyBear>,
    );

    pub fn alu_ext_event_to_row_babybear(
        io: &ExtAluIo<Block<BabyBear>>,
        cols: &mut ExtAluValueCols<BabyBear>,
    );
    pub fn alu_ext_instr_to_row_babybear(
        instr: &ExtAluInstr<BabyBear>,
        cols: &mut ExtAluAccessCols<BabyBear>,
    );

    pub fn batch_fri_event_to_row_babybear(
        io: &BatchFRIEvent<BabyBear>,
        cols: &mut BatchFRICols<BabyBear>,
    );
    pub fn batch_fri_instr_to_row_babybear(
        instr: &BatchFRIInstrFFI<BabyBear>,
        cols: &mut BatchFRIPreprocessedCols<BabyBear>,
        index: usize,
    );

    pub fn exp_reverse_bits_event_to_row_babybear(
        io: &ExpReverseBitsEventFFI<BabyBear>,
        i: usize,
        cols: &mut ExpReverseBitsLenCols<BabyBear>,
    );
    pub fn exp_reverse_bits_instr_to_row_babybear(
        instr: &ExpReverseBitsInstrFFI<BabyBear>,
        i: usize,
        len: usize,
        cols: &mut ExpReverseBitsLenPreprocessedCols<BabyBear>,
    );

    pub fn fri_fold_event_to_row_babybear(
        io: &FriFoldEvent<BabyBear>,
        cols: &mut FriFoldCols<BabyBear>,
    );
    pub fn fri_fold_instr_to_row_babybear(
        instr: &FriFoldInstrFFI<BabyBear>,
        i: usize,
        cols: &mut FriFoldPreprocessedCols<BabyBear>,
    );

    pub fn public_values_event_to_row_babybear(
        io: &CommitPublicValuesEvent<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesCols<BabyBear>,
    );
    pub fn public_values_instr_to_row_babybear(
        instr: &CommitPublicValuesInstr<BabyBear>,
        digest_idx: usize,
        cols: &mut PublicValuesPreprocessedCols<BabyBear>,
    );

    pub fn select_event_to_row_babybear(
        io: &SelectEvent<BabyBear>,
        cols: &mut SelectCols<BabyBear>,
    );
    pub fn select_instr_to_row_babybear(
        instr: &SelectInstr<BabyBear>,
        cols: &mut SelectPreprocessedCols<BabyBear>,
    );

    pub fn poseidon2_skinny_event_to_row_babybear(
        io: &Poseidon2Event<BabyBear>,
        cols: *mut Poseidon2<BabyBear>,
    );
    pub fn poseidon2_skinny_instr_to_row_babybear(
        instr: &Poseidon2Instr<BabyBear>,
        i: usize,
        cols: &mut Poseidon2PreprocessedColsSkinny<BabyBear>,
    );

    pub fn poseidon2_wide_event_to_row_babybear(
        input: *const BabyBear,
        input_row: *mut BabyBear,
        sbox_state: bool,
    );
    pub fn poseidon2_wide_instr_to_row_babybear(
        instr: &Poseidon2Instr<BabyBear>,
        cols: &mut Poseidon2PreprocessedColsWide<BabyBear>,
    );

    //#[cfg(feature = "cuda")]
    pub fn  process_alu_base_events_gpu(
        events_h: *const BaseAluIo<BabyBear>,
        events_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    pub fn process_alu_base_instructions_gpu(
        instrs_ptr: *const BaseAluInstr<BabyBear>,
        instrs_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    pub fn  process_alu_ext_events_gpu(
        events_h: *const ExtAluIo<Block<BabyBear>>,
        events_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    pub fn process_alu_ext_instructions_gpu(
        instrs_ptr: *const ExtAluInstr<BabyBear>,
        instrs_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

     pub fn  process_batch_fri_events_gpu(
        events_h: *const BatchFRIEvent<BabyBear>,
        events_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    /*
    //v1 ok
    pub fn process_batch_fri_instructions_gpu(
        instrs_ptr: *const BatchFRIInstrRowFFI<BabyBear>,
        instrs_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );*/
    pub fn process_batch_fri_instructions_gpu(
        instrs_h: *const BatchFRIInstrFlat<BabyBear>,
        instrs_len: usize,
        instrs_index_info_h: *const InstrsFlatIdex, 
        instrs_index_info_len: usize,  
        all_base_p_at_x_h: *const Address<BabyBear>,
        all_base_p_at_x_len: usize,
        all_ext_p_at_z_h: *const Address<BabyBear>,
        all_ext_p_at_z_len: usize,
        all_ext_alpha_pow_h: *const Address<BabyBear>,
        all_ext_alpha_pow_len: usize,    
        num_original_instrs: usize,
        rows_per_single_instr: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    pub fn  process_fri_fold_events_gpu(
        events_h: *const FriFoldEvent<BabyBear>,
        events_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    pub fn process_fri_fold_instructions_gpu(
        instrs_flat_h: *const FriFoldInstrFlat<BabyBear>,
        instrs_flat_len: usize,
        instrs_index_info_h: *const InstrsFlatIdex, 
        instrs_index_info_len: usize,
        all_ext_mat_opening_h: *const Address<BabyBear>,
        all_ext_mat_opening_len: usize,
        all_ext_ps_at_z_h: *const Address<BabyBear>,
        all_ext_ps_at_z_len: usize,
        all_alpha_pow_input_h: *const Address<BabyBear>,
        all_alpha_pow_input_len: usize,
        all_ro_input_h: *const Address<BabyBear>,
        all_ro_input_len: usize,
        all_alpha_pow_output_h: *const Address<BabyBear>,
        all_alpha_pow_output_len: usize,
        all_ro_output_h: *const Address<BabyBear>,
        all_ro_output_len: usize,
        all_alpha_pow_mults_h: *const BabyBear,
        all_alpha_pow_mults_len: usize,
        all_ro_mults_h: *const BabyBear,
        all_ro_mults_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_total_output_rows: usize,
        num_cols_per_row: usize,
    );

     pub fn  process_public_values_events_gpu(
        events_h: *const CommitPublicValuesEvent<BabyBear>,
        events_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        digit_size: usize,
        num_cols_per_row: usize,
    );

     pub fn process_public_values_instructions_gpu(
        instrs_ptr: *const CommitPublicValuesInstr<BabyBear>,
        instrs_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        digit_size: usize,
        num_cols_per_row: usize,
    );

     pub fn  process_select_events_gpu(
        events_h: *const SelectEvent<BabyBear>,
        events_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );
     
     pub fn process_select_instructions_gpu(
        instrs_ptr: *const SelectInstr<BabyBear>,
        instrs_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    pub fn process_exp_reverse_bits_events_gpu(
        events_h: *const ExpReverseBitsEventFlatFFI<BabyBear>,
        events_len: usize,
        events_index_info_h: *const ExpReverseBitsFlatIdex<BabyBear>, 
        events_index_info_len: usize,  
        all_events_exp_h: *const BabyBear,
        all_events_exp_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        total_events: usize,  //=total rows
        num_cols_per_row: usize,
    );

    pub fn process_exp_reverse_bits_instructions_gpu(
        instrs_h: *const ExpReverseBitsInstrFlatFFI<BabyBear>,
        instrs_len: usize,
        instrs_index_info_h: *const InstrsFlatIdex, 
        instrs_index_info_len: usize,  
        all_exp_bits_h: *const Address<BabyBear>,
        all_exp_bits_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        total_events: usize,  //=total rows
        num_cols_per_row: usize,
    );

    pub fn  process_poseidon2_wide_events_gpu(
        events_h: *const Poseidon2Event<BabyBear>,
        events_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        total_rows: usize,
        num_cols_per_row: usize,
        sbox_state: bool,
        width: usize,
    );

    pub fn process_poseidon2_wide_instructions_gpu(
        instrs_ptr: *const Poseidon2SkinnyInstr<BabyBear>,
        instrs_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    pub fn  process_poseidon2_skinny_events_gpu(
        events_h: *const Poseidon2Event<BabyBear>,
        events_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_cols_per_row: usize,
    );

    pub fn process_poseidon2_skinny_instructions_gpu(
        instrs_ptr: *const Poseidon2SkinnyInstr<BabyBear>,
        instrs_len: usize,
        output_h: *mut BabyBear,
        output_len: usize,
        num_rows_per_instr: usize,
        num_cols_per_row: usize,
    );
     
}


