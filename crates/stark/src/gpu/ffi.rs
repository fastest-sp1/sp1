use p3_baby_bear::BabyBear;


#[link(name = "sp1_stark_cuda", kind = "static")]
unsafe extern "C" {
    pub fn fast_coset_lde_batch_gpu(
        data: *mut BabyBear,
        rows: i32,
        cols: i32,
        added_bits: i32,
        shift: BabyBear,
        inverse_twiddles: *const BabyBear,
        forward_twiddles: *const BabyBear,
    ) -> i32;

    pub fn op_fast_coset_lde_batch_gpu(
        data: *mut BabyBear,
        rows: i32,
        cols: i32,
        added_bits: i32,
        shift: BabyBear,
        inverse_twiddles: *const BabyBear,
        forward_twiddles: *const BabyBear,
    ) -> i32;

    pub fn naive_dft_gpu(data: *mut BabyBear, h: i32, w: i32) -> i32;
    pub fn naive_idft_gpu(data: *mut BabyBear, h: i32, w: i32, inverse_twiddles: *const BabyBear) -> i32;
    pub fn naive_coset_dft_gpu(data: *mut BabyBear, h: i32, w: i32, shift: BabyBear) -> i32;

    //
    pub fn fast_dft_batch_gpu(data: *mut BabyBear, h: i32, w: i32, forward_twiddles: *const BabyBear) -> i32;
    pub fn fast_idft_gpu(data: *mut BabyBear, h: i32, w: i32, inverse_twiddles: *const BabyBear) -> i32;
    pub fn fast_coset_dft_gpu(data: *mut BabyBear, h: i32, w: i32, shift: BabyBear) -> i32;

    //only for test
     pub fn sp1_stark_bit_reverse_gpu(
        data: *mut BabyBear,
        rows: i32,
        cols: i32,
    ) -> i32;

    //for test
    pub fn naive_coset_lde_batch_gpu(
        data: *mut BabyBear,
        rows: i32,
        cols: i32,
        added_bits: i32,
        shift: BabyBear,
    ) -> i32;

    //for test
    pub fn sp1_stark_apply_shift_gpu(data: *mut BabyBear, rows: i32, cols: i32, shift: BabyBear) -> i32;
} 

