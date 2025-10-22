use p3_baby_bear::BabyBear;
use std::os::raw::{c_int, c_uint, c_void};
//std::ffi::c_void
use crate::baby_bear_poseidon2::Challenge;
use crate::gpu::matrix::GpuMatrixC;
use crate::septic_digest::SepticDigest;

#[allow(unused_doc_comments)]
#[link(name = "sp1_stark_cuda", kind = "static")]
/// cbindgen:ignore
unsafe extern "C" {
    pub fn selectors_on_coset_gpu(
        trace_log_size: c_int,
        h_trace_subgroup_generator: *const BabyBear,
        coset_log_size: c_int,
        h_coset_shift: *const BabyBear,
        h_coset_subgroup_generator: *const BabyBear,
        h_is_first_row: *mut BabyBear,
        h_is_last_row: *mut BabyBear,
        h_is_transition: *mut BabyBear,
        h_inv_vanishing: *mut BabyBear,
    ) -> c_int;

    //test
    pub fn generate_selectors_on_device_gpu(
        trace_log_size: c_int,
        h_trace_subgroup_generator: BabyBear,
        coset_log_size: c_int,
        h_coset_shift: BabyBear,
        h_coset_subgroup_generator: BabyBear,
        h_is_first_row: *mut BabyBear,
        h_is_last_row: *mut BabyBear,
        h_is_transition: *mut BabyBear,
        h_inv_vanishing: *mut BabyBear,
    ) -> c_int;

    pub fn fast_coset_lde_batch_gpu(
        data: *mut BabyBear,
        rows: i32,
        cols: i32,
        added_bits: i32,
        shift: BabyBear,
        inverse_twiddles: *const BabyBear,
        forward_twiddles: *const BabyBear,
    ) -> i32;

    pub fn fast_coset_lde_batch_data_in_gpu(
        input_matrix: *const GpuMatrixC,
        lde_matrix: *mut GpuMatrixC,
        rows: i32,
        cols: i32,
        added_bits: i32,
        shift: BabyBear,
        inverse_twiddles: *const BabyBear,
        forward_twiddles: *const BabyBear,
    ) -> i32;

    pub fn bit_reverse_rows_gpu(
        input_matrix: *const GpuMatrixC,
        lde_matrix: *mut GpuMatrixC,
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

    //
    pub fn fast_dft_batch_gpu(
        data: *mut BabyBear,
        h: i32,
        w: i32,
        forward_twiddles: *const BabyBear,
    ) -> i32;
    pub fn fast_idft_gpu(
        data: *mut BabyBear,
        h: i32,
        w: i32,
        inverse_twiddles: *const BabyBear,
    ) -> i32;
    pub fn fast_coset_dft_gpu(data: *mut BabyBear, h: i32, w: i32, shift: BabyBear) -> i32;

    pub fn stark_merkle_commit_gpu(
        matrix_rows: *const BabyBear,
        matrix_dims: *const i32,
        num_matrices: i32,
        root_out: *mut BabyBear,
        device_tree_handle: *mut *mut c_void,
    ) -> i32;

    pub fn stark_merkle_commit_data_in_gpu(
        matrices: *const GpuMatrixC,
        num_matrices: i32,
        root_out: *mut BabyBear,
        device_tree_handle: *mut *mut c_void,
    ) -> i32;

    pub fn stark_merkle_open_batch_gpu(
        device_tree_handle: *mut c_void,
        index: i32,
        matrix_dims: *const i32,
        num_matrices: i32,
        opened_values_out: *mut BabyBear,
        proof_out: *mut BabyBear,
    ) -> i32;

    //op
    pub fn stark_merkle_generate_proofs_gpu(
        prover_data_ptrs: *const *const c_void,
        flat_indices: *const i32,
        offsets: *const i32,
        num_trees: i32,
        total_queries: i32,
        digist_elems: i32,
        flat_proofs_buffer: *mut BabyBear,
    ) -> i32;

    pub fn stark_merkle_free_gpu(device_tree_handle: *mut c_void);

    pub fn stark_test_poseidon2_permute_gpu(state: *mut BabyBear) -> i32;

    //pcs open
    pub fn fri_pcs_compute_quotient_for_height_gpu(
        specific_height: i32,
        // An array of pointers to LDE data buffers
        lde_data_ptrs: *const *const BabyBear,
        // Array of widths for these matrices
        lde_widths: *const i32,
        num_ldes: i32,

        // The evaluation domain
        coset: *const BabyBear,

        // Batching challenges
        alpha: *const Challenge,
        alpha_powers: *const Challenge,
        num_alpha_powers: i32,

        points_z_flat: *const Challenge,
        opened_values_y_flat: *const Challenge,
        num_points_per_mat: *const i32, // Starting offset for alpha^n

        // Output buffer
        quotient_evals_out: *mut Challenge,
    ) -> i32;

    pub fn fri_pcs_compute_quotient_for_height_data_in_gpu(
        // LDEs: data saved in GPU
        ldes: *const GpuMatrixC, //=Vec<GpuMatrixC>
        num_ldes: i32,
        // The evaluation domain in GPU
        coset: *const GpuMatrixC,
        // Batching challenges
        alpha: *const Challenge,
        //alpha_powers: *const GpuMatrixC,
        alpha_powers: *const Challenge,
        num_alpha_powers: i32,

        points_z_flat: *const Challenge,
        opened_values_y_flat: *const Challenge,
        num_points_per_mat: *const i32, // Starting offset for alpha^n
        // Output buffer
        quotient_evals_out: *mut Challenge,
    ) -> i32;

    //prove
    pub fn fold_even_odd_gpu(
        h_current_evals: *const Challenge,
        num_current_evals: i32,
        beta: Challenge,
        h_powers: *const Challenge,
        h_next_evals_out: *mut Challenge,
    ) -> i32;

    pub fn fri_commit_on_gpu(
        d_evals: *const Challenge,
        num_evals: i32,
        h_root_out: *mut BabyBear, // Assuming root is a digest of BabyBear
        h_prover_data_handle_out: *mut *mut c_void,
    ) -> i32;

    pub fn fri_fold_on_gpu(
        d_current_evals: *mut c_void,
        num_current_evals: i32,
        beta: Challenge,
        h_powers: *const Challenge,
        d_next_evals_out: *mut *mut c_void,
    ) -> i32;

    pub fn fri_injection_gpu(
        d_target_evals: *mut c_void,
        h_injected_evals: *const Challenge,
        num_elements: i32,
    ) -> i32;

    pub fn stark_fri_generate_proofs_gpu(
        // Input: A batch of proof generation tasks for multiple trees (layers)
        h_prover_data_handles: *const *const c_void, // Corresponds to `const void* const*`
        h_query_indices_flat: *const c_uint,         // Corresponds to `const unsigned int*`
        h_query_offsets: *const c_uint,              // Corresponds to `const unsigned int*`
        num_trees: c_int,
        total_queries: c_int,
        digest_size: c_int,

        // Output: A pointer to a host buffer where the flattened proofs will be written.
        h_proofs_out_flat: *mut BabyBear, // Corresponds to `bb31_t*`
    ) -> c_int;

    pub fn cuda_malloc(devPtr: *mut *mut c_void, size: usize) -> i32;

    /// Copies memory from the CPU (host) to the GPU (device).
    ///
    /// # Safety
    /// `dst_device` and `src_host` must be valid pointers, and `count` must not
    /// exceed the allocated size of either buffer.
    pub fn cuda_memcpy_htod(dst_device: *mut c_void, src_host: *const c_void, count: usize) -> i32;

    pub fn cuda_malloc_and_memset_zero(devPtr: *mut *mut c_void, size: usize) -> i32;
    pub fn cuda_memcpy_dtoh(dst: *mut c_void, src: *const c_void, count: usize) -> i32;
    pub fn cuda_memcpy_dtod(dst: *mut c_void, src: *const c_void, count: usize) -> i32;
    pub fn cuda_free(devPtr: *mut c_void) -> i32;

    // STARK quotient
    pub fn quotient_values_gpu(
        chip_id: c_int,
        h_main_trace: *const BabyBear,
        main_width: c_int,
        //main_height: c_int,
        h_prep_trace: *const BabyBear,
        prep_width: c_int,
        h_powers_of_alpha: *const Challenge,
        num_constraints: c_int,
        quotient_domain_size: c_int,
        h_perm_trace: *const BabyBear,
        perm_width: c_int,
        next_step: c_int,
        batch_size: c_int,
        h_perm_challenges: *const Challenge,
        h_local_cumulative_sum: *const Challenge,
        h_public_values: *const BabyBear,
        num_public_values: c_int,
        h_global_cumulative_sum: *const SepticDigest<BabyBear>,
        alpha_offset_base_alu: c_int,
        trace_log_size: c_int,
        coset_log_size: c_int,
        trace_subgroup_generator: BabyBear,
        coset_shift: BabyBear,
        coset_subgroup_generator: BabyBear,
        h_quotient_values_out: *mut Challenge,
    ) -> c_int;

    //the trace data is storaged in GPU
    pub fn quotient_values_data_in_gpu(
        chip_id: c_int,
        h_main_trace_lde: *const GpuMatrixC,
        h_prep_trace_lde: *const GpuMatrixC,
        h_perm_trace_lde: *const GpuMatrixC,
        h_powers_of_alpha: *const Challenge,
        num_constraints: c_int,
        quotient_domain_size: c_int,
        next_step: c_int,
        batch_size: c_int,
        h_perm_challenges: *const Challenge,
        h_local_cumulative_sum: *const Challenge,
        h_public_values: *const BabyBear,
        num_public_values: c_int,
        h_global_cumulative_sum: *const SepticDigest<BabyBear>,
        alpha_offset_base_alu: c_int,
        trace_log_size: c_int,
        coset_log_size: c_int,
        trace_subgroup_generator: BabyBear,
        coset_shift: BabyBear,
        coset_subgroup_generator: BabyBear,
        h_quotient_values_out: *mut Challenge,
    ) -> c_int;

    pub fn split_matrix_gpu(
        in_matrix: *const GpuMatrixC,
        h_out_matrices: *const GpuMatrixC,
        num_chunks: c_int,
    ) -> c_int;
    //
    pub fn generate_permutation_trace_gpu(
        chip_id: i32,
        main_lde: *const GpuMatrixC,
        prep_lde: *const GpuMatrixC,
        perm_trace_out: *mut GpuMatrixC,
        random_elements: *const Challenge,
        batch_size: i32,
        local_cumulative_sum_out: *mut Challenge,
    ) -> i32;

    pub fn flatten_permutation_trace_gpu(
        in_matrix: *const GpuMatrixC,
        out_matrix: *mut GpuMatrixC,
    ) -> i32;

    pub fn compute_inverse_denominators_for_points_gpu(
        h_points: *const Challenge,
        num_points: i32,
        h_max_log_heights: *const i32,
        d_coset: *const BabyBear,
        d_inv_denoms_ptr_array: *mut *mut Challenge, // Output: array of device pointers
    ) -> i32;

    pub fn interpolate_coset_gpu(
        //coset_evals: *const GpuMatrixC,
        coset_evals: *mut GpuMatrixC, //debug
        shift: BabyBear,
        point: Challenge,
        coset: *const GpuMatrixC,
        diff_invs: *const GpuMatrixC,
        h_interpolated_values_out: *mut Challenge,
    ) -> i32;
}

#[allow(unused_doc_comments)]
/// cbindgen:ignore
unsafe extern "C" {
    pub fn test_vpc_case_constant(
        h_main_row: *const BabyBear,
        h_prep_row: *const BabyBear,
        main_width: c_int,
        prep_width: c_int,
    ) -> BabyBear;

    pub fn test_vpc_case_single_main(
        h_main_row: *const BabyBear,
        h_prep_row: *const BabyBear,
        main_width: c_int,
        prep_width: c_int,
    ) -> BabyBear;

    pub fn test_vpc_case_complex(
        h_main_row: *const BabyBear,
        h_prep_row: *const BabyBear,
        main_width: c_int,
        prep_width: c_int,
    ) -> BabyBear;

    pub fn test_vpc_case_is_real(
        h_main_row: *const BabyBear,
        h_prep_row: *const BabyBear,
        main_width: c_int,
        prep_width: c_int,
    ) -> BabyBear;

    //test
    pub fn sp1_stark_bit_reverse_gpu(data: *mut BabyBear, rows: i32, cols: i32) -> i32;

    //test
    pub fn sp1_stark_apply_shift_gpu(
        data: *mut BabyBear,
        rows: i32,
        cols: i32,
        shift: BabyBear,
    ) -> i32;

    //test
    pub fn stark_transpose_gpu(
        input: *const BabyBear,
        output: *mut BabyBear,
        h: i32,
        w: i32,
    ) -> i32;

    pub fn stark_test_hash_leaves_gpu(
        flat_data: *const BabyBear,
        matrix_info: *const i32,
        num_tallest_matrices: i32,
        h_tallest: i32,
        digest_out: *mut BabyBear,
    ) -> i32;

    pub fn stark_test_hash_leaves_data_in_gpu(
        matrices: *const GpuMatrixC,
        num_matrices: i32,
        digest_out: *mut BabyBear,
    ) -> i32;

    pub fn stark_test_mat_compress_gpu(
        lde_data: *const BabyBear,
        h: i32,
        w: i32,
        alpha_powers: *const Challenge,
        mat_compressed_out: *mut Challenge,
    ) -> i32;

    pub fn stark_test_inv_denoms_gpu(
        z: Challenge,
        coset: *const BabyBear,
        h: i32,
        inv_denoms_out: *mut Challenge,
    ) -> i32;

    pub fn stark_test_quotient_loop_gpu(
        mat_compressed: *const Challenge,
        inv_denoms: *const Challenge,
        y_mat: Challenge,
        alpha_pow_offset: Challenge,
        H: i32,
        gpu_result: *mut Challenge,
    ) -> i32;

    //pub fn  stark_test_quartic_mul(
    //    a: *const Challenge,  b: *const Challenge, out: *mut Challenge)-> i32;

    //pass
    /*pub fn stark_test_quartic_pow(
        base: *const Challenge,
        exponent: u64,
        out: *mut Challenge,
    ) -> i32;*/
}
