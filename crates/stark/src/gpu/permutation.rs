//! GPU-accelerated implementation for generating the permutation trace.
//!

use crate::gpu::ffi::*;
use crate::{CudaResultCheck, GpuMatrix, GpuMatrixC, GpuMemBlk};

use crate::baby_bear_poseidon2::{Challenge, Val};
use p3_field::PrimeCharacteristicRing;

/// Generates the permutation trace for a chip entirely on the GPU.
pub fn generate_perm_trace(
    chip_id: i32,
    perm_width: i32,
    batch_size: i32,
    preprocessed_lde: &GpuMatrix<Val>,
    main_lde: &GpuMatrix<Val>,
    random_elements: &[Challenge],
    gpu_mem_blk: &GpuMemBlk,
) -> (GpuMatrix<Challenge>, Challenge) {
    let height = main_lde.height;

    if perm_width == 0 {
        //return (GpuMatrix::new(height, 0), Challenge::ZERO);
        panic!("perm_width is 0!");
    }

    // 1. Allocate the output permutation trace on the GPU.
    let perm_trace_gpu = GpuMatrix::<Challenge>::new(height, perm_width as usize, gpu_mem_blk);

    // 4. Call the main FFI orchestration function.
    let mut local_cumulative_sum = Challenge::ZERO;
    unsafe {
        let main_lde_c: GpuMatrixC = (main_lde).into();
        let preprocessed_lde_c: GpuMatrixC = (preprocessed_lde).into();
        let mut perm_trace_gpu_c: GpuMatrixC = (&perm_trace_gpu).into();
        let _ = generate_permutation_trace_gpu(
            chip_id,
            &main_lde_c,
            &preprocessed_lde_c,
            &mut perm_trace_gpu_c,
            random_elements.as_ptr(),
            batch_size,
            &mut local_cumulative_sum,
        )
        .check("generate_permutation_trace_gpu failed");
    }

    (perm_trace_gpu, local_cumulative_sum)
}
