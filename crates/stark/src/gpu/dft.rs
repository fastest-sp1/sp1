use crate::gpu::ffi::*;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
//use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView, BitReversibleMatrix, };
use p3_baby_bear::BabyBear;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use spin::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::gpu::matrix::CudaResultCheck;
use crate::{GpuMatrix, GpuMatrixC, GpuMemBlk};

/// A thread-safe cache for twiddle factors.
#[derive(Default, Clone, Debug)]
struct TwiddleCache<F: TwoAdicField> {
    /// Twiddle factors for the forward DFT.
    forward_twiddles: Arc<RwLock<BTreeMap<usize, Vec<F>>>>,
    /// Twiddle factors for the inverse DFT.
    inverse_twiddles: Arc<RwLock<BTreeMap<usize, Vec<F>>>>,
}

impl<F: TwoAdicField + Ord> TwiddleCache<F> {
    /// Gets or computes the twiddle factors for an n-point forward DFT, where n = 1 << log_n.
    fn get_forward_twiddles(&self, log_n: usize) -> Vec<F> {
        if let Some(twiddles) = self.forward_twiddles.read().get(&log_n) {
            return twiddles.clone();
        }

        let mut cache = self.forward_twiddles.write();
        if let Some(twiddles) = cache.get(&log_n) {
            return twiddles.clone();
        }

        let n = 1 << log_n;
        let root = F::two_adic_generator(log_n);
        //let twiddles: Vec<F> = root.powers().take(n / 2).collect(); //radix2
        let twiddles: Vec<F> = root.powers().take(n).collect(); //naive

        cache.insert(log_n, twiddles.clone());
        twiddles
    }

    /// Gets or computes the twiddle factors for an n-point inverse DFT.
    fn get_inverse_twiddles(&self, log_n: usize) -> Vec<F> {
        if let Some(twiddles) = self.inverse_twiddles.read().get(&log_n) {
            return twiddles.clone();
        }

        let mut cache = self.inverse_twiddles.write();
        if let Some(twiddles) = cache.get(&log_n) {
            return twiddles.clone();
        }

        let n = 1 << log_n;
        let root_inv = F::two_adic_generator(log_n).inverse();
        let twiddles: Vec<F> = root_inv.powers().take(n / 2).collect();

        cache.insert(log_n, twiddles.clone());
        twiddles
    }
}

/// A DFT/LDE implementation accelerated by CUDA.
#[derive(Default, Clone, Debug)]
pub struct GpuDft {
    // Each GpuDft instance shares a single global cache to avoid redundant computations.
    twiddle_cache: TwiddleCache<BabyBear>,
}

impl<F> TwoAdicSubgroupDft<F> for GpuDft
where
    F: TwoAdicField + Ord,
    F: From<BabyBear> + Into<BabyBear>,
{
    type Evaluations = RowMajorMatrix<F>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        if h == 0 {
            return mat;
        }
        let log_h = log2_strict_usize(h);
        let twiddles = self.twiddle_cache.get_forward_twiddles(log_h);

        unsafe {
            let _ = fast_dft_batch_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                twiddles.as_ptr() as *const BabyBear,
            )
            .check("fast_dft_batch_gpu failed.");
        }

        mat
    }

    fn idft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        if h == 0 {
            return mat;
        }
        let log_h = log2_strict_usize(h);

        let inverse_twiddles = self.twiddle_cache.get_inverse_twiddles(log_h);

        unsafe {
            let _ = fast_idft_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                inverse_twiddles.as_ptr() as *const BabyBear,
            )
            .check("fast_idft_gpu failed.");
        };

        mat
    }

    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        //let start = std::time::Instant::now();
        let h = mat.height();
        let w = mat.width();
        let log_h = log2_strict_usize(h);
        let log_lde_h = log_h + added_bits;
        let lde_h = 1 << log_lde_h;

        let inverse_twiddles = self.twiddle_cache.get_inverse_twiddles(log_h);
        let forward_twiddles = self.twiddle_cache.get_forward_twiddles(log_lde_h);

        mat.values.resize(lde_h * w, F::ZERO);

        unsafe {
            let _ = fast_coset_lde_batch_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                w as i32,
                added_bits as i32,
                shift.into(),
                inverse_twiddles.as_ptr() as *const BabyBear,
                forward_twiddles.as_ptr() as *const BabyBear,
            )
            .check("fast_coset_lde_batch_gpu failed.");
        };

        mat
    }
}

impl GpuDft {
    /// Performs an out-of-place LDE, reading from one GPU buffer and writing to a new one.
    pub fn coset_lde_batch_gpu(
        &self,
        input_matrix: &GpuMatrix<BabyBear>,
        added_bits: usize,
        shift: BabyBear,
        gpu_mem_blk: &GpuMemBlk,
    ) -> GpuMatrix<BabyBear> {
        let h = input_matrix.height;
        let w = input_matrix.width;
        let log_h = log2_strict_usize(h);
        let log_lde_h = log_h + added_bits;
        let lde_h = 1 << log_lde_h;
        // Allocate a *new* GPU matrix for the output LDE.
        let lde_matrix = GpuMatrix::<BabyBear>::new(lde_h, w, gpu_mem_blk);

        // Precompute twiddles on the CPU (this is fast).
        let inverse_twiddles = self.twiddle_cache.get_inverse_twiddles(log_h);
        let forward_twiddles = self.twiddle_cache.get_forward_twiddles(log_lde_h);

        // Call a new FFI function that takes separate input and output pointers.
        unsafe {
            let input_matrix_c: GpuMatrixC = (input_matrix).into();
            let mut lde_matrix_c: GpuMatrixC = (&lde_matrix).into();
            let _ = fast_coset_lde_batch_data_in_gpu(
                &input_matrix_c,
                &mut lde_matrix_c,
                h as i32,
                w as i32,
                added_bits as i32,
                shift,
                inverse_twiddles.as_ptr(),
                forward_twiddles.as_ptr(),
            )
            .check("fast_coset_lde_batch_out_of_place_gpu failed");
        }

        lde_matrix
    }

    pub fn bit_reverse_rows(
        &self,
        input_matrix: &GpuMatrix<BabyBear>,
        domain_size: usize,
        gpu_mem_blk: &GpuMemBlk,
    ) -> GpuMatrix<BabyBear> {
        let w = input_matrix.width;

        // Allocate a *new* GPU matrix for the output LDE.
        let lde_matrix = GpuMatrix::<BabyBear>::new(domain_size, w, gpu_mem_blk);

        unsafe {
            let input_matrix_c: GpuMatrixC = (input_matrix).into();
            let mut lde_matrix_c: GpuMatrixC = (&lde_matrix).into();
            let _ = bit_reverse_rows_gpu(&input_matrix_c, &mut lde_matrix_c)
                .check("bit_reverse_rows_gpu failed");
        }

        lde_matrix
    }
}

//#[cfg(all(test, feature = "recursion_cuda"))]
#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_dft::{NaiveDft, Radix2DitParallel};
    use p3_field::Field;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::DenseMatrix;
    use p3_matrix::util::reverse_matrix_index_bits;

    use rand::distributions::{Distribution, Standard};
    use rand::{Rng, thread_rng};

    // Helper function to generate a random matrix.
    fn generate_random_matrix(h: usize, w: usize) -> RowMajorMatrix<BabyBear> {
        let mut rng = thread_rng();
        let values = (0..h * w).map(|_| rng.sample(Standard)).collect();
        RowMajorMatrix::new(values, w)
    }

    #[test]
    fn test_fast_coset_lde_batch() {
        //pass
        const H: usize = 8192 * 2 * 2 * 2 * 2 * 2 * 2 * 2;
        const W: usize = 4;
        let log_h = log2_strict_usize(H);

        const ADDED_BITS: usize = 1; // blowup_factor = 4

        println!("Generating random matrix of size {}x{}", H, W);
        let input_matrix = generate_random_matrix(H, W);
        //let values :Vec<BabyBear>= vec![BabyBear::from_u32(1), BabyBear::from_u32(2),BabyBear::from_u32(3),BabyBear::from_u32(4),
        //                                BabyBear::from_u32(5),BabyBear::from_u32(6),BabyBear::from_u32(7),BabyBear::from_u32(8)];
        //let input_matrix = DenseMatrix::<BabyBear>::new(values, 2);

        let shift: BabyBear = BabyBear::from_u32(31);

        // --- CPU reference ---
        let cpu_dft = Radix2DitParallel::<BabyBear>::default();
        let cpu_result_evals = cpu_dft.coset_lde_batch(input_matrix.clone(), ADDED_BITS, shift);
        let cpu_result_matrix = cpu_result_evals.to_row_major_matrix();
        println!("CPU implementation finished.");

        // --- CUDA version ---
        let cuda_dft = GpuDft::default();
        let cuda_result_evals = cuda_dft.coset_lde_batch(input_matrix.clone(), ADDED_BITS, shift); //borrow the name coset_lde_batch
        let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();

        // --- Verification ---
        assert_eq!(
            cpu_result_matrix, cuda_result_matrix,
            "test_naive_coset_lde_batch results do not match!"
        );
        println!("test finished.");
    }

    #[test]
    fn test_fast_coset_lde_batch_gpu() {
        //const H: usize = 16; //pass
        //const W: usize = 49;

        //const H: usize = 1; //pass
        //const W: usize = 6;

        //const H: usize = 4096; //pass
        //const W: usize = 10;
        //const H: usize = 256; //pass
        //const W: usize = 12;

        const H: usize = 8192; //pass
        const W: usize = 32;

        let log_h = log2_strict_usize(H);

        const ADDED_BITS: usize = 2; // blowup_factor = 4

        println!("Generating random matrix of size {}x{}", H, W);
        let input_matrix = generate_random_matrix(H, W);
        //let values :Vec<BabyBear>= vec![BabyBear::from_u32(1), BabyBear::from_u32(2),BabyBear::from_u32(3),BabyBear::from_u32(4),
        //                                BabyBear::from_u32(5),BabyBear::from_u32(6),BabyBear::from_u32(7),BabyBear::from_u32(8)];
        //let input_matrix = DenseMatrix::<BabyBear>::new(values, 2);

        let shift: BabyBear = BabyBear::from_u32(31);

        // --- CPU reference ---
        let cpu_dft = Radix2DitParallel::<BabyBear>::default();
        let cpu_origin_evals = cpu_dft.coset_lde_batch(input_matrix.clone(), ADDED_BITS, shift);

        println!("CPU implementation finished.");

        // --- CUDA version ---
        let cuda_dft = GpuDft::default();

        // Allocate the matrix directly on the GPU.
        //println!("--cpu-matrix, H={}, W={}, values={:?}.",H, W, input_matrix.values);
        let gpu_mem_blk = GpuMemBlk::new(100 * 1024 * 1024) //enough?
            .expect("Failed to create GPU memory block.");
        let gpu_matrix = GpuMatrix::<BabyBear>::from_vec(&input_matrix.values, H, W, &gpu_mem_blk);

        println!("--gpu-matrix, H={}, W={},", gpu_matrix.height, gpu_matrix.width);
        //test
        //let temp = gpu_matrix.to_host();
        //println!("--gpu-matrix, =values={:?}--", temp);

        let lde_matrix = cuda_dft.coset_lde_batch_gpu(&gpu_matrix, ADDED_BITS, shift, &gpu_mem_blk); //borrow the name coset_lde_batch
        let gpu_origin_evals = lde_matrix.to_host();

        let bit_reverse_lde =
            cuda_dft.bit_reverse_rows(&lde_matrix, lde_matrix.height, &gpu_mem_blk);
        let row_major_values = bit_reverse_lde.to_host();
        let gpu_result_rowmajor_matrix = RowMajorMatrix::new(row_major_values, W);

        // --- Verification ---

        assert_eq!(
            cpu_origin_evals.inner.values, gpu_origin_evals,
            "cpu_origin_evals results do not match!"
        );

        let cpu_result_rowmajor_matrix = cpu_origin_evals.to_row_major_matrix();
        assert_eq!(
            cpu_result_rowmajor_matrix, gpu_result_rowmajor_matrix,
            "cpu_result_rowmajor_matrix results do not match!"
        );

        println!("test finished.");
    }

    #[test]
    fn test_fast_idft_batch() {
        //pass
        const H: usize = 256;
        const W: usize = 412;

        println!("Generating random matrix of size {}x{}", H, W);
        let input_matrix = generate_random_matrix(H, W);
        //let values :Vec<BabyBear>= vec![BabyBear::from_u32(1), BabyBear::from_u32(2),BabyBear::from_u32(3),BabyBear::from_u32(4),
        //                                BabyBear::from_u32(5),BabyBear::from_u32(6),BabyBear::from_u32(7),BabyBear::from_u32(8)];
        //let input_matrix = DenseMatrix::<BabyBear>::new(values, 2);
        println!("input_matrix:");

        //for row in input_matrix.row_slices() {
        //    println!("row:{:?}", row);
        //}

        // --- Run Plonky3's reference implementation (CPU) ---
        println!("Running reference CPU implementation (NaiveDft)...");
        let cpu_dft = NaiveDft::default();
        let cpu_dft_matrix = cpu_dft.idft_batch(input_matrix.clone());

        // --- Run our CUDA implementation ---
        println!("Running CUDA FFI implementation...");
        let cuda_dft = GpuDft::default();
        let cuda_dft_matrix = cuda_dft.idft_batch(input_matrix.clone());
        //let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();
        println!("CUDA implementation finished.");

        assert_eq!(cpu_dft_matrix, cuda_dft_matrix, "DFT results do not match!");
    }

    #[test]
    fn test_fast_dft_batch() {
        //pass
        const H: usize = 256;
        const W: usize = 412;

        println!("Generating random matrix of size {}x{}", H, W);
        let input_matrix = generate_random_matrix(H, W);
        //let values :Vec<BabyBear>= vec![BabyBear::from_u32(1), BabyBear::from_u32(2),BabyBear::from_u32(3),BabyBear::from_u32(4),
        //                                BabyBear::from_u32(5),BabyBear::from_u32(6),BabyBear::from_u32(7),BabyBear::from_u32(8)];
        //let input_matrix = DenseMatrix::<BabyBear>::new(values, 2);
        println!("input_matrix:");

        for row in input_matrix.row_slices() {
            println!("row:{:?}", row);
        }

        // --- Run Plonky3's reference implementation (CPU) ---
        let cpu_dft = NaiveDft::default();
        let cpu_dft_matrix = cpu_dft.dft_batch(input_matrix.clone());

        // --- Run our CUDA implementation ---
        let cuda_dft = GpuDft::default();
        let cuda_dft_matrix = cuda_dft.dft_batch(input_matrix.clone());
        //let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();
        println!("CUDA implementation finished.");

        // --- Verification ---
        // ... (the rest of the verification logic is the same)
        assert_eq!(cpu_dft_matrix, cuda_dft_matrix, "DFT results do not match!");
    }

    #[test]
    fn test_transpose() {
        //pass
        //use p3_matrix::dense::transpose;
        // Use non-square dimensions to catch errors
        const H: usize = 131072;
        const W: usize = 57;

        println!("\nTesting transpose for a {}x{} matrix...", H, W);
        let input_matrix = generate_random_matrix(H, W);
        //println!("input_matrix:");
        //for row in input_matrix.row_slices() {
        //    println!("row:{:?}", row);
        //}

        let cpu_result = input_matrix.transpose(); // New width is H
        //println!("after transpose:");
        //for row in cpu_result.row_slices() {
        //    println!("row:{:?}", row);
        // }

        // --- Run our CUDA implementation ---
        let mut gpu_transposed_values = vec![BabyBear::default(); W * H];
        let result = unsafe {
            stark_transpose_gpu(
                input_matrix.values.as_ptr(),
                gpu_transposed_values.as_mut_ptr(),
                H as i32,
                W as i32,
            )
        };
        assert_eq!(result, 0, "CUDA transpose function failed.");
        let gpu_result = RowMajorMatrix::new(gpu_transposed_values, H);

        // --- Verification ---
        assert_eq!(cpu_result, gpu_result, "CUDA transpose does not match the reference.");
    }
}
