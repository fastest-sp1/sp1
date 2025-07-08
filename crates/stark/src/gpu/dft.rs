use crate::gpu::ffi::*;
use p3_dft::{TwoAdicSubgroupDft, compute_inverse_twiddles, compute_twiddles, };
use p3_field::{Field, Powers, TwoAdicField};
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView, BitReversibleMatrix, };
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use std::collections::BTreeMap;
use std::sync::Arc;
use spin::RwLock;
use p3_util::log2_strict_usize;
use p3_baby_bear::BabyBear;


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
        let twiddles: Vec<F> = root.powers().take(n ).collect();//naive
        
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
pub struct CudaDft {
    // Each CudaDft instance shares a single global cache to avoid redundant computations.
    twiddle_cache: TwiddleCache<BabyBear>,
}

impl<F> TwoAdicSubgroupDft<F> for CudaDft
where
    F: TwoAdicField + Ord,
    F: From<BabyBear> + Into<BabyBear>,
{
    type Evaluations = RowMajorMatrix<F>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        if h == 0 { return mat; }
        let log_h = log2_strict_usize(h);
        let twiddles = self.twiddle_cache.get_forward_twiddles(log_h);
        
        //naive is ok
        /*let result = unsafe {
            naive_dft_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                //twiddles.as_ptr() as *const BabyBear,
            )
        };*/
        let result = unsafe {
            fast_dft_batch_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                twiddles.as_ptr() as *const BabyBear,
            )
        };
        
        mat
    }

    fn idft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        if h == 0 { return mat; }
        let log_h = log2_strict_usize(h);

        let inverse_twiddles = self.twiddle_cache.get_inverse_twiddles(log_h);
   
        //naive ok
        /*let result = unsafe {
            naive_idft_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                inverse_twiddles.as_ptr() as *const BabyBear,
            )
        };*/

        let result = unsafe {
            fast_idft_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                inverse_twiddles.as_ptr() as *const BabyBear,
            )
        };
        
        mat
    }

    //naive is ok
    fn coset_dft_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> Self::Evaluations {
        let h = mat.height();
        if h == 0 { return mat; }
        let log_h = log2_strict_usize(h);
   
        let result = unsafe {
            naive_coset_dft_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                shift.into(),
            )
        };
        
        mat
    }


    //the following is ok for test_naive_coset_lde_batch
    /*fn coset_lde_batch(&self, 
                mut mat: RowMajorMatrix<F>,
                added_bits: usize,
                shift: F,) -> Self::Evaluations {
        let start = std::time::Instant::now();

        let h = mat.height();
        let w = mat.width();
        let log_h = log2_strict_usize(h);
        let log_lde_h = log_h + added_bits;
        let lde_h = 1 << log_lde_h;
        
        mat.values.resize(lde_h * w, F::ZERO);
        let result = unsafe {
            naive_coset_lde_batch_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                w as i32,
                added_bits as i32,
                shift.into(),
            )
        };
        let duration = start.elapsed();
        println!("naive_coset_lde_batch_gpu, duration:{:?}", duration);
        mat
    }*/

    fn coset_lde_batch(&self, 
                mut mat: RowMajorMatrix<F>,
                added_bits: usize,
                shift: F,) -> Self::Evaluations {
        //let start = std::time::Instant::now();

        let h = mat.height();
        let w = mat.width();
        let log_h = log2_strict_usize(h);
        let log_lde_h = log_h + added_bits;
        let lde_h = 1 << log_lde_h;

        let inverse_twiddles = self.twiddle_cache.get_inverse_twiddles(log_h);
        let forward_twiddles = self.twiddle_cache.get_forward_twiddles(log_lde_h);
        
        mat.values.resize(lde_h * w, F::ZERO);
        //fast ok
        /*let result = unsafe {
            fast_coset_lde_batch_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                w as i32,
                added_bits as i32,
                shift.into(),
                inverse_twiddles.as_ptr()  as *const BabyBear,
                forward_twiddles.as_ptr() as *const BabyBear,
            )
        };*/
        let result = unsafe {
            op_fast_coset_lde_batch_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                w as i32,
                added_bits as i32,
                shift.into(),
                inverse_twiddles.as_ptr()  as *const BabyBear,
                forward_twiddles.as_ptr() as *const BabyBear,
            )
        };


        //let duration = start.elapsed();
        //println!("fast_coset_lde_batch_gpu, duration:{:?}", duration);
        mat
    }

}


// It requires the `recursion_cuda` feature to be enabled for the test to run.
#[cfg(all(test, feature = "recursion_cuda"))]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2DitParallel, NaiveDft};
    use p3_field::Field;
    use p3_matrix::util::reverse_matrix_index_bits;
    use p3_field::PrimeCharacteristicRing;
     use p3_matrix::dense::DenseMatrix;

    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    // Helper function to generate a random matrix.
    fn generate_random_matrix(h: usize, w: usize) -> RowMajorMatrix<BabyBear> {
        let mut rng = thread_rng();
        let values = (0..h * w).map(|_| rng.sample(Standard)).collect();
        RowMajorMatrix::new(values, w)
    }

    #[test]
    fn test_fast_coset_lde_batch() { //pass
        const H: usize = 524288;
        const W: usize = 32;
        let log_h = log2_strict_usize(H);

        const ADDED_BITS: usize = 2; // blowup_factor = 4

        println!("Generating random matrix of size {}x{}", H, W);
        let input_matrix = generate_random_matrix(H, W);
        //let values :Vec<BabyBear>= vec![BabyBear::from_u32(1), BabyBear::from_u32(2),BabyBear::from_u32(3),BabyBear::from_u32(4),
        //                                BabyBear::from_u32(5),BabyBear::from_u32(6),BabyBear::from_u32(7),BabyBear::from_u32(8)];
        //let input_matrix = DenseMatrix::<BabyBear>::new(values, 2);

        let shift: BabyBear = BabyBear::from_u32(4);

        println!("shift:{:?}", shift);
        //println!("input_matrix:");
        //for row in input_matrix.row_slices() {
        //    println!("row:{:?}", row);
        //}
        
        // --- CPU reference ---
        //let cpu_dft = NaiveDft::default();
        //let dft_coeffs_view = cpu_dft.coset_lde_batch(input_matrix.clone(), ADDED_BITS, shift);
        //let cpu_coeffs = dft_coeffs_view.to_row_major_matrix();
        let cpu_dft = Radix2DitParallel::<BabyBear>::default();
        let cpu_result_evals = cpu_dft.coset_lde_batch(input_matrix.clone(), ADDED_BITS, shift);
        let cpu_result_matrix = cpu_result_evals.to_row_major_matrix();
        println!("CPU implementation finished.");

        // --- CUDA version ---
        let cuda_dft = CudaDft::default();
        let cuda_result_evals = cuda_dft.coset_lde_batch(input_matrix.clone(), ADDED_BITS, shift); //borrow the name coset_lde_batch
        let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();

        // --- Verification ---
        assert_eq!(cpu_result_matrix, cuda_result_matrix, "test_naive_coset_lde_batch results do not match!");
    }

    #[test]
    fn test_naive_coset_lde_batch() {//pass
        const H: usize = 256;
        const W: usize = 32;
        let log_h = log2_strict_usize(H);

        const ADDED_BITS: usize = 2; // blowup_factor = 4

        println!("Generating random matrix of size {}x{}", H, W);
        let input_matrix = generate_random_matrix(H, W);
        let shift: BabyBear = BabyBear::from_u32(4);

        println!("shift:{:?}", shift);
        //println!("input_matrix:");
        //for row in input_matrix.row_slices() {
        //    println!("row:{:?}", row);
        //}
        
        // --- CPU reference ---
        let cpu_dft = NaiveDft::default();
     
        let dft_coeffs_view = cpu_dft.coset_lde_batch(input_matrix.clone(), ADDED_BITS, shift);
        let cpu_coeffs = dft_coeffs_view.to_row_major_matrix();

        // --- CUDA version ---
        let cuda_dft = CudaDft::default();
        let cuda_result_evals = cuda_dft.coset_lde_batch(input_matrix.clone(), ADDED_BITS, shift); //borrow the name coset_lde_batch
        let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();

        // --- Verification ---
        assert_eq!(cpu_coeffs, cuda_result_matrix, "test_naive_coset_lde_batch results do not match!");
    }


    #[test]
    fn test_naive_idft_batch() {//pass
        const H: usize = 4;
        const W: usize = 2;

        println!("Generating random matrix of size {}x{}", H, W);
        let input_matrix = generate_random_matrix(H, W);
         println!("input_matrix:");

        for row in input_matrix.row_slices() {
            println!("row:{:?}", row);
        }

        // --- Run Plonky3's reference implementation (CPU) ---
        println!("Running reference CPU implementation (NaiveDft)...");
        let cpu_dft = NaiveDft::default();
        let cpu_dft_matrix = cpu_dft.idft_batch(input_matrix.clone());


        // --- Run our CUDA implementation ---
        println!("Running CUDA FFI implementation...");
        let cuda_dft = CudaDft::default();
        let cuda_dft_matrix = cuda_dft.idft_batch(input_matrix.clone());
        //let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();
        println!("CUDA implementation finished.");

        // --- Verification ---
        // ... (the rest of the verification logic is the same)
        assert_eq!(cpu_dft_matrix, cuda_dft_matrix, "DFT results do not match!");
    }

     #[test]
    fn test_fast_idft_batch() {//pass
        const H: usize = 4;
        const W: usize = 2;

        println!("Generating random matrix of size {}x{}", H, W);
        //let input_matrix = generate_random_matrix(H, W);
        let values :Vec<BabyBear>= vec![BabyBear::from_u32(1), BabyBear::from_u32(2),BabyBear::from_u32(3),BabyBear::from_u32(4),
                                        BabyBear::from_u32(5),BabyBear::from_u32(6),BabyBear::from_u32(7),BabyBear::from_u32(8)];
        let input_matrix = DenseMatrix::<BabyBear>::new(values, 2);
         println!("input_matrix:");

        for row in input_matrix.row_slices() {
            println!("row:{:?}", row);
        }

        // --- Run Plonky3's reference implementation (CPU) ---
        println!("Running reference CPU implementation (NaiveDft)...");
        let cpu_dft = NaiveDft::default();
        let cpu_dft_matrix = cpu_dft.idft_batch(input_matrix.clone());


        // --- Run our CUDA implementation ---
        println!("Running CUDA FFI implementation...");
        let cuda_dft = CudaDft::default();
        let cuda_dft_matrix = cuda_dft.idft_batch(input_matrix.clone());
        //let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();
        println!("CUDA implementation finished.");

        assert_eq!(cpu_dft_matrix, cuda_dft_matrix, "DFT results do not match!");
    }

    #[test]
    fn test_fast_dft_batch() {//pass
        const H: usize = 8;
        const W: usize = 4;

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
        let cuda_dft = CudaDft::default();
        let cuda_dft_matrix = cuda_dft.dft_batch(input_matrix.clone());
        //let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();
        println!("CUDA implementation finished.");

        // --- Verification ---
        // ... (the rest of the verification logic is the same)
        assert_eq!(cpu_dft_matrix, cuda_dft_matrix, "DFT results do not match!");
    }

    // ...
    #[test]
    fn test_naive_dft_batch() {//pass
        const H: usize = 4;
        const W: usize = 2;

        println!("Generating random matrix of size {}x{}", H, W);
        let input_matrix = generate_random_matrix(H, W);
         println!("input_matrix:");

        for row in input_matrix.row_slices() {
            println!("row:{:?}", row);
        }

        // --- Run Plonky3's reference implementation (CPU) ---
        println!("Running reference CPU implementation (Radix2DitParallel)...");
        let cpu_dft = NaiveDft::default();
        let cpu_dft_matrix = cpu_dft.dft_batch(input_matrix.clone());
  
        println!("CPU implementation finished.");

        // --- Run our CUDA implementation ---
        println!("Running CUDA FFI implementation...");
        let cuda_dft = CudaDft::default();
        let cuda_dft_matrix = cuda_dft.dft_batch(input_matrix.clone());
        //let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();
        println!("CUDA implementation finished.");

        // --- Verification ---
        // ... (the rest of the verification logic is the same)
        assert_eq!(cpu_dft_matrix, cuda_dft_matrix, "DFT results do not match!");
    }

    #[test]
    fn test_coset_dft_batch() {//pass
        const H: usize = 4;
        const W: usize = 2;

        let input_matrix = generate_random_matrix(H, W); 
        let shift: BabyBear = BabyBear::from_u32(4);
        println!("input_matrix:");
        for row in input_matrix.row_slices() {
            println!("row:{:?}", row);
        }

        println!("shift:{:?}", shift);

        println!("\nTesting coset_dft_batch for a {}x{} matrix...", H, W);

        let cpu_dft = NaiveDft::default();
        let cpu_coset_dft = cpu_dft.coset_dft_batch(input_matrix.clone(), shift);
        
        println!("CPU coset shift finished.");
        
        // --- CUDA version (remains the same) ---
         println!("Running CUDA FFI implementation...");
        let cuda_dft = CudaDft::default();
        let cuda_coset_dft = cuda_dft.coset_dft_batch(input_matrix.clone(), shift);
        //let cuda_result_matrix = cuda_result_evals.to_row_major_matrix();
        println!("CUDA implementation finished.");
 
        assert_eq!(
            cpu_coset_dft,
            cuda_coset_dft, 
            "Coset shift results do not match!"
        );
        println!("SUCCESS: CUDA Coset Shift is correct.");
    }
}