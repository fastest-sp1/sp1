use crate::gpu::ffi::*;
use p3_dft::{TwoAdicSubgroupDft};
use p3_field::{TwoAdicField, PrimeCharacteristicRing};

//use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView, BitReversibleMatrix, };
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use std::collections::BTreeMap;
use std::sync::Arc;
use spin::RwLock;
use p3_util::log2_strict_usize;
use p3_baby_bear::BabyBear;
use p3_matrix::Dimensions;
use std::os::raw::c_void;

use crate::baby_bear_poseidon2::Val;

// Handle for a batch of LDEs on the GPU
pub struct GpuLdeBatchHandle {
    pub d_data: *mut c_void,
    pub dimensions: Vec<Dimensions>,
    total_elements: usize,
}

impl Drop for GpuLdeBatchHandle {
    fn drop(&mut self) {
        if !self.d_data.is_null() {
            unsafe { cuda_free(self.d_data); }
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct GpuDft {
    // Each GpuDft instance shares a single global cache to avoid redundant computations.
    twiddle_cache: TwiddleCache<BabyBear>,
}

fn flatten_twiddles(twiddle_map: &BTreeMap<usize, Vec<Val>>) -> (Vec<Val>, Vec<i32>) {
    if twiddle_map.is_empty() {
        return (Vec::new(), vec![0, 0]); // Handle empty case
    }
    
    let mut flat_twiddles = Vec::new();
    let max_log_n = *twiddle_map.keys().max().unwrap();
    let mut offsets = vec![0i32; max_log_n + 2];
    
    let mut cumulative_size = 0;
    for log_n in 0..=max_log_n {
        offsets[log_n] = cumulative_size as i32;
        if let Some(twiddles) = twiddle_map.get(&log_n) {
            flat_twiddles.extend(twiddles);
            cumulative_size += twiddles.len();
        }
    }
    offsets[max_log_n + 1] = cumulative_size as i32;
    
    (flat_twiddles, offsets)
}

impl GpuDft {
    /// Performs a batch of coset LDEs in parallel on the GPU.
    /// The results are kept on the GPU, and a handle is returned.
    pub fn batch_coset_lde_on_gpu(
        &self,
        lde_requests: Vec<(RowMajorMatrix<Val>, usize, Val)>,
    ) -> GpuLdeBatchHandle {
        if lde_requests.is_empty() {
            return GpuLdeBatchHandle {
                d_data: std::ptr::null_mut(),
                dimensions: Vec::new(),
                total_elements: 0,
            };
        }

        // 1. Flatten inputs for FFI
        let mut h_evals_flat = Vec::new();
        let mut h_poly_info = Vec::new();
        let mut h_shifts = Vec::new();
        let mut dimensions = Vec::new();
        let mut required_inv_twiddles = BTreeMap::new();
        let mut required_fwd_twiddles = BTreeMap::new();

        for (evals, log_blowup, shift) in lde_requests {
            let h = evals.height();
            let w = evals.width();
            if h == 0 { continue; } // Skip empty matrices
            
            let log_h = log2_strict_usize(h);
            let log_lde_h = log_h + log_blowup;
            
            h_poly_info.extend_from_slice(&[h as i32, w as i32, log_blowup as i32]);
            h_shifts.push(shift);
            h_evals_flat.extend_from_slice(&evals.values);
            dimensions.push(Dimensions { height: 1 << log_lde_h, width: w });
            
            // Collect required twiddle sizes for both IDFT and DFT
            if !required_inv_twiddles.contains_key(&log_h) {
                required_inv_twiddles.insert(log_h, self.twiddle_cache.get_inverse_twiddles(log_h));
            }
            if !required_fwd_twiddles.contains_key(&log_lde_h) {
                required_fwd_twiddles.insert(log_lde_h, self.twiddle_cache.get_forward_twiddles(log_lde_h));
            }
        }

        // ================== THE FIX IS APPLIED HERE ==================
        // Flatten BOTH sets of twiddles and create their correct cumulative offset maps.
        let (h_all_inv_twiddles, h_inv_twiddle_offsets) = flatten_twiddles(&required_inv_twiddles);
        let (h_all_fwd_twiddles, h_fwd_twiddle_offsets) = flatten_twiddles(&required_fwd_twiddles);
        // =============================================================

        // 2. Call the FFI function
        let mut d_ldes_flat_out: *mut c_void = std::ptr::null_mut();
        let mut total_elements_out: usize = 0;
        
        let result = unsafe {
            dft_batch_lde_on_gpu(
                h_evals_flat.as_ptr(),
                h_evals_flat.len(),
                h_poly_info.as_ptr(),
                h_shifts.as_ptr(),
                dimensions.len() as i32, // num_polys
                h_all_inv_twiddles.as_ptr(),
                h_inv_twiddle_offsets.as_ptr(),
                h_all_fwd_twiddles.as_ptr(),
                h_fwd_twiddle_offsets.as_ptr(),
                &mut d_ldes_flat_out,
                &mut total_elements_out,
            )
        };
        
        if result != 0 {
            panic!("dft_batch_lde_on_gpu FFI call failed");
        }

        // 3. Return the handle
        GpuLdeBatchHandle {
            d_data: d_ldes_flat_out,
            dimensions,
            total_elements: total_elements_out,
        }
    }
    
    /// Downloads LDE data from a GPU handle to a CPU Vec of matrices.
    pub fn download_lde_batch(&self, handle: &GpuLdeBatchHandle) -> Vec<RowMajorMatrix<Val>> {
        if handle.total_elements == 0 {
            return Vec::new();
        }
        
        // 1. Create a CPU buffer of the correct total size.
        let mut cpu_buffer = vec![Val::ZERO; handle.total_elements];
        
        // 2. Copy the entire batch of LDEs from GPU to CPU in one go.
        let result = unsafe {
            cuda_memcpy_dtoh(
                cpu_buffer.as_mut_ptr() as *mut c_void,
                handle.d_data,
                handle.total_elements * std::mem::size_of::<Val>(),
            )
        };

        if result != 0 {
            panic!("download_lde_batch cuda_memcpy_dtoh failed: {}", result);
        }


        // 3. Un-flatten the buffer back into separate matrices.
        let mut result_matrices = Vec::with_capacity(handle.dimensions.len());
        let mut current_offset = 0;
        for dims in &handle.dimensions {
            let matrix_size = dims.height * dims.width;
            let matrix_values = cpu_buffer[current_offset..current_offset + matrix_size].to_vec();
            result_matrices.push(RowMajorMatrix::new(matrix_values, dims.width));
            current_offset += matrix_size;
        }
        
        result_matrices
    }
}

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


impl<F> TwoAdicSubgroupDft<F> for GpuDft
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

        let result = unsafe {
            fast_dft_batch_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                twiddles.as_ptr() as *const BabyBear,
            )
        };
        if result != 0 {
            panic!("fast_dft_batch_gpu  failed: {}", result);
        }
        
        mat
    }

    fn idft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        if h == 0 { return mat; }
        let log_h = log2_strict_usize(h);

        let inverse_twiddles = self.twiddle_cache.get_inverse_twiddles(log_h);
   
        let result = unsafe {
            fast_idft_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                mat.width() as i32,
                inverse_twiddles.as_ptr() as *const BabyBear,
            )
        };

        if result != 0 {
            panic!("fast_idft_gpu  failed: {}", result);
        }
        
        mat
    }

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
        
        //debug
        //let duration = start.elapsed();
        //println!("coset_lde_batch:twiddles, duration:{:?}", duration);

        mat.values.resize(lde_h * w, F::ZERO);
        //fast ok
        let result = unsafe {
            fast_coset_lde_batch_gpu(
                mat.values.as_mut_ptr() as *mut BabyBear,
                h as i32,
                w as i32,
                added_bits as i32,
                shift.into(),
                inverse_twiddles.as_ptr()  as *const BabyBear,
                forward_twiddles.as_ptr() as *const BabyBear,
            )
        };

        if result != 0 {
            panic!("fast_coset_lde_batch_gpu  failed: {}", result);
        }

        //let duration = start.elapsed();
        //println!("coset_lde_batch:op_fast_coset_lde_batch_gpu, duration:{:?}", duration);
        mat
    }

}


// It requires the `recursion_cuda` feature to be enabled for the test to run.
//#[cfg(all(test, feature = "recursion_cuda"))]
#[cfg(test)]
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
        const H: usize = 8192*2*2*2*2*2*2*2;
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
        assert_eq!(cpu_result_matrix, cuda_result_matrix, "test_naive_coset_lde_batch results do not match!");
        println!("test finished.");
    }

     #[test]
    fn test_fast_idft_batch() {//pass
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
    fn test_fast_dft_batch() {//pass
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
        assert_eq!(cpu_dft_matrix, cuda_dft_matrix, "DFT results do not match!");
    }

    #[test]
    fn test_transpose() {//pass
        const H: usize = 131072;
        const W: usize = 57;

        println!("\nTesting transpose for a {}x{} matrix...", H, W);
        let input_matrix = generate_random_matrix(H, W);

        let cpu_result = input_matrix.transpose(); // New width is H
        
        // --- Run our CUDA implementation ---
        let mut gpu_transposed_values = vec![BabyBear::default(); W * H];
        let result = unsafe {
            stark_transpose_gpu(
                input_matrix.values.as_ptr(),
                gpu_transposed_values.as_mut_ptr(),
                H as i32,
                W as i32
            )
        };
        assert_eq!(result, 0, "CUDA transpose function failed.");
        let gpu_result = RowMajorMatrix::new(gpu_transposed_values, H);

        // --- Verification ---
        assert_eq!(cpu_result, gpu_result, "CUDA transpose does not match the reference.");

    }

    use p3_dft::TwoAdicSubgroupDft;
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_matrix::Matrix;
    use itertools::izip; // For zipping iterators

    #[test]
    fn test_batch_lde_on_gpu_and_download() {
        // 1. SETUP: Create multiple polynomials of varying sizes.
        let mut rng = thread_rng();
        let mut lde_requests_cpu = Vec::new();
        let mut lde_requests_gpu = Vec::new();

        // Create a diverse batch of requests
        let test_cases = [
            (128, 4, 2), // h, w, log_blowup
            //(64, 10, 3),
            //(256, 16, 1),
        ];
        
        for (h, w, log_blowup) in test_cases {
            //let coeffs = generate_random_matrix(h, w);
             let values = vec![BabyBear::from_u32(1), BabyBear::from_u32(2),
                        BabyBear::from_u32(3), BabyBear::from_u32(4),
                        BabyBear::from_u32(5), BabyBear::from_u32(6),
                        BabyBear::from_u32(7), BabyBear::from_u32(8),];
            let coeffs = RowMajorMatrix::new(values, 2);
            
            //let shift: BabyBear = rng.r#gen();
            let shift = BabyBear::GENERATOR;

            lde_requests_cpu.push((coeffs.clone(), log_blowup, shift));
            lde_requests_gpu.push((coeffs, log_blowup, shift));
        }
        
        println!("\n--- Testing batched LDE on GPU for {} polynomials ---", lde_requests_cpu.len());
        
        // 2. COMPUTE ON CPU (The ground truth)
        println!("Computing reference LDEs on CPU...");
        let start_cpu = std::time::Instant::now();
        let cpu_dft = Radix2DitParallel::<BabyBear>::default(); 
       
        let expected_ldes: Vec<_> = lde_requests_cpu
            .into_iter()
            .map(|(mat, log_blowup, shift)| {
                let mut coeffs = cpu_dft.idft_batch(mat);
                
                // PANICS: possible panic if the new resized length overflows
                coeffs.values.resize(
                    coeffs
                        .values
                        .len()
                        .checked_shl(log_blowup.try_into().unwrap())
                        .unwrap(),
                    BabyBear::ZERO,
                );
                //cpu_dft.coset_dft_batch(coeffs, shift).to_row_major_matrix()
                cpu_dft.coset_dft_batch(coeffs, shift)
            })
            .collect();
        let duration_cpu = start_cpu.elapsed();
        println!("CPU computation finished in {:?}", duration_cpu);


        // 3. COMPUTE ON GPU (The implementation under test)
        println!("Computing LDEs on GPU and downloading...");
        let start_gpu = std::time::Instant::now();
        
        let gpu_dft = GpuDft::default();

        // NOW, run the GPU version which should follow the same Coeffs -> LDE logic
        let gpu_lde_handle = gpu_dft.batch_coset_lde_on_gpu(lde_requests_gpu);

        // 3b. Download the results from the GPU handle.
        let actual_ldes = gpu_dft.download_lde_batch(&gpu_lde_handle);
        let duration_gpu = start_gpu.elapsed();
        println!("GPU computation and download finished in {:?}", duration_gpu);

        // 4. COMPARE RESULTS
        assert_eq!(
            expected_ldes.len(),
            actual_ldes.len(),
            "Number of output matrices does not match!"
        );

        let mut all_match = true;
        for (i, (expected_mat, actual_mat)) in izip!(expected_ldes.iter(), actual_ldes.iter()).enumerate() {
             //println!("cpu-expected_mat: {:?}, ", expected_mat.inner.values);
            // println!("gpu-actual_mat: {:?}, ", actual_mat.values);
            if expected_mat.inner.values != actual_mat.values {
                println!("\n==================== MISMATCH FOUND in Matrix {} ====================", i);
                println!("  - Dimensions mismatch? Expected: {:?}, Actual: {:?}", expected_mat.dimensions(), actual_mat.dimensions());
                // Find first differing element for detailed error
                for (j, (exp_val, act_val)) in expected_mat.inner.values.iter().zip(&actual_mat.values).enumerate() {
                    if exp_val != act_val {
                        println!("  - First differing value at flat index {}:", j);
                        println!("    Expected (CPU): {:?}", exp_val);
                        println!("    Actual (GPU):   {:?}", act_val);
                        break;
                    }
                }
                all_match = false;
                break; // Stop at the first mismatched matrix
            }
        }

        if !all_match {
            panic!("Batched LDE results from GPU do not match CPU reference.");
        } else {
            println!("\nSUCCESS: `batch_coset_lde_on_gpu` and `download_lde_batch` are correct.");
        }
    }

}