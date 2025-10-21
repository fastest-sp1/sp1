
//! A module for managing matrices allocated in GPU device memory.

// This entire module is only compiled when the `recursion_cuda` feature is enabled.
//#![cfg(feature = "recursion_cuda")]

use crate::gpu::ffi::*; // Assuming your CUDA FFI bindings are in this module
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;
use std::slice;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use crate::{baby_bear_poseidon2::{Val, Challenge}, GpuMemBlk};
use std::fmt;
//use std::sync::Mutex;
//use parking_lot::Mutex; 
//use once_cell::sync::Lazy;

// A global lock to serialize all CUDA memory allocation calls.
//static CUDA_ALLOC_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

use std::sync::Arc;



///FFI
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuMatrixC {
    pub ptr: *mut c_void,
    pub width: usize,
    pub height: usize,
}

// Implement the conversion from your safe GpuMatrix wrapper to the FFI struct.
impl<'a, T> From<&'a GpuMatrix<T>> for GpuMatrixC {
    fn from(gpu_matrix: &'a GpuMatrix<T>) -> Self {
        GpuMatrixC {
            // Get the raw pointer from the inner `GpuBuffer`.
            ptr: gpu_matrix.ptr,
            width: gpu_matrix.width,
            height: gpu_matrix.height,
        }
    }
}

#[derive(Debug)]
pub struct CudaError {
    pub code: i32,
    pub message: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA FFI Error (code {}): {}", self.code, self.message)
    }
}

impl std::error::Error for CudaError {}

/// A simple trait to provide a `.check()` method on CUDA result codes for cleaner error handling.
pub trait CudaResultCheck {
    type Error;
    fn check(self, msg: &str) -> Result<(), Self::Error>;
}

impl CudaResultCheck for i32 {
    type Error = CudaError;

    #[inline]
    fn check(self, msg: &str) -> Result<(), Self::Error> {
        if self == 0 {
            // cudaSuccess is 0, so this is the Ok case.
            Ok(())
        } else {
            // Any non-zero value is an error.
            Err(CudaError {
                code: self,
                message: msg.to_string(),
            })
        }
    }
}

/// This struct is `Copy`-able and does NOT manage the lifetime of the GPU memory it points to.
/// It is intended to be created by a `GpuArena`, which is the sole owner of the memory.
/// It has no `Drop` implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct GpuMatrix<T> {
    /// A raw pointer to the allocated memory on the CUDA device.
    pub ptr: *mut c_void,
    /// The number of columns in the matrix.
    pub width: usize,
    /// The number of rows in the matrix.
    pub height: usize,
    /// A zero-sized marker to inform the Rust compiler about the generic type `T`.
    _marker: PhantomData<T>,
}

// Make GpuMatrix usable across threads, which is necessary for parallel provers.
// This is safe because CUDA device pointers can be used from any thread
// that has access to the CUDA context.
unsafe impl<T> Send for GpuMatrix<T> {}
unsafe impl<T> Sync for GpuMatrix<T> {}

impl<T> GpuMatrix<T> {
    

    /// # Safety
    /// The caller is responsible for ensuring the pointer is valid and that its lifetime
    /// is managed correctly (e.g., by a `GpuArena`).
    pub unsafe fn from_raw_parts(ptr: *mut c_void, width: usize, height: usize) -> Self {
        Self {
            ptr,
            width,
            height,
            _marker: PhantomData,
        }
    }
    
    /// Returns the total number of elements in the matrix.
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Returns `true` if the matrix has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    


    pub fn row(&self, row_idx: usize) -> Result<Vec<T>, CudaError> {
        // 1. Perform bounds checking.
        assert!(
            row_idx < self.height,
            "Row index out of bounds: the height is {} but the index is {}",
            self.height,
            row_idx
        );

        if self.width == 0 {
            // If the matrix has no columns, every row is empty.
            return Ok(Vec::new());
        }

        // 2. Allocate a `Vec<T>` on the host for the result.
        // We can create it uninitialized because we guarantee it will be filled.
        let mut host_vec = Vec::with_capacity(self.width);

        // 3. Calculate memory offset and size for the specific row.
        let element_size = mem::size_of::<T>();
        let row_size_bytes = self.width * element_size;
        
        // The starting address of the row on the device is the base pointer
        // plus the offset for all preceding rows.
        let device_ptr_offset = unsafe {
            (self.ptr as *mut u8).add(row_idx * self.width * element_size)
        };

        // 4. Call `cudaMemcpy` to copy the single row.
        unsafe {
            // This is safe because we allocated the Vec with enough capacity.
            host_vec.set_len(self.width);

            cuda_memcpy_dtoh(
                host_vec.as_mut_ptr() as *mut c_void,
                device_ptr_offset as *const c_void,
                row_size_bytes,
            )
            .check(&format!("GpuMatrix::row({}) failed", row_idx))?;
        }
        
        // 5. Return the host vector.
        Ok(host_vec)
    }

    /// Copies data from a host `Vec` into this existing GPU matrix buffer.
    ///
    /// # Panics
    /// Panics if the length of `host_vec` does not match the total number of elements
    /// in the GPU matrix (`width * height`).
    pub fn copy_from_host(&self, host_vec: &[T]) {
        assert_eq!(
            host_vec.len(),
            self.width * self.height,
            "Host vector size does not match GPU matrix dimensions"
        );
        let size_bytes = host_vec.len() * mem::size_of::<T>();
        if size_bytes > 0 {
            unsafe {
                cuda_memcpy_htod(self.ptr, host_vec.as_ptr() as *const c_void, size_bytes)
                    .check("GpuMatrix::copy_from_host failed");
            }
        }
    }

    /// Copies data from this GPU matrix back to a new host `Vec`.
    pub fn to_host(&self) -> Vec<T> {
        let num_elements = self.width * self.height;
        if num_elements == 0 {
            return Vec::new();
        }
        
        // Create a Vec on the host with the correct capacity, but uninitialized.
        let mut host_vec = Vec::with_capacity(num_elements);
        
        let size_bytes = num_elements * mem::size_of::<T>();
        
        unsafe {
            // Set the length of the vector, as we are about to write into its buffer.
            // This is safe because we are guaranteeing the memory will be filled by the CUDA call.
            host_vec.set_len(num_elements);

            cuda_memcpy_dtoh(
                host_vec.as_mut_ptr() as *mut c_void,
                self.ptr,
                size_bytes,
            )
            .check("GpuMatrix::to_host failed");
        }
        
        host_vec
    }

    /// Returns the raw device pointer as a mutable typed pointer.
    /// This is intended for use in FFI calls.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Returns the raw device pointer as a constant typed pointer.
    /// This is intended for use in FFI calls.
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    
    pub fn from_vec(host_vec: &Vec<T>, height: usize, width: usize, gpu_mem_blk: &GpuMemBlk) -> Self {
        assert_eq!(host_vec.len(), height * width, "Vector length does not match matrix dimensions");
        //let _lock = CUDA_ALLOC_LOCK.lock(); // Also lock here
        //let gpu_mat = Self::new(height, width);
        let gpu_mat = gpu_mem_blk.alloc_matrix::<T>(height, width);

        gpu_mat.copy_from_host(host_vec);
        gpu_mat
    }

    pub fn new(height: usize, width: usize, gpu_mem_blk: &GpuMemBlk) -> Self {
        //let mut ptr: *mut c_void = std::ptr::null_mut();
        
        //unsafe { GpuMatrix::from_raw_parts(ptr, width, height) }
        let mat = gpu_mem_blk.alloc_matrix::<T>(height, width);
        mat
    }


    pub fn cut_rows(&self, r: usize, gpu_mem_blk: &GpuMemBlk )-> Self {
        // 1. Allocate a new GPU buffer for the clone.
        let new_matrix = gpu_mem_blk.alloc_matrix::<T>(r, self.width);
        
        let size_bytes = r * self.width * std::mem::size_of::<T>();
        if size_bytes > 0 {
            // 2. Perform a device-to-device memory copy.
            unsafe {
                cuda_memcpy_dtod(
                    new_matrix.ptr, // Destination
                    self.ptr,       // Source
                    size_bytes,
                )
                .check("GpuMatrix::cut_rows failed during dtod memcpy");
            }
        }
        
        // 3. Return the new matrix with its own, distinct pointer.
        new_matrix
    }
}

impl GpuMatrix<Challenge> {
    pub fn flatten_to_base(&self, gpu_mem_blk: &GpuMemBlk) -> GpuMatrix<Val> {
        //let _lock = CUDA_ALLOC_LOCK.lock(); // Also lock here
        let base_width = self.width * 4; // 4 is the extension degree
        //let mut flat_matrix = GpuMatrix::<Val>::new(self.height, base_width);
        let mut flat_matrix = gpu_mem_blk.alloc_matrix::<Val>(self.height, base_width);

        unsafe {
            flatten_permutation_trace_gpu(
                &self.into(), // Convert to GpuMatrixC
                &mut (&flat_matrix).into(),
                //self as *const GpuMatrix<Challenge>,
               // &mut flat_matrix as *mut GpuMatrix<Val>, 
            ).check("flatten_permutation_trace_gpu failed");
        }
        
        flat_matrix
    }
}

/*
/// The `Drop` implementation ensures that GPU memory is automatically freed
/// when a `GpuMatrix` goes out of scope in Rust, preventing memory leaks.
impl<T> Drop for GpuMatrix<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            //let _lock = CUDA_ALLOC_LOCK.lock(); // Also lock here
        //println!("---GPuMatrix, free.ptr={:p}", self.ptr);
            unsafe {
                // Call the CUDA FFI function to free the memory.
                cuda_free(self.ptr).check("GpuMatrix::drop failed");
            }
            // Set the pointer to null to prevent double-freeing.
            self.ptr = std::ptr::null_mut();
        }
    }

}

impl<T> Clone for GpuMatrix<T> {
    fn clone(&self) -> Self {
        // 1. Allocate a new GPU buffer for the clone.
        //let _lock = CUDA_ALLOC_LOCK.lock(); // Also lock here
        let new_matrix = Self::new(self.height, self.width);
        
        let size_bytes = self.height * self.width * std::mem::size_of::<T>();
        if size_bytes > 0 {
            // 2. Perform a device-to-device memory copy.
            unsafe {
                cuda_memcpy_dtod(
                    new_matrix.ptr, // Destination
                    self.ptr,       // Source
                    size_bytes,
                )
                .check("GpuMatrix::clone failed during dtod memcpy");
            }
        }
        
        // 3. Return the new matrix with its own, distinct pointer.
        new_matrix
    }
}

*/

// You can also implement helpful conversions for testing and debugging.
#[cfg(test)]
impl<T: Clone + Send + Sync + PartialEq> PartialEq<RowMajorMatrix<T>> for GpuMatrix<T> {
    fn eq(&self, other: &RowMajorMatrix<T>) -> bool {
        if self.width != other.width() || self.height != other.height() {
            return false;
        }
        let host_vec = self.to_host();
        host_vec == other.values
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::ffi::*;
    // Add necessary use statements if they aren't already there
    use crate::gpu::matrix::{GpuMatrix, GpuMatrixC, GpuMemBlk, };
    use crate::baby_bear_poseidon2::{Val, Challenge, StarkConfigCpu};
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{Rng, thread_rng};
    use p3_field::PrimeCharacteristicRing;

    use crate::gpu::matrix::CudaResultCheck;
    use crate::{BabyBearPoseidon2Inner, StarkGenericConfig};

    use p3_baby_bear::Poseidon2BabyBear;
    use p3_commit::{Pcs, Mmcs, PolynomialSpace};
    use p3_field::extension::BinomialExtensionField;

    use p3_symmetric::TruncatedPermutation;
    use p3_matrix::Matrix;

    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;

    use p3_fri::{FriConfig,   TwoAdicFriPcs};
    use p3_merkle_tree::MerkleTreeMmcs;

    use p3_util::log2_strict_usize;
    use p3_challenger::FieldChallenger;
    use p3_field::{PackedValue, Field};
    use p3_symmetric::PaddingFreeSponge;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    //type F = BabyBear;

    #[test]
    fn test_flatten_permutation_trace_gpu() {
        // 1. SETUP
        println!("--- Setting up test for flatten_permutation_trace_gpu ---");
        let mut rng = thread_rng();

        // Define dimensions for a non-trivial test matrix
        let height = 131072;
        let width = 2;

        //let height = 128;
        //let width = 2;

        // Create a random matrix of Challenge elements on the CPU
        let source_values_challenge: Vec<Challenge> = (0..height * width)
            .map(|_| rng.r#gen())
            .collect();
        let source_matrix_cpu = RowMajorMatrix::new(source_values_challenge, width);
        
        println!("Created a {}x{} source matrix of Challenge elements.", height, width);

        // 2. CPU REFERENCE COMPUTATION
        println!("Flattening matrix on CPU...");
        let start_cpu = std::time::Instant::now();
        let expected_flat_matrix_cpu:RowMajorMatrix<Val> = source_matrix_cpu.clone().flatten_to_base();
        println!("CPU flattening finished in {:?}", start_cpu.elapsed());

        // 3. GPU EXECUTION
        println!("Flattening matrix on GPU...");
        // a) Create a GpuMatrix<Challenge> and copy the source data to it.
        let gpu_mem_blk = GpuMemBlk::new(100 * 1024 *1024)   //100M ?
                .expect("Failed to create GPU memory block");

        let source_matrix_gpu = GpuMatrix::<Challenge>::from_vec(
            &source_matrix_cpu.values,
            height,
            width,
            &gpu_mem_blk,
        );

        // b) Call the GPU-based flatten_to_base method. This is the function we're testing.
        let start_gpu = std::time::Instant::now();
        let flat_matrix_gpu_handle: GpuMatrix<Val> = source_matrix_gpu.flatten_to_base(&gpu_mem_blk);
        println!("GPU flattening finished in {:?}", start_gpu.elapsed());

        // c) Download the result from the GPU handle back to the host.
        let actual_flat_values_gpu = flat_matrix_gpu_handle.to_host();

        // 4. COMPARE RESULTS
        println!("Comparing CPU and GPU results...");

        // a) Verify dimensions
        let expected_width = width * 4; // 4 is the extension degree
        assert_eq!(expected_flat_matrix_cpu.height(), height, "Height mismatch");
        assert_eq!(expected_flat_matrix_cpu.width(), expected_width, "Width mismatch");
        assert_eq!(flat_matrix_gpu_handle.height, height, "GPU handle height is incorrect");
        assert_eq!(flat_matrix_gpu_handle.width, expected_width, "GPU handle width is incorrect");
        assert_eq!(actual_flat_values_gpu.len(), height * expected_width, "Downloaded vector has incorrect length");

        // b) Verify data content
        assert_eq!(
            expected_flat_matrix_cpu.values,
            actual_flat_values_gpu,
            "The flattened data from the GPU does not match the CPU reference!"
        );

        println!("\nSUCCESS: GpuMatrix::flatten_to_base is correct.");
    }

    #[test]
    #[cfg(not(feature = "recursion_cuda"))]
    fn test_split_matrix_gpu() {
        // 1. SETUP
        println!("--- Setting up test for split_matrix_gpu ---");
        let mut rng = thread_rng();
        //let config = StarkConfigCpu::default();
        let config = BabyBearPoseidon2Inner::default();
        let pcs = config.pcs();

        // Define dimensions for a source matrix
        let height = 256;
        let width = 16;

        //let height = 131072;
        //let width = 8;
        let num_chunks = 4; // We will split it into 4 smaller matrices
        assert_eq!(height % num_chunks, 0, "Height must be divisible by num_chunks");

        // Create random source data on the CPU
        let source_matrix_cpu = RowMajorMatrix::<Val>::rand(&mut rng, height, width);
        let source_domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                        config.pcs(), height);
        
        println!(
            "Created a {}x{} source matrix, to be split into {} chunks.",
            height, width, num_chunks
        );

        // 2. CPU REFERENCE COMPUTATION
        println!("Splitting matrix on CPU...");
        let start_cpu = std::time::Instant::now();
        let expected_chunks_cpu: Vec<RowMajorMatrix<Val>> =
            source_domain.split_evals(num_chunks, source_matrix_cpu.clone());
        println!("CPU split finished in {:?}", start_cpu.elapsed());

        // 3. GPU EXECUTION
        println!("Splitting matrix on GPU...");
        // a) Copy source data to a GpuMatrix
        let gpu_mem_blk = GpuMemBlk::new(100 * 1024 *1024)   //100M ?
                .expect("Failed to create GPU memory block");

        let source_matrix_gpu = GpuMatrix::from_vec(
            &source_matrix_cpu.values,
            height,
            width,
            &gpu_mem_blk,
        );

        // b) Allocate empty output GpuMatrix handles for the chunks
        let chunk_height = height / num_chunks;
        let chunk_width = width;
        let  output_chunks_gpu: Vec<GpuMatrix<Val>> = (0..num_chunks)
            .map(|_| GpuMatrix::new(chunk_height, chunk_width, &gpu_mem_blk))
            .collect();
        
        // c) Create C-compatible descriptors for the output matrices
        let source_matrix_gpu_c: GpuMatrixC = (&source_matrix_gpu).into();
        let output_chunks_c: Vec<GpuMatrixC> = output_chunks_gpu.iter().map(|m| m.into()).collect();
        
        // d) Call the FFI function
        let start_gpu = std::time::Instant::now();
        unsafe {
            split_matrix_gpu(
               // &source_matrix_gpu.into(),
               // output_chunk_descriptors_c.as_ptr(),
               &source_matrix_gpu_c,// as  *const GpuMatrix<Val>,
               output_chunks_c.as_ptr(),// as  *const GpuMatrix<Val>,
                num_chunks as i32,
            ).check("split_matrix_gpu FFI call failed");
        }
        println!("GPU split finished in {:?}", start_gpu.elapsed());


        // 4. COMPARE RESULTS
        println!("Comparing CPU and GPU results...");
        assert_eq!(expected_chunks_cpu.len(), output_chunks_gpu.len());

        for i in 0..num_chunks {
            let cpu_chunk = &expected_chunks_cpu[i];
            let gpu_chunk_handle = &output_chunks_gpu[i];
            
            // Download data from the GPU for this chunk
            let gpu_chunk_values = gpu_chunk_handle.to_host();

            // Compare dimensions
            assert_eq!(cpu_chunk.height(), gpu_chunk_handle.height, "Height mismatch for chunk {}", i);
            assert_eq!(cpu_chunk.width(), gpu_chunk_handle.width, "Width mismatch for chunk {}", i);

            // Compare data
            assert_eq!(
                cpu_chunk.values,
                gpu_chunk_values,
                "Data mismatch for chunk {}", i
            );
        }
        
        println!("\nSUCCESS: split_matrix_gpu is correct.");
    }

    #[test]
    fn test_gpu_matrix_row_method() {
        // 1. Create a known matrix on the CPU
        let cpu_values = vec![
            1, 2, 3, // row 0
            4, 5, 6, // row 1
            7, 8, 9, // row 2
        ].into_iter().map(Val::from_u32).collect();
        
        // 2. Copy it to the GPU
        let gpu_mem_blk = GpuMemBlk::new(100 * 1024 *1024)   //100M ?
                .expect("Failed to create GPU memory block");

        let gpu_matrix = GpuMatrix::from_vec(&cpu_values, 3, 3, &gpu_mem_blk);

        // 3. Fetch a specific row using the new method
        let row_1_gpu = gpu_matrix.row(1).unwrap();

        // 4. Define the expected result
        let row_1_expected = vec![4, 5, 6].into_iter().map(Val::from_u32).collect::<Vec<_>>();

        // 5. Assert that they are equal
        assert_eq!(row_1_gpu, row_1_expected);

        println!("Successfully fetched row 1 from GPU.");
    }
}