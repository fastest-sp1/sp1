use crate::gpu::ffi::*;
use p3_commit::{BatchOpening, Mmcs, BatchOpeningRef};
use p3_matrix::{Dimensions, Matrix, };

use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::null_mut;
use p3_field::{PackedValue, PrimeField, PrimeCharacteristicRing};

use p3_baby_bear::BabyBear;
use p3_util::log2_strict_usize;
use p3_merkle_tree::MerkleTreeError;
use p3_merkle_tree::MerkleTreeError::{
    EmptyBatch, IncompatibleHeights, RootMismatch, WrongBatchSize, WrongHeight,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::BTreeMap;

//use crate::DIGEST_SIZE;
//use crate::baby_bear_poseidon2::{Val,};

pub type BatchedQueries = BTreeMap<usize, Vec<usize>>;

pub type BatchedOpenings<T, M> = BTreeMap<usize, BTreeMap<usize, BatchOpening<T, M>>>;

// A handle for GPU resources, not fully implemented for now.
#[derive(Debug)]
pub struct GpuMerkleTreeHandle(pub *mut c_void);

// Implement thread-safe sharing
unsafe impl Send for GpuMerkleTreeHandle {}
unsafe impl Sync for GpuMerkleTreeHandle {}

impl Default for GpuMerkleTreeHandle {
    fn default() -> Self {
        // The default value for a handle should be a null pointer,
        // indicating it doesn't point to any valid GPU resource yet.
        GpuMerkleTreeHandle(null_mut())
    }
}

impl Drop for GpuMerkleTreeHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // This ensures GPU memory is freed when the last Arc is dropped.
            unsafe { stark_merkle_free_gpu(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

// This is the ProverData our CUDA MMCS will use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMerkleProverData<M> {
    // A shared pointer to the GPU handle.
    #[serde(skip)] 
    pub handle: Arc<GpuMerkleTreeHandle>,
    
    // The matrices, stored in the order that the GPU used to build the tree (sorted by height, descending).
    pub sorted_inputs: Vec<M>,
    
    // A mapping to recover the original order.
    // `original_to_sorted_indices[original_index]` gives us the index in `sorted_inputs`.
    pub original_to_sorted_indices: Vec<usize>,
    
    // We also need the dimensions in the original order for `verify_batch`.
    pub dimensions: Vec<Dimensions>,

    pub _phantom: std::marker::PhantomData<M>
}

impl<M> GpuMerkleProverData<M> {
    // A constructor for when the data is fully on the GPU
    pub fn new_gpu_only(
        handle: Arc<GpuMerkleTreeHandle>,
        dimensions: Vec<Dimensions>
    ) -> Self {
        Self {
            handle,
            dimensions,
            sorted_inputs: Vec::new(),
            original_to_sorted_indices: Vec::new(),
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct GpuMerkleTreeMmcs<P, PW, H, C, const DIGEST_ELEMS: usize> {
    hash: H,
    compress: C,
    _phantom: PhantomData<(P, PW)>,
}

impl<P, PW, H, C, const DIGEST_ELEMS: usize> GpuMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS> {
    pub fn new(hash: H, compress: C) -> Self {
        Self { hash, compress, _phantom: PhantomData }
    }
}

// The implementation is now for the concrete type `BabyBear`.
impl<P, PW, H, C, const DIGEST_ELEMS: usize> Mmcs<P::Value> for GpuMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS>
where
    P: PackedValue<Value = BabyBear>,
    PW: PackedValue<Value = BabyBear>,

    P::Value: PrimeField + Into<BabyBear> + From<BabyBear>,
    PW::Value: PrimeField + Into<BabyBear> + From<BabyBear>,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
        + Sync,
    PW::Value: Eq,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    // The associated types are now concrete.
    type ProverData<M> = GpuMerkleProverData<M>;
    //type Commitment = Hash<BabyBear, BabyBear, DIGEST_ELEMS>;
    //type Proof = Vec<[BabyBear; DIGEST_ELEMS]>;
    type Commitment = Hash<P::Value, PW::Value, DIGEST_ELEMS>;
    type Proof = Vec<[PW::Value; DIGEST_ELEMS]>;
    type Error = MerkleTreeError;


    // The signature matches the trait exactly. `M` is the only generic type parameter.
    fn commit<M: Matrix<P::Value> + Clone>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        if inputs.is_empty() {
            panic!("Cannot commit to empty batch");
        }

         //let inputs_for_prover_data = inputs.clone();
        let num_inputs = inputs.len();
        let dimensions: Vec<Dimensions> = inputs.iter().map(|m| m.dimensions()).collect();

        // The rest of the logic operates on the `inputs` that were moved into this function.
        let mut indexed_inputs: Vec<(usize, M)> = inputs.into_iter().enumerate().collect();
        indexed_inputs.sort_by_key(|(_, m)| std::cmp::Reverse(m.height()));

        // `sorted_inputs` is the list of matrices in the order the GPU will see them.
        let sorted_inputs: Vec<M> = indexed_inputs.iter().map(|(_, m)| m.clone()).collect();

        // Create the mapping from original index to the new sorted index.
        let mut original_to_sorted_indices = vec![0; num_inputs];
        for (sorted_idx, (original_idx, _)) in indexed_inputs.iter().enumerate() {
            original_to_sorted_indices[*original_idx] = sorted_idx;
        }
        
        let mut flat_data: Vec<P::Value> = Vec::new();
        let mut matrix_info = Vec::new();

        // We iterate over references to avoid consuming `indexed_inputs` yet.
        for (_, m) in &indexed_inputs {
            let offset = flat_data.len();
            matrix_info.extend([offset as i32, m.height() as i32, m.width() as i32]);
            for r in 0..m.height() {
                flat_data.extend(m.row(r).unwrap());
            }
        }
        
        // --- 4. Call the FFI Function ---
        let mut root_out = [P::Value::ZERO; DIGEST_ELEMS];
        let mut handle_out: *mut c_void = null_mut();
        
        let result = unsafe {
            // Pass the sorted and flattened data to the GPU.
            stark_merkle_commit_gpu(
                flat_data.as_ptr() as *const BabyBear,
                matrix_info.as_ptr(),
                indexed_inputs.len() as i32,
                root_out.as_mut_ptr() as *mut BabyBear,
                &mut handle_out,
            )
        };
        if result != 0 {
            panic!("CUDA Merkle commit failed with code: {}", result);
        }
        
        // --- 5. Construct and Return Final ProverData ---
        // Construct the new, comprehensive ProverData.
        let prover_data = GpuMerkleProverData {
            handle: Arc::new(GpuMerkleTreeHandle(handle_out)),
            sorted_inputs,
            original_to_sorted_indices,
            dimensions, // Storing original dimensions for verify_batch
            _phantom: std::marker::PhantomData,
        };

        (root_out.into(), prover_data)
    }

    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<P::Value, Self> {
        
        let num_matrices = prover_data.dimensions.len();
        let matrix_info: Vec<i32> = prover_data.dimensions.iter()
            .scan(0, |offset, dims| {
                let current_offset = *offset;
                *offset += dims.height * dims.width;
                Some([current_offset as i32, dims.height as i32, dims.width as i32])
            })
            .flatten()
            .collect();
            
        // 1. Allocate buffers for the FFI function to write into.
        let total_opened_values_len: usize = prover_data.dimensions.iter().map(|d| d.width).sum();
        let mut flat_opened_values = vec![P::Value::ZERO; total_opened_values_len];

        let log_max_height = prover_data.dimensions.iter().map(|d| d.height).max().unwrap_or(0).next_power_of_two().trailing_zeros() as usize;
        let mut proof_buffer = vec![[P::Value::ZERO; DIGEST_ELEMS]; log_max_height];

        // 2. Call the FFI function.
        let result = unsafe {
            stark_merkle_open_batch_gpu(
                prover_data.handle.0,
                index as i32,
                matrix_info.as_ptr(),
                num_matrices as i32,
                flat_opened_values.as_mut_ptr()  as *mut BabyBear,
                proof_buffer.as_mut_ptr() as *mut BabyBear,
            )
        };
        if result != 0 { panic!("CUDA Merkle open_batch failed"); }

        // 3. Reconstruct the openings from the flat buffer.
        let mut openings = Vec::with_capacity(num_matrices);
        let mut current_pos = 0;
        for dims in &prover_data.dimensions {
            let end_pos = current_pos + dims.width;
            openings.push(flat_opened_values[current_pos..end_pos].to_vec());
            current_pos = end_pos;
        }

        BatchOpening::new(openings, proof_buffer)
    }

    fn get_matrices<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>, // `prover_data` is `&'a GpuMerkleProverData`
    ) -> Vec<&'a M> {
        let num_matrices = prover_data.original_to_sorted_indices.len();
        if num_matrices == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(num_matrices);

        for original_idx in 0..num_matrices {
            let sorted_idx = prover_data.original_to_sorted_indices[original_idx];
            
            result.push(&prover_data.sorted_inputs[sorted_idx]);
        }

        result
    }

    // In GpuMerkleTreeMmcs impl
    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        mut index: usize,
        batch_proof: BatchOpeningRef<P::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = batch_proof.unpack();

        // Check that the openings have the correct shape.
        if dimensions.len() != opened_values.len() {
            return Err(WrongBatchSize);
        }

        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| std::cmp::Reverse(dims.height))
            .peekable();

        // Matrix heights that round up to the same power of two must be equal
        if !heights_tallest_first
            .clone()
            .map(|(_, dims)| dims.height)
            .tuple_windows()
            .all(|(curr, next)| {
                curr == next || curr.next_power_of_two() != next.next_power_of_two()
            })
        {
            return Err(IncompatibleHeights);
        }

        // Get the initial height padded to a power of two. As heights_tallest_first is sorted,
        // the initial height will be the maximum height.
        // Returns an error if either:
        //              1. proof.len() != log_max_height
        //              2. heights_tallest_first is empty.
        let mut curr_height_padded = match heights_tallest_first.peek() {
            Some((_, dims)) => {
                let max_height = dims.height.next_power_of_two();
                let log_max_height = log2_strict_usize(max_height);
                if opening_proof.len() != log_max_height {
                    return Err(WrongHeight {
                        log_max_height,
                        num_siblings: opening_proof.len(),
                    });
                }
                max_height
            }
            None => return Err(EmptyBatch),
        };

        // Hash all matrix openings at the current height.
        let mut root = self.hash.hash_iter_slices(
            heights_tallest_first
                .peeking_take_while(|(_, dims)| {
                    dims.height.next_power_of_two() == curr_height_padded
                })
                .map(|(i, _)| opened_values[i].as_slice()),
        );

        for &sibling in opening_proof {
            // The last bit of index informs us whether the current node is on the left or right.
            let (left, right) = if index & 1 == 0 {
                (root, sibling)
            } else {
                (sibling, root)
            };

            // Combine the current node with the sibling node to get the parent node.
            root = self.compress.compress([left, right]);
            index >>= 1;
            curr_height_padded >>= 1;

            // Check if there are any new matrix rows to inject at the next height.
            let next_height = heights_tallest_first
                .peek()
                .map(|(_, dims)| dims.height)
                .filter(|h| h.next_power_of_two() == curr_height_padded);
            if let Some(next_height) = next_height {
                // If there are new matrix rows, hash the rows together and then combine with the current root.
                let next_height_openings_digest = self.hash.hash_iter_slices(
                    heights_tallest_first
                        .peeking_take_while(|(_, dims)| dims.height == next_height)
                        .map(|(i, _)| opened_values[i].as_slice()),
                );

                root = self.compress.compress([root, next_height_openings_digest]);
            }
        }

        // The computed root should equal the committed one.
        if commit == &root {
            Ok(())
        } else {
            Err(RootMismatch)
        }
    }
}

pub trait BatchOpenableMmcs<T, M>: Mmcs<T>
where
    T: Send + Sync + Clone,
    M: Matrix<T> + Clone,
{
    fn open_batches_batched(
        &self,
        log_global_max_height: usize,
        queries: &BatchedQueries, // BatchedQueries is now generic-free
    ) -> BatchedOpenings<T, Self>;
}

impl<P, PW, H, C, const DIGEST_ELEMS: usize, M> 
    BatchOpenableMmcs<P::Value, M> for GpuMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS>
where
    M: Matrix<P::Value> + Clone,
    P: PackedValue<Value = BabyBear>,
    PW: PackedValue<Value = BabyBear>,

    P::Value: PrimeField + Into<BabyBear> + From<BabyBear>,
    PW::Value: PrimeField + Into<BabyBear> + From<BabyBear>,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
        + Sync,
    PW::Value: Eq,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    fn open_batches_batched(
        &self,
        log_global_max_height: usize,
        queries: &BatchedQueries,//==<<GpuMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS> as Mmcs>::ProverData<M>>
    ) -> BatchedOpenings<P::Value, Self> {
        
        if queries.is_empty() {
            return BTreeMap::new();
        }

        // 1. Prepare data for FFI (for proof generation)
        let mut prover_data_ptrs: Vec<*const c_void> = Vec::new();
        let mut flat_indices: Vec<u32> = Vec::new();
        let mut offsets: Vec<u32> = vec![0];
        let mut prover_data_vec: Vec<&<GpuMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS> as Mmcs<P::Value>>::ProverData<M>> = Vec::new();

        for (&pd_ptr_usize, indices) in queries {
            let pd_ptr = pd_ptr_usize as *const <GpuMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS> as Mmcs<P::Value>>::ProverData<M>;
            let pd = unsafe { &*pd_ptr };
            
            // Pass the raw GPU handle to the FFI function
            prover_data_ptrs.push(pd.handle.0);
            prover_data_vec.push(pd);

            // We must calculate the reduced_index for each query before putting it in `flat_indices`.
            let log_max_height_for_this_pd = log2_strict_usize(pd.sorted_inputs[0].height());
            let bits_reduced = log_global_max_height - log_max_height_for_this_pd;

            for &index in indices {
                let reduced_index = index >> bits_reduced;
                flat_indices.push(reduced_index as u32);
            }
        
            offsets.push(flat_indices.len() as u32);
        }
        
        let total_queries = flat_indices.len();
        let num_trees = prover_data_ptrs.len();

        // 2. Calculate output buffer size for proofs and allocate it on the HOST
        let total_proof_elements: usize = prover_data_vec.iter()
            .zip(offsets.windows(2))
            .map(|(pd, offset_window)| {
                let num_queries_for_tree = offset_window[1] - offset_window[0];
                let log_max_height: usize = pd.dimensions.iter().map(|d| d.height).max().unwrap_or(0).next_power_of_two().trailing_zeros() as usize;
                (num_queries_for_tree as usize) * log_max_height * DIGEST_ELEMS
            })
            .sum();

        let mut flat_proofs_host_buffer: Vec<PW::Value> = vec![PW::Value::ZERO; total_proof_elements];

        // 3. FFI Call to generate all proofs in a single batch
        let result = unsafe {
            stark_merkle_generate_proofs_gpu(
                prover_data_ptrs.as_ptr(),
                flat_indices.as_ptr() as *const i32,
                offsets.as_ptr() as *const i32,
                num_trees as i32,
                total_queries as i32,
                DIGEST_ELEMS as i32,
                flat_proofs_host_buffer.as_mut_ptr(),
            )
        };
        if result != 0 { panic!("stark_merkle_generate_proofs_gpu FFI call failed"); }

        // 4. Reconstruct the final results
        let mut results: BatchedOpenings<P::Value, Self> = BTreeMap::new();
        let mut proof_cursor = 0;

        // Iterate through each ProverData that was queried
        for (_i, pd) in prover_data_vec.iter().enumerate() {
            let pd_ptr_usize = *pd as *const _ as usize;
            let indices_for_this_tree = queries.get(&pd_ptr_usize).unwrap();
            let mut openings_for_this_tree: BTreeMap<usize, BatchOpening<P::Value, Self>> = BTreeMap::new();

            let log_max_height: usize = pd.dimensions.iter().map(|d| d.height).max().unwrap_or(0).next_power_of_two().trailing_zeros() as usize;


            // For each query for this tree, assemble its BatchOpening
            for &index in indices_for_this_tree {
                // PART A: Extract the proof generated by the GPU
                let proof_size_digests :usize = log_max_height;
                let proof_size_elements = proof_size_digests * DIGEST_ELEMS;
                let proof_slice_elements = &flat_proofs_host_buffer[proof_cursor..proof_cursor + proof_size_elements];
                let proof_siblings: Vec<[PW::Value; DIGEST_ELEMS]> = proof_slice_elements
                    .chunks_exact(DIGEST_ELEMS)
                    .map(|chunk| chunk.try_into().expect("Chunk size should be correct"))
                    .collect();
                let opening_proof = proof_siblings; // In MMCS, Proof is Vec<[PW::Value; DIGEST_ELEMS]>
                proof_cursor += proof_size_elements;

                // PART B: Get the opened values directly from CPU memory (this is fast)
                let opened_values: Vec<Vec<P::Value>> = pd.sorted_inputs.iter().map(|matrix| { // <-- Use `sorted_inputs`
                    let log_height = log2_strict_usize(matrix.height());
                    //let log_max_height_for_this_pd = log2_strict_usize(pd.sorted_inputs[0].height()); // Max height within this batch
                    
                    // Use the GLOBAL max height for the reduction, as required by FRI.
                    let bits_reduced = log_global_max_height - log_height;
                    let reduced_index = index >> bits_reduced;
                    
                    matrix.row(reduced_index).unwrap().into_iter().collect()
                }).collect();

                let batch_opening = BatchOpening { opened_values, opening_proof };
            
                // Insert into the inner map using the index as the key
                openings_for_this_tree.insert(index, batch_opening);
            }
            results.insert(pd_ptr_usize, openings_for_this_tree);
        }

        results
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::{Field, PackedValue};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use crate::baby_bear_poseidon2::{Val, Challenge, MyHash, MyCompress, my_perm};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;
    use crate::{DIGEST_SIZE, GpuValMmcs};
    use p3_symmetric::Permutation;

    // Redefine types for testing
    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type H = MyHash;
    type C = MyCompress;
    type P = <F as Field>::Packing;
    type PW = <F as Field>::Packing;

     #[test]
    fn test_only_cpu_avx_merkle_commit() {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        
        // Create CPU and GPU versions of the MMCS
        let cpu_mmcs = p3_merkle_tree::MerkleTreeMmcs::<P, PW, H, C, DIGEST_SIZE>::new(hash.clone(), compress.clone());
        
        // Create some test matrices
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let mat1 = RowMajorMatrix::<F>::rand(&mut rng, 512, 32);
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 65536, 18);
        let mat3 = RowMajorMatrix::<F>::rand(&mut rng, 16384, 39);
        let mat4 = RowMajorMatrix::<F>::rand(&mut rng, 8192, 412);
       
        let start = std::time::Instant::now();
        let (cpu_root, _) = cpu_mmcs.commit(vec![mat1.clone(), mat2.clone(), mat3.clone(), mat4.clone()]);
        let duration = start.elapsed();
        println!("-- cpu commit , duration:{:?}", duration);
    }

    #[test]
    fn test_merkle_commit() {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        
        // Create CPU and GPU versions of the MMCS
        let cpu_mmcs = p3_merkle_tree::MerkleTreeMmcs::<P, PW, H, C, DIGEST_SIZE>::new(hash.clone(), compress.clone());
        let gpu_mmcs = GpuValMmcs::new(hash, compress);

        // Create some test matrices
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let mat1 = RowMajorMatrix::<F>::rand(&mut rng, 512, 32);
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 65536, 18);
        let mat3 = RowMajorMatrix::<F>::rand(&mut rng, 16384, 39);
        let mat4 = RowMajorMatrix::<F>::rand(&mut rng, 8192, 412);
        println!("input, mat1(h * w):{}*{}, mat2(h * w):{}*{}", mat1.height(), mat1.width(),  mat2.height(), mat2.width());
        // Commit using both implementations
        let start = std::time::Instant::now();
        let (cpu_root, _) = cpu_mmcs.commit(vec![mat1.clone(), mat2.clone(), mat3.clone(), mat4.clone()]);
         let duration = start.elapsed();
        println!("-- cpu commit , duration:{:?}", duration);
        let start = std::time::Instant::now();
        let (gpu_root, _) = gpu_mmcs.commit(vec![mat1, mat2, mat3, mat4]);
         let duration = start.elapsed();
        println!("-- GPU commit , duration:{:?}", duration);
        // Assert they are equal
        assert_eq!(cpu_root, gpu_root, "CUDA Merkle root does not match CPU root!");
    }

    #[test]
    fn test_poseidon2_permute() {
        // 1. Setup
        let perm = my_perm();
        
        let inputs = BabyBear::new_array([1984058442, 1779686813, 786767462, 334328488, 664932607, 1211726978, 653708563, 1908711429, 
                                        748182753, 1702519043, 182110445, 760024660, 807892063, 531542087, 1190845413, 1009472915]);

        let cpu_output = perm.permute(inputs);

        let cpu_expected = BabyBear::new_array([347216488, 1080055031, 427057322, 1709109579, 163565340, 1772928872, 652116498, 1067633747, 1731532717, 1424536438, 18136182, 1535395943, 1657769785, 1786735392, 1322137382, 35236760]);
        
         assert_eq!(cpu_output, cpu_expected); 

        println!("\n--- Testing Poseidon2 Permutation ---");

        println!("[CPU]   State after permute: {:?}", cpu_output);

        // 4. Compute permutation on GPU via FFI
        let mut gpu_state = inputs.clone();
        let result = unsafe {
            stark_test_poseidon2_permute_gpu(gpu_state.as_mut_ptr())
        };
        assert_eq!(result, 0, "CUDA test_poseidon2_permute function failed.");
        println!("[GPU]   State after permute: {:?}", gpu_state);
        
        // 5. Assert they are equal
        assert_eq!(cpu_output, gpu_state, "GPU permutation result does not match CPU reference!");
        println!("\nSUCCESS: device::poseidon2_permute_mut is correct.");
    }

    #[test]
    fn test_cuda_verify_tampered_proof_fails() {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        
        let gpu_mmcs = GpuValMmcs::new(hash, compress);

        // 1. Create test data: 4 matrices of 8x1 and 4 matrices of 8x2.
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let mut mats: Vec<RowMajorMatrix<BabyBear>> = (0..4)
            .map(|_| RowMajorMatrix::<BabyBear>::rand(&mut rng, 8, 1))
            .collect_vec();
        mats.extend((0..4).map(|_| RowMajorMatrix::<BabyBear>::rand(&mut rng, 8, 2)));

        let dimensions: Vec<Dimensions> = mats.iter().map(|m| m.dimensions()).collect();

        // 2. Commit to the matrices using our GPU implementation.
        println!("Committing with GpuMerkleTreeMmcs...");
        let (commit, prover_data) = gpu_mmcs.commit(mats);

        // 3. Open a proof for a specific index (e.g., index 3).
        println!("Opening batch for index 3...");
        let mut batch_opening = gpu_mmcs.open_batch(3, &prover_data);
        
        // 4. Tamper with the proof!
        // We add ONE to the first element of the first sibling digest in the proof.
        // This makes the proof invalid.
        println!("Tampering with the proof...");
        let original_sibling_value = batch_opening.opening_proof[0][0];
        batch_opening.opening_proof[0][0] += BabyBear::ONE;
        println!("Original sibling[0][0]: {:?}, Tampered sibling[0][0]: {:?}", 
                 original_sibling_value, batch_opening.opening_proof[0][0]);


        // 5. Verify the tampered proof and assert that it fails.
        println!("Verifying the tampered proof...");
        let verification_result = gpu_mmcs.verify_batch(
            &commit,
            &dimensions,
            3,
            (&batch_opening).into(), // Convert to BatchOpeningRef
        );
        
        // We expect an error, so `is_err()` should be true.
        assert!(
            verification_result.is_err(),
            "Verification of a tampered proof unexpectedly succeeded!"
        );
        
    }

     #[test]
    fn test_open_batch() {
       let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        
        // Create CPU and GPU versions of the MMCS
        let cpu_mmcs = p3_merkle_tree::MerkleTreeMmcs::<P, PW, H, C, DIGEST_SIZE>::new(hash.clone(), compress.clone());
        let gpu_mmcs = GpuValMmcs::new(hash, compress);

        // Create some test matrices
        let mut rng = Xoroshiro128Plus::seed_from_u64(42);
        let mat1 = RowMajorMatrix::<BabyBear>::rand(&mut rng, 16, 4);
        let mat2 = RowMajorMatrix::<BabyBear>::rand(&mut rng, 16, 8); // Different height
        let inputs = vec![mat1, mat2];

        // 2. Commit using both implementations.
        println!("--- Committing with both CPU and GPU implementations ---");
        let (cpu_root, cpu_prover_data) = cpu_mmcs.commit(inputs.clone());
        let (gpu_root, gpu_prover_data) = gpu_mmcs.commit(inputs.clone());
        
        // Sanity check: roots must match.
        assert_eq!(cpu_root, gpu_root, "Commitment roots do not match before opening.");

        // 3. Choose an index to open. Let's pick a non-trivial one.
        let index_to_open = 5;
        println!("\n--- Opening batch for index {} ---", index_to_open);

        // 4. Open using both CPU and GPU implementations.
        println!("Opening with CPU MMCS...");
        let cpu_opening = cpu_mmcs.open_batch(index_to_open, &cpu_prover_data);
        
        println!("Opening with GPU MMCS...");
        let gpu_opening = gpu_mmcs.open_batch(index_to_open, &gpu_prover_data);

        // 5. Compare the results.
        println!("\n--- Verifying open_batch results ---");
        
        // 5.1 Compare the opened values.
        println!("CPU opened_values: {:?}", cpu_opening.opened_values);
        println!("GPU opened_values: {:?}", gpu_opening.opened_values);
        assert_eq!(
            cpu_opening.opened_values,
            gpu_opening.opened_values,
            "Opened values do not match!"
        );
        println!("✅ Opened values match.");

        // 5.2 Compare the opening proof (sibling digests).
        println!("CPU opening_proof: {:?}", cpu_opening.opening_proof);
        println!("GPU opening_proof: {:?}", gpu_opening.opening_proof);
        assert_eq!(
            cpu_opening.opening_proof,
            gpu_opening.opening_proof,
            "Opening proofs (sibling paths) do not match!"
        );
        println!("✅ Opening proofs match.");

        println!("\nSUCCESS: GpuMerkleTreeMmcs::open_batch is correct.");
    }

    #[test]
    fn test_hash_leaves_kernel() {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        //let compress = MyCompress::new(perm.clone());
            
        let mut rng = Xoroshiro128Plus::seed_from_u64(42);

        // 1. Prepare test data: two matrices of the same height but different widths.
        let h = 16;
        let mat1 = RowMajorMatrix::<BabyBear>::rand(&mut rng, h, 2);
        let mat2 = RowMajorMatrix::<BabyBear>::rand(&mut rng, h, 4);
        let inputs = vec![&mat1, &mat2];
        let cpu_inputs = inputs.clone();

        // 2. Calculate digests on CPU, row by row.
        println!("\n--- Testing leaf hashing for {}x{} and {}x{} matrices ---", h, 2, h, 4);
        let mut cpu_digests = Vec::new();
        for r in 0..h {
            let elements_to_hash = cpu_inputs.iter()
                .flat_map(|m| m.row(r).unwrap());
            let digest = hash.hash_iter(elements_to_hash);
            cpu_digests.push(digest);
        }
        
        // 3. Prepare data and call the GPU FFI function.
        let mut flat_data = Vec::new();
        let mut matrix_info = Vec::new();
        for m in &inputs {
            let offset = flat_data.len();
            matrix_info.extend([offset as i32, m.height() as i32, m.width() as i32]);
            //flat_data.extend(m.clone().to_row_major_matrix().values);
             for r in 0..m.height() {
                // This does not move `m` and correctly flattens the data.
                flat_data.extend(m.row(r).unwrap());
            }
        }

        let mut gpu_digests_flat = vec![BabyBear::ZERO; h * DIGEST_SIZE];
        
        let result = unsafe {
            stark_test_hash_leaves_gpu(
                flat_data.as_ptr(),
                matrix_info.as_ptr(),
                inputs.len() as i32,
                h as i32,
                gpu_digests_flat.as_mut_ptr(),
            )
        };
        assert_eq!(result, 0, "CUDA test_hash_leaves_kernel function failed.");
        
        // 4. Compare results.
        let gpu_digests: Vec<[BabyBear; DIGEST_SIZE]> = gpu_digests_flat
            .chunks_exact(DIGEST_SIZE)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();
            
        for r in 0..h {
            assert_eq!(cpu_digests[r], gpu_digests[r], "Digest for row {} does not match!", r);
        }

        println!("\nSUCCESS: hash_leaves_kernel is correct.");
    }

}