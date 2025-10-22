extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{PackedValue, PrimeField};
use std::ffi::c_void;
use std::sync::Arc;

//use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::InnerChallenge;
use crate::baby_bear_poseidon2::{Challenge, Challenger, MyCompress, MyHash, Val};
use crate::gpu::ffi::*;
use crate::{DIGEST_SIZE, GpuChallengeMmcs, GpuMerkleProverData, GpuMerkleTreeHandle};
use p3_challenger::CanSampleBits;
use p3_fri::{CommitPhaseProofStep, FriConfig, FriProof, QueryProof};
use p3_matrix::Dimensions;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use std::collections::BTreeMap;
use std::ptr::null_mut;
use tracing::instrument; //debug_span,

//debug
/*
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions}; //debug
use std::io::Write;
use serde_json; */
//end

trait CudaResultCheck {
    fn check(self, msg: &str);
}
impl CudaResultCheck for i32 {
    #[inline]
    fn check(self, msg: &str) {
        if self != 0 {
            panic!("CUDA FFI Error: {} (code {})", msg, self);
        }
    }
}

// CommitPhaseResult is now concrete for the MMCS type
pub struct CommitPhaseResult {
    pub commits: Vec<<GpuChallengeMmcs as Mmcs<Challenge>>::Commitment>,
    pub data: Vec<<GpuChallengeMmcs as Mmcs<Challenge>>::ProverData<RowMajorMatrix<Challenge>>>,
    pub layer_evals: Vec<Vec<Challenge>>,
    pub final_poly: Challenge,
}

#[instrument(name = "FRI prover GPU", skip_all)]
pub fn prove_gpu(
    config: &FriConfig<GpuChallengeMmcs>,
    input: &[Option<Vec<Challenge>>; 32],
    challenger: &mut Challenger,
) -> (FriProof<Challenge, GpuChallengeMmcs, Val>, Vec<usize>)
where
    Val: PrimeField,
    <Val as Field>::Packing: PackedValue<Value = Val>,
    MyHash: CryptographicHasher<Val, [Val; DIGEST_SIZE]> + Sync + Clone,
    MyCompress: PseudoCompressionFunction<[Val; DIGEST_SIZE], 2> + Sync + Clone,
    // 2. Bounds for the `Challenger`
    Challenger: FieldChallenger<Val>
        + GrindingChallenger<Witness = Val>
        + CanObserve<<GpuChallengeMmcs as Mmcs<Challenge>>::Commitment>,
{
    let log_max_height = input.iter().rposition(Option::is_some).unwrap();

    // 1. Commit Phase
    let commit_phase_result = commit_phase_gpu(config, input, log_max_height, challenger);
    /*let serialized = serde_json::to_vec(&commit_phase_result.commits.clone()).expect("Serialization failed");
    File::create("gpu_commits.json")
    .and_then(|mut f| f.write_all(&serialized))
    .expect("Failed to write gpu_commits to file");*/

    // 2. Sample Query Indices
    let pow_witness = challenger.grind(config.proof_of_work_bits);
    let query_indices: Vec<usize> =
        (0..config.num_queries).map(|_| challenger.sample_bits(log_max_height)).collect();
    //println!("--gpu_prove, proof_of_work_bits={}, pow_witness={:?}, query_indices={:?}", config.proof_of_work_bits, pow_witness, query_indices);

    // 3. Query Phase
    let query_proofs = answer_queries_gpu(
        //&config.mmcs,
        &commit_phase_result.data,
        &commit_phase_result.layer_evals,
        &query_indices,
    );

    //debug
    /* let serialized = serde_json::to_vec(&query_proofs.clone()).expect("Serialization failed");
    File::create("gpu_query_proofs.json")
    .and_then(|mut f| f.write_all(&serialized))
    .expect("Failed to write query_proofs to file");*/

    // 4. Assemble Final Proof
    (
        FriProof {
            commit_phase_commits: commit_phase_result.commits,
            query_proofs,
            final_poly: commit_phase_result.final_poly,
            pow_witness,
        },
        query_indices,
    )
}

#[instrument(name = "commit phase GPU", skip_all)]
pub fn commit_phase_gpu<Challenger>(
    config: &FriConfig<GpuChallengeMmcs>,
    input: &[Option<Vec<Challenge>>; 32],
    log_max_height: usize,
    challenger: &mut Challenger,
) -> CommitPhaseResult
where
    Challenger:
        FieldChallenger<Val> + CanObserve<<GpuChallengeMmcs as Mmcs<Challenge>>::Commitment>,
{
    let initial_evals = input[log_max_height].as_ref().unwrap().clone();
    // let mut current = input[log_max_height].as_ref().unwrap().clone();
    let mut d_current_evals: *mut c_void = std::ptr::null_mut();
    let initial_size_bytes = initial_evals.len() * std::mem::size_of::<Challenge>();

    unsafe {
        cuda_malloc(&mut d_current_evals, initial_size_bytes)
            .check("Initial commit_phase cuda_malloc failed");
        cuda_memcpy_htod(
            d_current_evals,
            initial_evals.as_ptr() as *const c_void,
            initial_size_bytes,
        )
        .check("Initial commit_phase cuda_memcpy_htod failed");
    }
    let mut current_num_evals = initial_evals.len();

    let mut commits = vec![];
    let mut data = vec![];
    let mut layer_evals: Vec<Vec<Challenge>> = vec![];
    layer_evals.push(initial_evals.clone());

    for log_folded_height in (config.log_blowup..log_max_height).rev() {
        if d_current_evals.is_null() {
            break;
        }
        let mut root_out_buffer = [Val::ZERO; DIGEST_SIZE];
        let mut prover_data_handle: *mut c_void = null_mut();

        unsafe {
            fri_commit_on_gpu(
                d_current_evals as *const InnerChallenge,
                current_num_evals as i32,
                root_out_buffer.as_mut_ptr(),
                &mut prover_data_handle,
            )
            .check("fri_commit_on_gpu failed");
        }

        let commitment: <GpuChallengeMmcs as Mmcs<Challenge>>::Commitment = root_out_buffer.into();
        challenger.observe(commitment.clone());

        commits.push(commitment);

        let prover_data_for_layer: GpuMerkleProverData<
            p3_matrix::extension::FlatMatrixView<Val, Challenge, RowMajorMatrix<Challenge>>,
        > = GpuMerkleProverData {
            handle: Arc::new(GpuMerkleTreeHandle(prover_data_handle)),
            dimensions: vec![Dimensions { height: current_num_evals / 2, width: 8 }], //be consistent with the function fri_commit_on_gpu
            sorted_inputs: vec![],
            original_to_sorted_indices: vec![],
            _phantom: std::marker::PhantomData,
        };

        let final_prover_data = unsafe { std::mem::transmute(prover_data_for_layer) };
        data.push(final_prover_data);

        if current_num_evals == 1 {
            break;
        }

        let beta: Challenge = challenger.sample_algebra_element();

        // let g_inv = Challenge::two_adic_generator(log_folded_height + 2).inverse();
        let g_inv = Challenge::two_adic_generator(log2_strict_usize(current_num_evals)).inverse();
        let one_half = Challenge::TWO.inverse();
        let half_beta = beta * one_half;
        let mut powers = g_inv.shifted_powers(half_beta).take(current_num_evals / 2).collect_vec();
        reverse_slice_index_bits(&mut powers);

        let mut d_next_evals: *mut c_void = std::ptr::null_mut();
        unsafe {
            fri_fold_on_gpu(
                d_current_evals,
                current_num_evals as i32,
                beta,
                powers.as_ptr() as *const InnerChallenge,
                &mut d_next_evals,
            )
            .check("fri_fold_on_gpu failed");
        }

        unsafe {
            cuda_free(d_current_evals).check("cuda_free of old layer failed");
        }
        d_current_evals = d_next_evals;
        current_num_evals /= 2;

        //Following  plonk3 commit_phase() injection:
        //if let Some(v) = &input[log_folded_height] {
        //    current.iter_mut().zip_eq(v).for_each(|(c, v)| *c += *v);
        //}
        if let Some(v) = &input[log_folded_height] {
            if !v.is_empty() {
                assert_eq!(v.len(), current_num_evals, "Injection vector length mismatch");
                unsafe {
                    fri_injection_gpu(
                        d_current_evals,
                        v.as_ptr() as *const InnerChallenge,
                        v.len() as i32,
                    )
                    .check("fri_injection_gpu failed");
                }
            }
        }

        if current_num_evals > 0 {
            let mut h_next_evals = vec![Challenge::ZERO; current_num_evals];
            unsafe {
                cuda_memcpy_dtoh(
                    h_next_evals.as_mut_ptr() as *mut c_void,
                    d_next_evals as *const c_void,
                    current_num_evals * std::mem::size_of::<Challenge>(),
                )
                .check("DtoH copy for layer evals failed");
            }

            layer_evals.push(h_next_evals);
        } else {
            layer_evals.push(Vec::new());
        }
    }
    //end folding

    let mut final_poly_vec = vec![Challenge::ZERO; current_num_evals];
    if current_num_evals > 0 {
        unsafe {
            cuda_memcpy_dtoh(
                final_poly_vec.as_mut_ptr() as *mut c_void,
                d_current_evals as *const c_void,
                current_num_evals * std::mem::size_of::<Challenge>(),
            )
            .check("Final cuda_memcpy_dtoh failed");
        }
    }
    let final_poly = final_poly_vec.get(0).copied().unwrap_or_else(|| Challenge::ZERO);
    //println!("gpu--final_poly:{:?}", final_poly);
    if !d_current_evals.is_null() {
        unsafe {
            cuda_free(d_current_evals).check("Final cuda_free failed");
        }
    }

    challenger.observe_algebra_element(final_poly);

    CommitPhaseResult { commits, data, layer_evals, final_poly }
}

#[instrument(name = "query phase GPU", skip_all)]
pub fn answer_queries_gpu(
    //mmcs: &GpuChallengeMmcs,
    commit_phase_data: &[<GpuChallengeMmcs as Mmcs<Challenge>>::ProverData<
        RowMajorMatrix<Challenge>,
    >],
    commit_phase_layer_evals: &[Vec<Challenge>],
    query_indices: &[usize],
) -> Vec<QueryProof<Challenge, GpuChallengeMmcs>> {
    if query_indices.is_empty() {
        return Vec::new();
    }

    // 1. Collect all UNIQUE queries for each layer.
    // Maps layer_index to a map of `index_pair -> original_query_indices`
    let mut queries_by_layer: BTreeMap<usize, BTreeMap<usize, Vec<usize>>> = BTreeMap::new();
    for &index in query_indices {
        for layer_idx in 0..commit_phase_data.len() {
            let index_i = index >> layer_idx;
            let index_pair = index_i >> 1;

            queries_by_layer
                .entry(layer_idx)
                .or_default()
                .entry(index_pair)
                .or_default()
                .push(index);
        }
    }

    // 2. Prepare data for a single, large FFI call.
    let mut prover_data_handles: Vec<*const c_void> = Vec::new();
    let mut flat_indices: Vec<u32> = Vec::new();
    let mut offsets: Vec<u32> = vec![0];
    let mut layer_map: BTreeMap<usize, (usize, usize)> = BTreeMap::new();

    for (layer_idx, pairs) in &queries_by_layer {
        let data = &commit_phase_data[*layer_idx];

        let cuda_prover_data = unsafe { &*(data as *const _ as *const GpuMerkleProverData<()>) };
        prover_data_handles.push(cuda_prover_data.handle.0);

        layer_map.insert(*layer_idx, (flat_indices.len(), pairs.len()));
        flat_indices.extend(pairs.keys().map(|&k| k as u32));
        offsets.push(flat_indices.len() as u32);
    }

    // 3. Allocate output buffer and make the FFI call.
    let total_proof_elements: usize = queries_by_layer
        .iter()
        .map(|(layer_idx, pairs)| {
            let log_height = log2_strict_usize(commit_phase_layer_evals[*layer_idx].len()) - 1;
            //println!("---gpu-rust, len1={}, log_height={}, pairs.len()={}", commit_phase_layer_evals[*layer_idx].len(), log_height, pairs.len());

            pairs.len() * log_height * DIGEST_SIZE
        })
        .sum();

    let mut flat_proofs_host_buffer = vec![Val::ZERO; total_proof_elements];
    //println!("---gpu-rust,  total_proof_elements={}",  total_proof_elements);
    let result = unsafe {
        stark_fri_generate_proofs_gpu(
            prover_data_handles.as_ptr(),
            flat_indices.as_ptr(),
            offsets.as_ptr(),
            prover_data_handles.len() as i32,
            flat_indices.len() as i32,
            DIGEST_SIZE as i32,
            flat_proofs_host_buffer.as_mut_ptr() as *mut BabyBear,
        )
    };
    if result != 0 {
        panic!("stark_fri_generate_proofs_gpu FFI call failed");
    }

    // 4. Reconstruct results. The proofs are now all in `flat_proofs_host_buffer`.
    // We can create a map from `(layer_idx, index_pair)` to the proof for fast lookup.
    //let mut proof_map: BTreeMap<(usize, usize), GpuChallengeMmcs::Proof> = BTreeMap::new();
    let mut proof_map: BTreeMap<(usize, usize), <GpuChallengeMmcs as Mmcs<Challenge>>::Proof> =
        BTreeMap::new();

    let mut proof_cursor = 0;
    for (layer_idx, (offset, count)) in &layer_map {
        //let log_height = log2_strict_usize(commit_phase_layer_evals[*layer_idx].len());
        let log_height = log2_strict_usize(commit_phase_layer_evals[*layer_idx].len()) - 1;
        let proof_size = log_height * DIGEST_SIZE;
        let indices_for_layer = &flat_indices[*offset..*offset + *count];
        for &index_pair in indices_for_layer {
            let proof_slice = &flat_proofs_host_buffer[proof_cursor..proof_cursor + proof_size];
            //let siblings : Vec<[F; DIGEST_SIZE]> = proof_slice.chunks_exact(DIGEST_SIZE).map(|c| c.try_into().unwrap()).collect();
            //proof_map.insert((*layer_idx, index_pair as usize), siblings.into());

            let siblings: <GpuChallengeMmcs as Mmcs<Challenge>>::Proof = proof_slice
                .chunks_exact(DIGEST_SIZE)
                .map(|chunk| chunk.try_into().unwrap())
                .collect();
            // ======================================================

            proof_map.insert((*layer_idx, index_pair as usize), siblings);
            proof_cursor += proof_size;
        }
    }

    // 5. Final assembly of QueryProof structs. This is now a fast, in-memory operation.
    query_indices
        .iter()
        .map(|&index| {
            let commit_phase_openings = commit_phase_data
                .iter()
                .zip(commit_phase_layer_evals)
                .enumerate()
                .map(|(layer_idx, (_data, evals))| {
                    let index_i = index >> layer_idx;
                    let index_i_sibling = index_i ^ 1;
                    let index_pair = index_i >> 1;
                    //println!("-----gpu_answer_query2222: index:{}, index_i:{},index_i_sibling:{} ,index_pair={}", index, index_i, index_i_sibling, index_pair);

                    //let sibling_value = evals[index_in_layer ^ 1];
                    let sibling_value = evals[index_i_sibling];
                    let opening_proof = proof_map.get(&(layer_idx, index_pair)).unwrap().clone();
                    //println!("***gpu opening_proof:{:?}", opening_proof);
                    //println!("***gpu sibling_value:{:?}", sibling_value);
                    CommitPhaseProofStep { sibling_value, opening_proof }
                })
                .collect();
            QueryProof { commit_phase_openings }
        })
        .collect()
}

//test
#[instrument(skip_all, level = "debug")]
pub fn fold_even_odd<F: TwoAdicField>(poly: Vec<F>, beta: F) -> Vec<F>
where
    F: Into<InnerChallenge> + From<InnerChallenge>, // Ensure conversion is possible
{
    if poly.is_empty() {
        return Vec::new();
    }

    // 1. Calculate the `powers` vector on the CPU, just like the original function.
    let n = poly.len();
    let g_inv = F::two_adic_generator(log2_strict_usize(n)).inverse();
    let one_half = F::TWO.inverse();
    let half_beta = beta * one_half;

    let mut powers = g_inv.shifted_powers(half_beta).take(n / 2).collect_vec();
    reverse_slice_index_bits(&mut powers);

    // 2. Prepare buffers for the FFI call.
    let mut next_evals = vec![F::ZERO; n / 2];

    // 3. Call the FFI function.
    let result = unsafe {
        // We need to cast our generic vectors to the concrete types the FFI expects.
        fold_even_odd_gpu(
            poly.as_ptr() as *const InnerChallenge,
            poly.len() as i32,
            beta.into(),
            powers.as_ptr() as *const InnerChallenge,
            next_evals.as_mut_ptr() as *mut InnerChallenge,
        )
    };
    if result != 0 {
        panic!("fold_even_odd_gpu FFI call failed");
    }

    // 4. Return the result from the FFI.
    next_evals
}

#[cfg(test)]
mod tests {
    use super::*; // Import items from the parent module (prover.rs)
    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    // We need to import the original plonky3 function to compare against.
    // The path might need adjustment depending on your `use` statements.
    use crate::baby_bear_poseidon2::{MyCompress, MyHash, my_perm};
    use crate::gpu::merkle::*;
    use crate::{GpuChallengeMmcs, StarkGenericConfig};
    use p3_commit::ExtensionMmcs;
    use p3_fri::fold_even_odd as fold_even_odd_cpu;
    use p3_merkle_tree::MerkleTreeMmcs;
    use std::ffi::c_void;
    use std::io::BufReader;
    use std::path::Path;

    use serde::{Deserialize, Serialize};
    use serde_json;
    use std::fs::{File, OpenOptions}; //debug
    use std::io::Write;

    // Define the concrete Challenge type for the test
    type Challenge = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_fold_even_odd_gpu() {
        // 1. Setup: Generate random test data
        let mut rng = Xoroshiro128Plus::seed_from_u64(1337); // Use a fixed seed

        // Use a reasonably large vector size to test performance and correctness
        //const LOG_N: usize = 10;
        //const N: usize = 1 << LOG_N;
        let N: usize = 1048576;
        let poly: Vec<Challenge> = (0..N).map(|_| rng.r#gen()).collect();
        let beta: Challenge = rng.r#gen();

        println!("Testing fold_even_odd_gpu with a vector of length {}", N);
        println!("beta: {:?}", beta);

        // 2. Compute the EXPECTED result using the original CPU implementation
        println!("Running original CPU fold_even_odd...");
        let start_cpu = std::time::Instant::now();
        let expected_result = fold_even_odd_cpu(poly.clone(), beta);
        let duration_cpu = start_cpu.elapsed();
        println!("CPU version finished in {:?}", duration_cpu);

        // 3. Compute the ACTUAL result using your new GPU-accelerated wrapper function
        println!("Running new GPU-accelerated fold_even_odd_gpu...");
        let start_gpu = std::time::Instant::now();
        let actual_result = fold_even_odd(poly, beta);
        let duration_gpu = start_gpu.elapsed();
        println!("GPU version finished in {:?}", duration_gpu);

        // 4. Compare the results
        assert_eq!(
            expected_result.len(),
            actual_result.len(),
            "Result vectors have different lengths!"
        );

        // For large vectors, asserting the whole vector can produce a huge output on failure.
        // It's often better to find the first differing element.
        let mut mismatch_found = false;
        for i in 0..expected_result.len() {
            if expected_result[i] != actual_result[i] {
                println!("\n==================== MISMATCH FOUND ====================");
                println!("Mismatch at index {}", i);
                println!("  Expected (CPU): {:?}", expected_result[i]);
                println!("  Actual (GPU):   {:?}", actual_result[i]);
                println!("========================================================");
                mismatch_found = true;
                break; // Stop at the first mismatch
            }
        }

        if mismatch_found {
            // Use a standard assert_eq to trigger the test failure with a diff
            assert_eq!(expected_result, actual_result, "The full vectors do not match.");
        } else {
            println!("\nSUCCESS: fold_even_odd_gpu produces the same result as the CPU version.");
        }
    }

    // Redefine types for testing clarity, matching your existing tests.
    type F = BabyBear;
    type Cfg = crate::baby_bear_poseidon2::StarkConfigCpu;
    //type Challenge = <Cfg as StarkGenericConfig>::Challenge;
    type ValMmcs<F> = p3_merkle_tree::MerkleTreeMmcs<F, F, MyHash, MyCompress, 8>;
    type FriMmcs<F, EF> = ExtensionMmcs<F, EF, ValMmcs<F>>;

    #[test]
    fn test_fri_commit_on_gpu() {
        // 1. SETUP
        // Create a random number generator
        let mut rng = Xoroshiro128Plus::seed_from_u64(2024);

        // Create the layer evaluations vector. This is what `commit_phase_gpu` receives.
        //const LOG_N: usize = 10;
        //const N: usize = 1 << LOG_N;
        let N: usize = 1048576;
        let evals: Vec<Challenge> = (0..N).map(|_| rng.r#gen()).collect();

        // Create a CPU-based MMCS to compute the expected "golden" root.
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::<F>::new(hash, compress);
        let cpu_fri_mmcs = FriMmcs::<F, Challenge>::new(val_mmcs);

        println!("Testing fri_commit_on_gpu with a vector of {} extension field elements.", N);

        // 2. COMPUTE ON CPU (The ground truth)
        println!("Computing commitment on CPU...");
        let start_cpu = std::time::Instant::now();
        // `ExtensionMmcs` expects a `Matrix`. We treat our `evals` Vec as a single-column matrix.
        let (cpu_root, _cpu_prover_data) =
            cpu_fri_mmcs.commit_matrix(RowMajorMatrix::new(evals.clone(), 2));
        let duration_cpu = start_cpu.elapsed();
        println!("CPU commit finished in {:?}", duration_cpu);

        // 3. COMPUTE ON GPU (The implementation under test)
        println!("Computing commitment on GPU via FFI...");
        let start_gpu = std::time::Instant::now();

        // 3a. Copy input data from Host to Device
        let mut d_evals: *mut c_void = std::ptr::null_mut();
        let size_bytes = evals.len() * std::mem::size_of::<Challenge>();
        unsafe {
            cuda_malloc(&mut d_evals, size_bytes).check("Test: cuda_malloc failed");
            cuda_memcpy_htod(d_evals, evals.as_ptr() as *const c_void, size_bytes)
                .check("Test: cuda_memcpy_htod failed");
        }

        // 3b. Prepare output buffers on the Host
        let mut gpu_root_buffer = [BabyBear::ZERO; DIGEST_SIZE];
        let mut gpu_handle: *mut c_void = std::ptr::null_mut();

        // 3c. Call the FFI function
        let result = unsafe {
            fri_commit_on_gpu(
                d_evals as *const Challenge,
                evals.len() as i32,
                gpu_root_buffer.as_mut_ptr(),
                &mut gpu_handle,
            )
        };
        if result != 0 {
            // Cleanup GPU memory before panicking
            unsafe {
                cuda_free(d_evals).check("Test: cuda_free (on error) failed");
            }
            panic!("FFI call fri_commit_on_gpu failed with code: {}", result);
        }

        // 3d. Convert the output buffer to the correct Commitment type.
        //let gpu_root: Hash<BabyBear, BabyBear, DIGEST_SIZE> = gpu_root_buffer.into();
        let gpu_root: <GpuChallengeMmcs as Mmcs<Challenge>>::Commitment = gpu_root_buffer.into();

        let duration_gpu = start_gpu.elapsed();
        println!("GPU commit finished in {:?}", duration_gpu);

        // 3e. IMPORTANT: Clean up all GPU memory
        unsafe {
            cuda_free(d_evals).check("Test: cuda_free for evals failed");
            stark_merkle_free_gpu(gpu_handle); //.check("Test: stark_merkle_free_gpu for handle failed");
        }

        // 4. COMPARE RESULTS
        println!("\n--- Comparing Roots ---");
        println!("CPU Root: {:?}", cpu_root);
        println!("GPU Root: {:?}", gpu_root);

        assert_eq!(
            cpu_root, gpu_root,
            "The GPU-computed Merkle root for the FRI layer does not match the CPU reference!"
        );

        println!("\nSUCCESS: fri_commit_on_gpu is correct.");
    }

    #[test]
    fn test_fri_query_gpu() {
        // 1. SETUP
        // Create a random number generator
        let mut rng = Xoroshiro128Plus::seed_from_u64(2024);

        // Create the layer evaluations vector. This is what `commit_phase_gpu` receives.
        //const LOG_N: usize = 10;
        //const N: usize = 1 << LOG_N;
        let N: usize = 32;
        let index: usize = 15;
        let evals: Vec<Challenge> = (0..N).map(|_| rng.r#gen()).collect();

        //println!("----Matrix-data: {:?}", evals);
        //println!("----Matrix-data: [0]:{:?},  [1]:{:?}", evals[0], evals[1]);

        //println!("----Matrix-data: [2]:{:?},  [3]:{:?}", evals[2], evals[3]);
        //println!("----Matrix-data: [4]:{:?},  [5]:{:?}", evals[4], evals[5]);

        // Create a CPU-based MMCS to compute the expected "golden" root.
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::<F>::new(hash, compress);
        let cpu_fri_mmcs = FriMmcs::<F, Challenge>::new(val_mmcs);

        println!("Testing fri_commit_on_gpu with a vector of {} extension field elements.", N);

        // 2. COMPUTE ON CPU (The ground truth)
        println!("Computing commitment on CPU...");
        let start_cpu = std::time::Instant::now();
        // `ExtensionMmcs` expects a `Matrix`. We treat our `evals` Vec as a single-column matrix.
        let (cpu_root, cpu_prover_data) =
            cpu_fri_mmcs.commit_matrix(RowMajorMatrix::new(evals.clone(), 2));
        let duration_cpu = start_cpu.elapsed();
        println!("CPU commit finished in {:?}", duration_cpu);

        // 3. COMPUTE ON GPU (The implementation under test)
        println!("Computing commitment on GPU via FFI...");
        let start_gpu = std::time::Instant::now();

        // 3a. Copy input data from Host to Device
        let mut d_evals: *mut c_void = std::ptr::null_mut();
        let size_bytes = evals.len() * std::mem::size_of::<Challenge>();
        unsafe {
            cuda_malloc(&mut d_evals, size_bytes).check("Test: cuda_malloc failed");
            cuda_memcpy_htod(d_evals, evals.as_ptr() as *const c_void, size_bytes)
                .check("Test: cuda_memcpy_htod failed");
        }

        // 3b. Prepare output buffers on the Host
        let mut gpu_root_buffer = [BabyBear::ZERO; DIGEST_SIZE];
        let mut gpu_handle: *mut c_void = std::ptr::null_mut();

        // 3c. Call the FFI function
        let result = unsafe {
            fri_commit_on_gpu(
                d_evals as *const Challenge,
                evals.len() as i32,
                gpu_root_buffer.as_mut_ptr(),
                &mut gpu_handle,
            )
        };
        if result != 0 {
            // Cleanup GPU memory before panicking
            unsafe {
                cuda_free(d_evals).check("Test: cuda_free (on error) failed");
            }
            panic!("FFI call fri_commit_on_gpu failed with code: {}", result);
        }

        // 3d. Convert the output buffer to the correct Commitment type.
        //let gpu_root: Hash<BabyBear, BabyBear, DIGEST_SIZE> = gpu_root_buffer.into();
        let gpu_root: <GpuChallengeMmcs as Mmcs<Challenge>>::Commitment = gpu_root_buffer.into();

        let duration_gpu = start_gpu.elapsed();
        println!("GPU commit finished in {:?}", duration_gpu);

        // 4. COMPARE RESULTS
        println!("\n--- Comparing Roots ---");
        //println!("CPU Root: {:?}", cpu_root);
        //println!("GPU Root: {:?}", gpu_root);

        assert_eq!(
            cpu_root, gpu_root,
            "The GPU-computed Merkle root for the FRI layer does not match the CPU reference!"
        );

        let i = 0;
        let index_i = index >> i;
        let index_i_sibling = index_i ^ 1;
        let index_pair = index_i >> 1;
        println!(
            "*****cpu_answer_query: index:{}, index_i_sibling:{},index_pair:{} ",
            index, index_i_sibling, index_pair
        );
        let (mut opened_rows, opening_proof) =
            cpu_fri_mmcs.open_batch(index_pair, &cpu_prover_data).unpack();
        assert_eq!(opened_rows.len(), 1);
        let opened_row = &opened_rows.pop().unwrap();
        assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
        let sibling_value = opened_row[index_i_sibling % 2];
        //println!("***cpu opening_proof:{:?}", opening_proof);
        println!("***cpu sibling_value:{:?}", sibling_value);

        println!("================ CPU query end------------");

        //GPU query
        let prover_data_for_layer: GpuMerkleProverData<
            p3_matrix::extension::FlatMatrixView<Val, Challenge, RowMajorMatrix<Challenge>>,
        > = GpuMerkleProverData {
            handle: Arc::new(GpuMerkleTreeHandle(gpu_handle)),
            dimensions: vec![],                 //not use
            sorted_inputs: vec![],              //not use
            original_to_sorted_indices: vec![], //not use
            _phantom: std::marker::PhantomData,
        };

        println!("*********** GPU query------------------");
        let single_query_index = index; //as same as cpu's
        //let log_max_height = 3;
        //let log_max_height = log2_strict_usize(evals.len()) -1;
        let log_max_height = evals.len() / 2;

        let query_one = vec![single_query_index];
        let mut data = vec![];
        let mut layer_evals: Vec<Vec<Challenge>> = vec![];
        data.push(prover_data_for_layer);
        layer_evals.push(evals);

        let query_proofs = answer_queries_gpu(
            &data,
            &layer_evals,
            &query_one,
            //log_max_height,
        );

        // 3e. IMPORTANT: Clean up all GPU memory
        unsafe {
            cuda_free(d_evals).check("Test: cuda_free for evals failed");
        }

        assert_eq!(
            opening_proof, query_proofs[0].commit_phase_openings[0].opening_proof,
            "The GPU opening_proof not match the CPU !"
        );
        assert_eq!(
            sibling_value, query_proofs[0].commit_phase_openings[0].sibling_value,
            "The GPU sibling_value not match the CPU !"
        );
    }

    //type Challenge = BinomialExtensionField<BabyBear, 4>;

    mod challenge_serde {
        use super::{BabyBear, Challenge};
        use p3_field::BasedVectorSpace;
        use p3_field::PackedFieldExtension;
        use p3_field::PackedValue;
        use p3_field::PrimeCharacteristicRing;
        use p3_field::PrimeField32;
        use serde::Serialize;
        use serde::{self, Deserialize, Deserializer, Serializer};

        pub fn serialize<S>(challenge: &Challenge, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let coeffs: [u32; 4] = challenge
                .as_basis_coefficients_slice()
                .iter()
                .map(|b: &BabyBear| b.as_canonical_u32())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            coeffs.serialize(serializer)
        }

        pub fn deserialize<'de, D>(deserializer: D) -> Result<Challenge, D::Error>
        where
            D: Deserializer<'de>,
        {
            let coeffs_u32 = <[u32; 4]>::deserialize(deserializer)?;
            let coeffs_bb = coeffs_u32
                .iter()
                .map(|&x| BabyBear::from_u32(x))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            Ok(Challenge::new(coeffs_bb))
        }
    }

    #[derive(serde::Deserialize, Debug)]
    struct FieldElementWrapper {
        #[serde(with = "challenge_serde")]
        value: Challenge,
        _phantom: serde_json::Value,
    }

    /// Reads a vector of `Challenge` elements from a JSON file.
    /// The JSON file is expected to be an array of `FieldElementWrapper` objects.
    pub fn read_challenges_from_file<P: AsRef<Path>>(path: P) -> Vec<Challenge> {
        let file = File::open(path).expect("Failed to open file");
        let reader = BufReader::new(file);
        let wrappers: Vec<FieldElementWrapper> = serde_json::from_reader(reader)
            .expect("Failed to deserialize JSON into Vec<FieldElementWrapper>");

        // Extract the `value` field from each wrapper
        wrappers.into_iter().map(|w| w.value).collect()
    }
    #[test]
    fn test_first_fri_commit_consistency() {
        // 1. Load the EXACT input that `commit_phase_gpu` uses from the file.
        let initial_evals = read_challenges_from_file("gpu_current_initial.json");
        assert!(!initial_evals.is_empty(), "Input file is empty!");

        println!("Loaded {} elements from gpu_current_initial.json", initial_evals.len());

        // 2. Compute the expected root on the CPU using these evals.
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::<F>::new(hash, compress);
        let cpu_mmcs = FriMmcs::<F, Challenge>::new(val_mmcs);

        let (expected_root, _) =
            cpu_mmcs.commit_matrix(RowMajorMatrix::new(initial_evals.clone(), 2));
        println!("Expected (CPU) root: {:?}", expected_root);

        // 3. Compute the actual root on the GPU, mimicking `commit_phase_gpu`.
        let mut d_current_evals: *mut c_void = std::ptr::null_mut();
        let size_bytes = initial_evals.len() * std::mem::size_of::<Challenge>();

        let mut gpu_root_buffer = [Val::ZERO; DIGEST_SIZE];
        let mut gpu_handle: *mut c_void = std::ptr::null_mut();

        unsafe {
            // Copy data to GPU
            //cuda_malloc(&mut d_current_evals, size_bytes).unwrap();
            //cuda_memcpy_htod(d_current_evals, initial_evals.as_ptr() as *const c_void, size_bytes).unwrap();

            cuda_malloc(&mut d_current_evals, size_bytes).check("Test: cuda_malloc failed");
            cuda_memcpy_htod(d_current_evals, initial_evals.as_ptr() as *const c_void, size_bytes)
                .check("Test: cuda_memcpy_htod failed");
            // Call the FFI function
            fri_commit_on_gpu(
                d_current_evals as *const InnerChallenge,
                initial_evals.len() as i32,
                gpu_root_buffer.as_mut_ptr(),
                &mut gpu_handle,
            )
            .check("fri_commit_on_gpu failed");

            // Cleanup
            stark_merkle_free_gpu(gpu_handle); //.check("free handle failed");
            cuda_free(d_current_evals).check("free evals failed");
        }

        let actual_root: <GpuChallengeMmcs as Mmcs<Challenge>>::Commitment = gpu_root_buffer.into();
        println!("Actual (GPU) root:   {:?}", actual_root);

        // 4. Compare the results!
        assert_eq!(expected_root, actual_root);
    }
}
