use p3_commit::{Pcs, Mmcs,  OpenedValues};

use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, 
    coset::TwoAdicMultiplicativeCoset, ExtensionField, Field,
     TwoAdicField, PrimeCharacteristicRing,  
};
use p3_baby_bear::{BabyBear,  };
use p3_fri::{TwoAdicFriPcs, FriConfig, TwoAdicFriPcsProof, };
use p3_interpolation::interpolate_coset_with_precomputation;
use p3_matrix::{
    bitrev::{ BitReversedMatrixView},
    dense::{RowMajorMatrix, RowMajorMatrixView, },
    Matrix, 
};

use p3_challenger::{FieldChallenger, };
use p3_util::{log2_strict_usize, reverse_slice_index_bits, linear_map::LinearMap};

use itertools::{izip, Itertools};
use std::collections::BTreeMap;
use crate::gpu::merkle::{BatchedQueries, BatchedOpenings, 
                            BatchOpenableMmcs};
use crate::gpu::ffi::*;
use crate::gpu::gpu_prover::*;

// Import SP1 specific types
use crate::{InnerChallenge,  GpuChallengeMmcs, GpuValMmcs,GpuPcs, InnerDft,};
use crate::baby_bear_poseidon2::{Val, Challenge,Challenger};

#[cfg(feature = "recursion_cuda")]
use crate::GpuDft;

//debug
/*use serde::{Deserialize, Serialize};
use std::fs::File; //debug
use std::io::Write;
use serde_json; 
use std::fs::OpenOptions;*/

#[repr(C)]
pub struct OpeningPointInfo {
    points_z: *const InnerChallenge,
    opened_values_y: *const InnerChallenge,
    num_points: i32,
}

struct FfiMatrixData<'a, Val, Challenge> 
where
    Val: p3_field::Field,
    Challenge: p3_field::Field,
{
   // global_index: usize,
    mat: RowMajorMatrixView<'a, Val>,
    points_for_mat: &'a Vec<Challenge>,
    openings_for_mat: &'a Vec<Vec<Challenge>>,
}

/// A GPU-accelerated implementation of the Pcs trait, wrapping plonky3's TwoAdicFriPcs.
/// It delegates most methods to the inner PCS, but provides custom, GPU-accelerated
/// implementations for performance-critical methods like `open`, `mmc commit`.
#[derive(Debug)]
pub struct GpuFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub inner_pcs: TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>,
}

impl<Val, Dft, InputMmcs, FriMmcs> GpuFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub fn new(dft: Dft, mmcs: InputMmcs, fri_config: FriConfig<FriMmcs>) -> Self {
        Self {
            inner_pcs: TwoAdicFriPcs::new(dft, mmcs,fri_config),
        }
    }
}

impl Pcs<Challenge, Challenger> for GpuPcs {
    
    // All associated types are defined using concrete types.
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::Commitment;
    type ProverData = <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::ProverData;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixView<'a, Val>>;
    type Proof = <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::Proof;
    type Error = <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::Error;
    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &self.inner_pcs,
            degree,
        )
    }

    #[cfg(feature = "recursion_cuda")]
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let lde_requests: Vec<_> = evaluations
                .into_iter()
                .map(|(domain, evals)| {
                    let shift = Val::GENERATOR / domain.shift();
                    (evals, self.inner_pcs.fri_config().log_blowup, shift)
                })
                .collect();
    
        // 2. Perform batched LDE, with results staying on the GPU.
        //let lde_gpu_handle = self.inner_pcs.dft.batch_coset_lde_on_gpu(lde_requests);
        let lde_gpu_handle = GpuDft::batch_coset_lde_on_gpu(&self.inner_pcs.dft, lde_requests);
        
        // 3. Download the LDE results to CPU *once* for ProverData storage.
        //let ldes_for_prover_data = self.inner_pcs.dft.download_lde_batch(&lde_gpu_handle);
        let ldes_for_prover_data = GpuDft::download_lde_batch(&self.inner_pcs.dft, &lde_gpu_handle);

        // 4. Commit to the data that is already on the GPU.
        // This is the key step that avoids the Host-to-Device copy.
        self.inner_pcs.mmcs.commit_on_gpu(lde_gpu_handle, ldes_for_prover_data)

    }

    #[cfg(not(feature = "recursion_cuda"))]
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::commit(
            &self.inner_pcs,
            evaluations)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::get_evaluations_on_domain(
            &self.inner_pcs,
            prover_data, idx, domain)
    }

    fn open(
        &self,
        rounds: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        
        //debug
        let start = std::time::Instant::now();
        println!("INFO: Using  GPU-accelerated `open` method!, rounds.len:{}", rounds.len());
        // This entire function body is a carefully adapted version of the original
        // `TwoAdicFriPcs::open` method.

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                let mats = self.inner_pcs
                    .mmcs
                    .get_matrices::<RowMajorMatrix<Val>>(*data)
                    .into_iter()
                    .map(|m| m.as_view())
                    .collect_vec();
                debug_assert_eq!(
                    mats.len(),
                    points.len(),
                    "each matrix should have a corresponding set of evaluation points"
                );
                (mats, points)
            })
            .collect_vec();

        // Find the maximum height and the maximum width of matrices in the batch.
        // These do not need to correspond to the same matrix.
        let (global_max_height, global_max_width) = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats.iter().map(|m| (m.height(), m.width())))
            .reduce(|(hmax, wmax), (h, w)| (hmax.max(h), wmax.max(w)))
            .expect("No Matrices Supplied?");
        let log_global_max_height = log2_strict_usize(global_max_height);

        let mut coset = cyclic_subgroup_coset_known_order(
            Val::two_adic_generator(log_global_max_height),
            Val::GENERATOR,
            global_max_height,
        )
        .collect_vec();
        reverse_slice_index_bits(&mut coset);
        
        let inv_denoms = compute_inverse_denominators(&mats_and_points, &coset);

        // Evaluate coset representations and write openings to the challenger
        let all_opened_values = mats_and_points
            .iter()
            .map(|(mats, points)| {
                izip!(mats.iter(), points.iter())
                    .map(|(mat, points_for_mat)| {
                        let h = mat.height() >> self.inner_pcs.fri.log_blowup;
                        // `subgroup` and `mat` are both in bit-reversed order, so we can truncate.
                        let (low_coset, _) = mat.split_rows(h);
                        let coset_h = &coset[..h];

                        points_for_mat
                            .iter()
                            .map(|&point| {
                                // Use Barycentric interpolation to evaluate the matrix at the given point.
                                let ys =
                                    {
                                        let inv_denoms = &inv_denoms.get(&point).unwrap()[..h];
                                        interpolate_coset_with_precomputation(
                                                &low_coset,
                                                Val::GENERATOR,
                                                point,
                                                coset_h,
                                                inv_denoms,
                                        )
                                    };
                                ys.iter()
                                    .for_each(|&y| challenger.observe_algebra_element(y));
                                ys
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Batch combination challenge
        // TODO: Should we be computing a different alpha for each height?
        let alpha: Challenge = challenger.sample_algebra_element();

        // We precompute powers of alpha as we need the same powers for each matrix.
        // We compute both a vector of unpacked powers and a vector of packed powers.
        //no packing
        let alpha_powers: Vec<Challenge> = alpha.powers().take(global_max_width).collect();

        //debug
        let duration = start.elapsed();
        println!("-- GPU-open-stage-1 , duration:{:?}", duration);

        // --- GPU-accelerated Quotient Polynomial Computation ---
        // --- Group matrices by log_height ---
        let mut matrices_by_log_height: BTreeMap<usize, Vec<FfiMatrixData<Val, Challenge>>> = BTreeMap::new();
        for ((mats, points), openings_for_round) in
            mats_and_points.iter().zip(all_opened_values.iter())
        {
            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), openings_for_round.iter())
            {
                let log_height = log2_strict_usize(mat.height());
                matrices_by_log_height
                    .entry(log_height)
                    .or_default()
                    .push(FfiMatrixData {
                        mat: *mat,
                        points_for_mat,
                        openings_for_mat,
                    });
            }
        }


        // --- Main Loop: Iterate over each height group and call GPU ONCE ---
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        
        for (log_height, data_for_height) in matrices_by_log_height {
            let height = 1 << log_height;
            let mut quotient_evals = vec![Challenge::ZERO; height];

            // --- Prepare BATCHED data for the FFI call ---
            let h_lde_data_ptrs: Vec<*const Val> = data_for_height.iter().map(|d| d.mat.values.as_ptr()).collect();
            let h_lde_widths: Vec<i32> = data_for_height.iter().map(|d| d.mat.width() as i32).collect();
            let h_num_points_per_mat: Vec<i32> = data_for_height.iter().map(|d| d.points_for_mat.len() as i32).collect();

            let h_points_z_flat: Vec<Challenge> = data_for_height.iter().flat_map(|d| d.points_for_mat.iter().copied()).collect();
            let h_opened_values_y_flat: Vec<Challenge> = data_for_height.iter().flat_map(|d| d.openings_for_mat.iter().flatten().copied()).collect();
            
            // --- Call the new batched FFI function ---
            let result = unsafe {
                fri_pcs_compute_quotient_for_height_gpu(
                    height as i32,
                    h_lde_data_ptrs.as_ptr() as *const  *const BabyBear,
                    h_lde_widths.as_ptr(),
                    data_for_height.len() as i32,
                    coset.as_ptr() as *const BabyBear,
                    &alpha as *const Challenge as *const InnerChallenge,
                    alpha_powers.as_ptr() as *const InnerChallenge,
                    alpha_powers.len() as i32,
                    h_points_z_flat.as_ptr() as *const InnerChallenge,
                    h_opened_values_y_flat.as_ptr() as *const InnerChallenge,
                    h_num_points_per_mat.as_ptr(),
                    quotient_evals.as_mut_ptr() as *mut InnerChallenge,
                )
            };
            if result != 0 { panic!("GPU batched quotient computation failed for log_height {}", log_height); }
            
            reduced_openings[log_height] = Some(quotient_evals);
        }

        //debug
        let duration = start.elapsed();
        println!("-- GPU-open-stage-2 , duration:{:?}", duration);

        let (fri_proof, query_indices) = prove_gpu(
            &self.inner_pcs.fri,
            &reduced_openings,
            challenger,
        );
        
        // Then do the batched query phase...
         //debug
        let duration = start.elapsed();
        println!("-- GPU-open-stage-3 , duration:{:?}", duration);
            
       // Step 1: Collect all queries into a map.
        let mut queries_by_data: BatchedQueries = BTreeMap::new(); //save <Self::ProverData>
        for index in &query_indices {
            for (data, _) in &rounds {
                let log_max_height = log2_strict_usize(self.inner_pcs.mmcs.get_max_height::<RowMajorMatrix<Val>>(data));
                let bits_reduced = log_global_max_height - log_max_height;
                let reduced_index = *index >> bits_reduced;
                let data_ptr_usize = *data as *const _ as usize;
                queries_by_data.entry(data_ptr_usize).or_default().push(reduced_index);
            }
        }

        // Step 2: Call the custom batched method on the concrete `CudaMerkleTreeMmcs` instance.
        // Note that `self.inner_pcs.mmcs` is our `CudaMerkleTreeMmcs` instance.
        let batched_results: BatchedOpenings<Val, GpuValMmcs> = 
                                    <GpuValMmcs as BatchOpenableMmcs<Val, RowMajorMatrix<Val>>>::open_batches_batched(
                                            &self.inner_pcs.mmcs,
                                            log_global_max_height,
                                            &queries_by_data,
                                        );


        // Step 3: Reconstruct the final `query_openings` Vec from the results map.
        // This is now a fast, in-memory operation with no GPU interaction.
        let query_openings = query_indices //this step is for SP1. Plonky3 does not have this!
            .into_iter()
            .map(|index| {
                rounds
                    .iter()
                    .map(|(data, _)| {
                        // Re-calculate the index to find the correct proof
                        let log_max_height = log2_strict_usize(self.inner_pcs.mmcs.get_max_height::<RowMajorMatrix<Val>>(data));
                        let bits_reduced = log_global_max_height - log_max_height;
                        let reduced_index = index >> bits_reduced;

                        let data_ptr_usize = *data as *const _ as usize;
                        
                        // Direct, unambiguous lookup!
                        batched_results
                            .get(&data_ptr_usize)
                            .expect("ProverData should exist in results")
                            .get(&reduced_index)
                            .expect("Index should exist in results for this ProverData")
                            .clone() // Clone the BatchOpening
                    })
                    .collect()
            })
            .collect();
        
        //debug
        let duration = start.elapsed();
        println!("-- GPU-open-stage-4 , duration:{:?}", duration);

        (
            all_opened_values,
            TwoAdicFriPcsProof {
                fri_proof,
                query_openings,
            },
        )
    }

    fn verify(
        &self,
        rounds: Vec<(Self::Commitment, Vec<(Self::Domain, Vec<(Challenge, Vec<Challenge>)>)>)>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::verify(
            &self.inner_pcs, rounds, proof, challenger)
    }

}


fn compute_inverse_denominators<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &Vec<Vec<EF>>)],
    coset: &[F],
) -> LinearMap<EF, Vec<EF>> {
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in izip!(mats, *points) {
            let log_height = log2_strict_usize(mat.height());
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            (
                z,
                batch_multiplicative_inverse(
                    &coset[..(1 << log_height)]
                        .iter()
                        .map(|&x| z - x)
                        .collect_vec(),
                ),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use crate::baby_bear_poseidon2::{Val, Challenge, MyHash, MyCompress, my_perm};
    use crate::InnerChallengeMmcs;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;
    use p3_maybe_rayon::prelude::ParallelIterator;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_commit::ExtensionMmcs;
    use p3_dft::{Radix2DitParallel, NaiveDft};

     use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, };


    #[test]
    fn test_mat_compress() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        const H: usize = 16;
        const W: usize = 8;

        let lde_matrix = RowMajorMatrix::<BabyBear>::rand(&mut rng, H, W);
        let alpha: Challenge = rng.r#gen();
        let alpha_powers: Vec<Challenge> = alpha.powers().take(W).collect();

        // 1. CPU Reference
        let cpu_result: Vec<Challenge> = lde_matrix.par_rows()
            .map(|row| dot_product(alpha_powers.iter().copied(), row.map(|x| Challenge::from_prime_subfield(x))))
            .collect();

        // 2. GPU Call
        let mut gpu_result = vec![Challenge::ZERO; H];
        let res = unsafe {
            stark_test_mat_compress_gpu(
                lde_matrix.values.as_ptr(), H as i32, W as i32,
                alpha_powers.as_ptr(),
                gpu_result.as_mut_ptr(),
            )
        };
        //println!("---cpu_result:{:?}", cpu_result);
        //println!("---gpu_result:{:?}", gpu_result);
        assert_eq!(res, 0);

        // 3. Compare
        assert_eq!(cpu_result, gpu_result);
        println!("SUCCESS: compute_mat_compressed_kernel is correct.");
    }

    #[test]
    fn test_inv_denoms() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        const H: usize = 8;
        let log_h = log2_strict_usize(H);

        let z: Challenge = rng.r#gen();
        let g = BabyBear::two_adic_generator(log_h);
        let coset: Vec<BabyBear> = g.powers().take(H).collect();
        println!("---input,H:{}", H);
        println!("---input,z:{:?}", z);
        println!("---input,coset:{:?}", coset);
        
        // 1. CPU Reference
        let cpu_result: Vec<Challenge> = coset.iter()
            .map(|&x| (z - Challenge::from_prime_subfield(x)).inverse())
            .collect();

        // 2. GPU Call
        let mut gpu_result = vec![Challenge::ZERO; H];
        let res = unsafe {
            stark_test_inv_denoms_gpu(
                z, coset.as_ptr(), H as i32, gpu_result.as_mut_ptr()
            )
        };
        assert_eq!(res, 0);

        // 3. Compare
        assert_eq!(cpu_result, gpu_result);
        println!("SUCCESS: compute_inv_denoms_kernel is correct.");
    }

    #[test]
    fn test_quotient_loop() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        const H: usize = 16;

        // 1. Prepare random inputs
        let mat_compressed: Vec<Challenge> = (0..H).map(|_| rng.r#gen()).collect();
        let inv_denoms: Vec<Challenge> = (0..H).map(|_| rng.r#gen()).collect();
        let mut quotient_evals: Vec<Challenge> = (0..H).map(|_| rng.r#gen()).collect();
        let y_mat: Challenge = rng.r#gen();
        let alpha_pow_offset: Challenge = rng.r#gen();

        // 2. CPU Reference
        let mut cpu_result = quotient_evals.clone();
        for r in 0..H {
            let term = (y_mat - mat_compressed[r]) * inv_denoms[r];
            cpu_result[r] += alpha_pow_offset * term;
        }

        // 3. GPU Call
        let mut gpu_result = quotient_evals.clone();
        let res = unsafe {
            stark_test_quotient_loop_gpu(
                mat_compressed.as_ptr(), inv_denoms.as_ptr(),
                y_mat, alpha_pow_offset, H as i32,
                gpu_result.as_mut_ptr()
            )
        };
        assert_eq!(res, 0);
        
        // 4. Compare
        assert_eq!(cpu_result, gpu_result);
        println!("SUCCESS: compute_quotient_main_loop_kernel is correct.");
    }

    fn generate_evaluations(
        num_matrices: usize,
        heights: &[usize],
        widths: &[usize],
    ) -> Vec<(TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>)> {
        let mut rng = thread_rng();

        let mut evaluations = Vec::new();
        for i in 0..num_matrices {
            let height = heights[i % heights.len()];
            let width = widths[i % widths.len()];
            let log_n = log2_strict_usize(height);
            let domain = TwoAdicMultiplicativeCoset:: new(Val::GENERATOR, log_n);
            let evals = RowMajorMatrix::rand(&mut rng, height, width);

            /*let width = 2;
            let height = 4;
            let log_n = log2_strict_usize(height);
            let domain = TwoAdicMultiplicativeCoset:: new(Val::GENERATOR, log_n);
            let values = vec![BabyBear::from_u32(1), BabyBear::from_u32(2),
                        BabyBear::from_u32(3), BabyBear::from_u32(4),
                        BabyBear::from_u32(5), BabyBear::from_u32(6),
                        BabyBear::from_u32(7), BabyBear::from_u32(8),];
            let evals = RowMajorMatrix::new(values, 2);
            */

            evaluations.push((domain.unwrap(), evals));
        }
        evaluations
    }


    #[test]
    fn test_fused_gpu_commit_consistency() {
        // 1. SETUP
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        
        // a. Create the CPU PCS for our ground truth
        let cpu_dft = Radix2DitParallel::<Val>::default();
        let cpu_val_mmcs = MerkleTreeMmcs::<Val, Val, _, _, 8>::new(hash.clone(), compress.clone());
        let cpu_challenge_mmcs = ExtensionMmcs::<Val, Challenge,
                    MerkleTreeMmcs<Val,
                            Val,
                            MyHash,
                            MyCompress,
                            8,
                        >>::new(cpu_val_mmcs.clone());
        let cpu_fri_config = FriConfig {
            log_blowup: 1,
            log_final_poly_len:0,
            num_queries: 10,
            proof_of_work_bits: 8,
            mmcs: cpu_challenge_mmcs,
        };
        let cpu_pcs = TwoAdicFriPcs::new(cpu_dft, cpu_val_mmcs, cpu_fri_config);
        
        // b. Create our GPU PCS to be tested
        let gpu_dft = GpuDft::default();
        let gpu_val_mmcs = GpuValMmcs::new(hash.clone(), compress.clone());
        let gpu_challenge_mmcs = GpuChallengeMmcs::new(gpu_val_mmcs.clone());
        let gpu_fri_config = FriConfig {
            log_blowup: 1,
            log_final_poly_len:0,
            num_queries: 10,
            proof_of_work_bits: 8,
            mmcs: gpu_challenge_mmcs,
        };
        let gpu_pcs = GpuPcs::new(gpu_dft, gpu_val_mmcs, gpu_fri_config);

        // c. Generate a batch of evaluations
        let evaluations = generate_evaluations(3, &[128, 256], &[4, 8]);
        
        println!("\n--- Testing fused GPU commit against CPU reference ---");

        // 2. COMPUTE ON CPU (The ground truth)
        println!("Running reference commit on CPU...");
        let start_cpu = std::time::Instant::now();
        let (expected_commit, expected_prover_data) = p3_commit::Pcs::<_, crate::baby_bear_poseidon2::Challenger>::commit(&cpu_pcs, evaluations.clone());
        let duration_cpu = start_cpu.elapsed();
        println!("CPU commit finished in {:?}", duration_cpu);
        // Get the LDEs from the CPU prover data to compare
        let expected_ldes = cpu_pcs.mmcs.get_matrices(&expected_prover_data);
        
        // 3. COMPUTE ON GPU (The implementation under test)
        println!("Running fused commit on GPU...");
        let start_gpu = std::time::Instant::now();
        let (actual_commit, actual_prover_data) = gpu_pcs.commit(evaluations);
        let duration_gpu = start_gpu.elapsed();
        println!("GPU commit finished in {:?}", duration_gpu);
        // Get the LDEs from the GPU prover data to compare
        let actual_ldes = gpu_pcs.inner_pcs.mmcs.get_matrices::<RowMajorMatrix<Val>>(&actual_prover_data);

        // 4. COMPARE RESULTS
        
        // 4a. Compare the final Merkle roots
        println!("\n--- Comparing Roots ---");
        println!("CPU Root: {:?}", expected_commit);
        println!("GPU Root: {:?}", actual_commit);
        assert_eq!(expected_commit, actual_commit, "The final commitment (Merkle root) does not match!");
        println!("✅ Commitments match.");

        // 4b. Compare the generated LDEs stored in the ProverData
        println!("\n--- Comparing LDE Matrices ---");
        assert_eq!(expected_ldes.len(), actual_ldes.len(), "Different number of LDE matrices were generated!");
        
        let mut all_ldes_match = true;
        for i in 0..expected_ldes.len() {
            // Convert to RowMajorMatrix for easier comparison if needed
            let expected_mat = expected_ldes[i].clone().to_row_major_matrix();
            let actual_mat = actual_ldes[i].clone().to_row_major_matrix();
            if expected_mat != actual_mat {
                println!("Mismatch in LDE matrix #{}", i);
                all_ldes_match = false;
                // You can add more detailed diffing here if needed
            }
        }
        assert!(all_ldes_match, "One or more LDE matrices do not match!");
        println!("✅ LDE matrices match.");
        
        println!("\nSUCCESS: The fused GPU commit process is correct.");
    }
    
}