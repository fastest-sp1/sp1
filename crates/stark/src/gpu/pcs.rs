use p3_commit::{Mmcs, OpenedValues, Pcs};

use p3_baby_bear::BabyBear;
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField, batch_multiplicative_inverse,
    coset::TwoAdicMultiplicativeCoset, cyclic_subgroup_coset_known_order,
};
use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsProof};
use p3_interpolation::interpolate_coset_with_precomputation;
use p3_matrix::{
    Matrix,
    bitrev::BitReversedMatrixView,
    dense::{RowMajorMatrix, RowMajorMatrixView},
};

use p3_challenger::FieldChallenger;
use p3_util::{linear_map::LinearMap, log2_strict_usize, reverse_slice_index_bits};

use crate::gpu::ffi::*;
use crate::gpu::gpu_prove::*;
use crate::gpu::merkle::{BatchOpenableMmcs, BatchedOpenings, BatchedQueries, GpuMerkleProverData};
use itertools::{Itertools, izip};
use p3_maybe_rayon::prelude::IntoParallelRefIterator;
use p3_maybe_rayon::prelude::ParallelIterator;
use std::collections::BTreeMap;
use std::os::raw::c_void;

// Import SP1 specific types
use crate::baby_bear_poseidon2::{Challenge, Challenger, MyCompress, MyHash, Val};
use crate::{GpuChallengeMmcs, GpuPcs, GpuValMmcs, InnerChallenge, InnerDft};

//#[cfg(feature = "recursion_cuda")]
use crate::{CudaResultCheck, GpuDft, GpuMatrix, GpuMatrixC, GpuMemBlk, GpuMerkleTreeMmcs};

//use serde::{Deserialize, Serialize};

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

struct FfiGpuMatrixData<'a, Val, Challenge>
where
    Val: p3_field::Field,
    Challenge: p3_field::Field,
{
    // global_index: usize,
    mat: &'a GpuMatrix<Val>,
    points_for_mat: &'a Vec<Challenge>,
    openings_for_mat: &'a Vec<Vec<Challenge>>,
}

pub type GpuMmcs =
    GpuMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;

/// A GPU-accelerated implementation of the Pcs trait, wrapping plonky3's TwoAdicFriPcs.
/// It delegates most methods to the inner PCS, but provides custom, GPU-accelerated
/// implementations for performance-critical methods like `open`, `mmc commit`.
#[derive(Debug)]
pub struct GpuFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub inner_pcs: TwoAdicFriPcs<Val, Dft, InputMmcs, FriMmcs>,
    pub gpu_dft: GpuDft,
    // pub gpu_mmcs: GpuMmcs,
}

impl<Val, Dft, InputMmcs, FriMmcs> GpuFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub fn new(dft: Dft, mmcs: InputMmcs, fri_config: FriConfig<FriMmcs>) -> Self {
        Self {
            inner_pcs: TwoAdicFriPcs::new(dft, mmcs, fri_config),
            gpu_dft: GpuDft::default(),
            //gpu_mmcs: GpuMmcs::new(mmcs.hash, mmcs.compress),
        }
    }
}

impl Pcs<Challenge, Challenger> for GpuPcs {
    // All associated types are defined using concrete types.
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<
        Challenge,
        Challenger,
    >>::Commitment;
    type ProverData = <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<
        Challenge,
        Challenger,
    >>::ProverData;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixView<'a, Val>>;
    type Proof = <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<
        Challenge,
        Challenger,
    >>::Proof;
    type Error = <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<
        Challenge,
        Challenger,
    >>::Error;
    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &self.inner_pcs,
            degree,
        )
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        <TwoAdicFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs> as Pcs<Challenge, Challenger>>::commit(
            &self.inner_pcs,
            evaluations,
        )
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
        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                let mats = self
                    .inner_pcs
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
                                let ys = {
                                    let inv_denoms = &inv_denoms.get(&point).unwrap()[..h];
                                    interpolate_coset_with_precomputation(
                                        &low_coset,
                                        Val::GENERATOR,
                                        point,
                                        coset_h,
                                        inv_denoms,
                                    )
                                };
                                ys.iter().for_each(|&y| challenger.observe_algebra_element(y));
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
        //let duration = start.elapsed();
        //println!("-- GPU-open-stage-1 , duration:{:?}", duration);

        // --- GPU-accelerated Quotient Polynomial Computation ---
        // --- Group matrices by log_height ---
        let mut matrices_by_log_height: BTreeMap<usize, Vec<FfiMatrixData<Val, Challenge>>> =
            BTreeMap::new();
        for ((mats, points), openings_for_round) in
            mats_and_points.iter().zip(all_opened_values.iter())
        {
            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), openings_for_round.iter())
            {
                let log_height = log2_strict_usize(mat.height());
                matrices_by_log_height.entry(log_height).or_default().push(FfiMatrixData {
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
            let h_lde_data_ptrs: Vec<*const Val> =
                data_for_height.iter().map(|d| d.mat.values.as_ptr()).collect();
            let h_lde_widths: Vec<i32> =
                data_for_height.iter().map(|d| d.mat.width() as i32).collect();
            let h_num_points_per_mat: Vec<i32> =
                data_for_height.iter().map(|d| d.points_for_mat.len() as i32).collect();

            let h_points_z_flat: Vec<Challenge> =
                data_for_height.iter().flat_map(|d| d.points_for_mat.iter().copied()).collect();
            let h_opened_values_y_flat: Vec<Challenge> = data_for_height
                .iter()
                .flat_map(|d| d.openings_for_mat.iter().flatten().copied())
                .collect();

            // --- Call the new batched FFI function ---
            unsafe {
                let _ = fri_pcs_compute_quotient_for_height_gpu(
                    height as i32,
                    h_lde_data_ptrs.as_ptr() as *const *const BabyBear,
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
                .check("fri_pcs_compute_quotient_for_height_gpu failed.");
            }

            reduced_openings[log_height] = Some(quotient_evals);
        }
        //end GPU
        //debug
        //let duration = start.elapsed();
        //println!("-- GPU-open-stage-2 , duration:{:?}", duration);

        //let g: TwoAdicFriGenericConfigForMmcs<Val, GpuValMmcs> =
        //    TwoAdicFriGenericConfig(PhantomData);

        let (fri_proof, query_indices) =
            prove_gpu(&self.inner_pcs.fri, &reduced_openings, challenger);

        //debug
        //let duration = start.elapsed();
        //println!("-- GPU-open-stage-3 , duration:{:?}", duration);

        // Step 1: Collect all queries into a map.
        let mut queries_by_data: BatchedQueries = BTreeMap::new(); //save <Self::ProverData>
        for index in &query_indices {
            //println!("----index:{}", index);
            for (data, _) in &rounds {
                let log_max_height = log2_strict_usize(
                    self.inner_pcs.mmcs.get_max_height::<RowMajorMatrix<Val>>(data),
                );
                let bits_reduced = log_global_max_height - log_max_height;
                let reduced_index = *index >> bits_reduced;
                //println!("====reduced_index:{}", reduced_index);
                let data_ptr_usize = *data as *const _ as usize;
                queries_by_data.entry(data_ptr_usize).or_default().push(reduced_index);
            }
        }

        // Step 2: Call the custom batched method on the concrete `GpuMerkleTreeMmcs` instance.
        // Note that `self.inner_pcs.mmcs` is our `GpuMerkleTreeMmcs` instance.
        let batched_results: BatchedOpenings<Val, GpuValMmcs> = 
            //self.inner_pcs.mmcs.open_batches_batched(log_global_max_height, &queries_by_data);
            <GpuValMmcs as BatchOpenableMmcs<Val, Val, 8, RowMajorMatrix<Val>>>::open_batches_batched(
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
                        let log_max_height = log2_strict_usize(
                            self.inner_pcs.mmcs.get_max_height::<RowMajorMatrix<Val>>(data),
                        );
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
        //let duration = start.elapsed();
        //println!("-- GPU-open-stage-4 , duration:{:?}", duration);

        (all_opened_values, TwoAdicFriPcsProof { fri_proof, query_openings })
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

impl GpuPcs {
    /// Commits to a batch of matrices that are already resident on the GPU.
    pub fn commit_gpu(
        &self,
        domains_and_gpu_traces: Vec<(TwoAdicMultiplicativeCoset<Val>, GpuMatrix<Val>)>,
        gpu_mem_blk: &GpuMemBlk,
    ) -> (
        // The return types are also concrete.
        <Self as Pcs<Challenge, Challenger>>::Commitment,
        GpuMerkleProverData<GpuMatrix<Val>>,
    ) {
        let log_blowup = self.inner_pcs.fri.log_blowup;

        // 1. Compute the LDEs for all traces on the GPU.
        let ldes: Vec<GpuMatrix<Val>> = domains_and_gpu_traces
            .par_iter() // We can still use Rayon to parallelize work across different matrices
            //.iter()
            .map(|(domain, gpu_trace)| {
                //let log_height = log2_strict_usize(gpu_trace.height);
                assert_eq!(domain.size(), gpu_trace.height);
                let shift = Val::GENERATOR / domain.shift();
                let added_bits = log_blowup;
                self.gpu_dft.coset_lde_batch_gpu(gpu_trace, added_bits, shift, gpu_mem_blk)
            })
            .collect();

        // At this point, `ldes` is a Vec of `GpuMatrix`, containing the LDEs on the device.
        // 2. Commit to the LDEs using the GPU-accelerated MMCS.
        let (commit, prover_data) = GpuValMmcs::commit_gpu(ldes);

        (commit, prover_data)
    }

    pub fn get_lde_on_domain_gpu(
        &self,
        gpu_prover_data: &GpuMerkleProverData<GpuMatrix<Val>>,
        idx: usize,
    ) -> GpuMatrix<Val> {
        // The ProverData is GpuMerkleProverData.
        let num_matrices = gpu_prover_data.original_to_sorted_indices.len();
        if num_matrices == 0 {
            //return GpuMatrix::<Val>::new(0, 0);
            panic!("no ldes.");
        }

        let sorted_idx = gpu_prover_data.original_to_sorted_indices[idx];

        //not bit-reverse()
        gpu_prover_data.sorted_inputs[sorted_idx].clone()
    }

    pub fn get_evaluations_on_domain_gpu(
        &self,
        gpu_prover_data: &GpuMerkleProverData<GpuMatrix<Val>>,
        idx: usize,
        domain_size: usize,
        gpu_mem_blk: &GpuMemBlk,
    ) -> GpuMatrix<Val> {
        // The ProverData is GpuMerkleProverData.
        let num_matrices = gpu_prover_data.original_to_sorted_indices.len();
        if num_matrices == 0 {
            //return GpuMatrix::<Val>::new(0, 0);
            panic!("no ldes.");
        }

        let sorted_idx = gpu_prover_data.original_to_sorted_indices[idx];
        let height = gpu_prover_data.sorted_inputs[sorted_idx].height;

        assert!(height as usize >= domain_size);

        let mut row_major_mat = self.gpu_dft.bit_reverse_rows(
            &gpu_prover_data.sorted_inputs[sorted_idx],
            domain_size,
            gpu_mem_blk,
        );
        //println!("get_evaluations_on_domain_gpu, return new lde.ptr={:p}, height={},width={}", row_major_mat.buffer.ptr, row_major_mat.height,row_major_mat.width);
        row_major_mat.height = domain_size;
        row_major_mat
    }

    //Notice: Returning Matix is NOT row-major matrix!
    pub fn get_all_ldes_gpu<'a>(
        &self,
        gpu_prover_data: &'a GpuMerkleProverData<GpuMatrix<Val>>,
    ) -> Vec<&'a GpuMatrix<Val>> {
        // The ProverData is GpuMerkleProverData.
        let num_matrices = gpu_prover_data.original_to_sorted_indices.len();
        if num_matrices == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(num_matrices);

        for original_idx in 0..num_matrices {
            let sorted_idx = gpu_prover_data.original_to_sorted_indices[original_idx];
            result.push(&gpu_prover_data.sorted_inputs[sorted_idx]);
        }

        result
    }

    pub fn get_matrix_heights(
        &self,
        gpu_prover_data: &GpuMerkleProverData<GpuMatrix<Val>>,
    ) -> Vec<usize> {
        self.get_all_ldes_gpu(gpu_prover_data).iter().map(|matrix| matrix.height).collect()
    }

    /// Get the largest height of any committed matrix.
    ///
    /// # Panics
    /// This may panic if there are no committed matrices.
    pub fn get_max_height(&self, gpu_prover_data: &GpuMerkleProverData<GpuMatrix<Val>>) -> usize {
        self.get_matrix_heights(gpu_prover_data)
            .into_iter()
            .max()
            .unwrap_or_else(|| panic!("No committed matrices?"))
    }

    pub fn open_gpu(
        &self,
        rounds: Vec<(&GpuMerkleProverData<GpuMatrix<Val>>, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
        gpu_mem_blk: &GpuMemBlk,
    ) -> (OpenedValues<Challenge>, <Self as Pcs<Challenge, Challenger>>::Proof) {
        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                let mats = self.get_all_ldes_gpu(*data); //Vec<&GpuMatrix>
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
            .flat_map(|(mats, _)| mats.iter().map(|m| (m.height, m.width)))
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

        let coset_gpu = GpuMatrix::from_vec(&coset, global_max_height, 1, gpu_mem_blk);
        let inv_denoms = compute_inverse_denominators_gpu(&mats_and_points, &coset_gpu);

        // Evaluate coset representations and write openings to the challenger
        let all_opened_values = mats_and_points
            .iter()
            .map(|(mats, points)| {
                izip!(mats.iter(), points.iter())
                    .map(|(mat, points_for_mat)| {
                        let h = mat.height >> self.inner_pcs.fri.log_blowup;
                        // `subgroup` and `mat` are both in bit-reversed order, so we can truncate.
                        let low_coset_gpu = &**mat; //Mat's data are in gpu

                        //let  coset_h_gpu = coset_gpu.cut_rows(h);
                        let mut coset_h_gpu = coset_gpu.clone();
                        coset_h_gpu.height = h;

                        points_for_mat
                            .iter()
                            .map(|&point| {
                                let ys = {
                                    let inv_denoms = inv_denoms.get(&point).unwrap();

                                    interpolate_coset_with_precomputation_gpu(
                                        &low_coset_gpu,
                                        Val::GENERATOR,
                                        point,
                                        &coset_h_gpu,
                                        &inv_denoms,
                                    )
                                };
                                ys.iter().for_each(|&y| challenger.observe_algebra_element(y));
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
        let alpha_powers: Vec<Challenge> = alpha.powers().take(global_max_width).collect();

        //debug
        //let duration = start.elapsed();
        //println!("-- GPU-open-stage-1 , duration:{:?}", duration);

        // --- GPU-accelerated Quotient Polynomial Computation ---
        // --- Group matrices by log_height ---
        let mut matrices_by_log_height: BTreeMap<usize, Vec<FfiGpuMatrixData<Val, Challenge>>> =
            BTreeMap::new();
        for ((mats, points), openings_for_round) in
            mats_and_points.iter().zip(all_opened_values.iter())
        {
            for (mat, points_for_mat, openings_for_mat) in
                izip!(mats.iter(), points.iter(), openings_for_round.iter())
            {
                let log_height = log2_strict_usize(mat.height);
                matrices_by_log_height.entry(log_height).or_default().push(FfiGpuMatrixData {
                    mat,
                    points_for_mat,
                    openings_for_mat,
                });
            }
        }

        // --- Main Loop: Iterate over each height group and call GPU ONCE ---
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);

        //let alpha_powers_gpu = GpuMatrix::<Challenge>::from_vec(alpha_powers, alpha_powers.len(),1));

        for (log_height, data_for_height) in matrices_by_log_height {
            let height = 1 << log_height; //=mat.height
            let mut quotient_evals = vec![Challenge::ZERO; height];

            // --- Prepare BATCHED data for the FFI call ---
            //let h_lde_data_ptrs: Vec<*const Val> = data_for_height.iter().map(|d| d.mat.values.as_ptr()).collect();
            //let h_lde_widths: Vec<i32> = data_for_height.iter().map(|d| d.mat.width() as i32).collect();
            //let h_num_points_per_mat: Vec<i32> = data_for_height.iter().map(|d| d.points_for_mat.len() as i32).collect();

            let h_num_points_per_mat: Vec<i32> =
                data_for_height.iter().map(|d| d.points_for_mat.len() as i32).collect();
            let h_points_z_flat: Vec<Challenge> =
                data_for_height.iter().flat_map(|d| d.points_for_mat.iter().copied()).collect();
            let h_opened_values_y_flat: Vec<Challenge> = data_for_height
                .iter()
                .flat_map(|d| d.openings_for_mat.iter().flatten().copied())
                .collect();

            //let h_lde_ptrs: Vec<GpuMatrix<Val>> =  data_for_height.iter().map(|d| d.mat.clone()).collect();
            let h_lde_ptrs: Vec<GpuMatrixC> =
                data_for_height.iter().map(|d| d.mat.into()).collect();

            // --- Call the new batched FFI function ---
            unsafe {
                let coset_gpu_c: GpuMatrixC = (&coset_gpu).into();
                let _ = fri_pcs_compute_quotient_for_height_data_in_gpu(
                    h_lde_ptrs.as_ptr(),
                    h_lde_ptrs.len() as i32,
                    &coset_gpu_c,
                    &alpha as *const Challenge,
                    //alpha_powers_gpu.as_ptr() as *const GpuMatrix<Challenge>,
                    alpha_powers.as_ptr() as *const InnerChallenge,
                    alpha_powers.len() as i32,
                    h_points_z_flat.as_ptr() as *const InnerChallenge,
                    h_opened_values_y_flat.as_ptr() as *const InnerChallenge,
                    h_num_points_per_mat.as_ptr(),
                    quotient_evals.as_mut_ptr() as *mut Challenge,
                )
                .check("fri_pcs_compute_quotient_for_height_data_in_gpu failed.");
            }

            reduced_openings[log_height] = Some(quotient_evals);
        }
        //end GPU
        //debug
        //let duration = start.elapsed();
        //println!("-- GPU-open-stage-2 , duration:{:?}", duration);

        let (fri_proof, query_indices) =
            prove_gpu(&self.inner_pcs.fri, &reduced_openings, challenger);

        //debug
        //let duration = start.elapsed();
        //println!("-- GPU-open-stage-3 , duration:{:?}", duration);

        // Step 1: Collect all queries into a map.
        let mut queries_by_data: BatchedQueries = BTreeMap::new(); //save <Self::ProverData>
        for index in &query_indices {
            //println!("----index:{}", index);
            for (data, _) in &rounds {
                //let log_max_height = log2_strict_usize(self.inner_pcs.mmcs.get_max_height::<RowMajorMatrix<Val>>(data));
                let log_max_height = log2_strict_usize(self.get_max_height(data));
                let bits_reduced = log_global_max_height - log_max_height;
                let reduced_index = *index >> bits_reduced;
                //println!("====reduced_index:{}", reduced_index);
                let data_ptr_usize = *data as *const _ as usize;
                queries_by_data.entry(data_ptr_usize).or_default().push(reduced_index);
            }
        }

        // Step 2: Call the custom batched method on the concrete `GpuMerkleTreeMmcs` instance.
        // Note that `self.inner_pcs.mmcs` is our `GpuMerkleTreeMmcs` instance.
        let batched_results: BatchedOpenings<Val, GpuValMmcs> =
            GpuValMmcs::open_batches_batched_gpu(
                // &self.inner_pcs.mmcs,
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
                        let log_max_height = log2_strict_usize(self.get_max_height(data));
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
        //let duration = start.elapsed();
        //println!("-- GPU-open-stage-4 , duration:{:?}", duration);

        (all_opened_values, TwoAdicFriPcsProof { fri_proof, query_openings })
    }
}

pub fn interpolate_coset_with_precomputation_gpu(
    coset_evals: &GpuMatrix<Val>,
    shift: Val,
    point: Challenge,
    coset: &GpuMatrix<Val>,
    diff_invs: &GpuMatrix<Challenge>,
) -> Vec<Challenge> {
    //assert_eq!(coset.height, coset_evals.height);//move to interpolate_coset_gpu

    //assert_eq!(diff_invs.height, coset_evals.height);
    assert_eq!(coset.width, 1);
    assert_eq!(diff_invs.width, 1);

    let num_interpolations = coset_evals.width;
    if num_interpolations == 0 {
        return Vec::new();
    }

    // 1. Allocate a host buffer to receive the final results.
    let mut interpolated_values = vec![Challenge::ZERO; num_interpolations];

    // 2. Call the FFI function.
    unsafe {
        let mut coset_evals_c: GpuMatrixC = coset_evals.into();
        let coset_c: GpuMatrixC = (coset).into();
        let diff_invs_c: GpuMatrixC = (diff_invs).into();
        let _ = interpolate_coset_gpu(
            &mut coset_evals_c,
            shift,
            point,
            &coset_c,
            &diff_invs_c, // as  *const GpuMatrix<Challenge>,
            interpolated_values.as_mut_ptr(),
        )
        .check("interpolate_coset_gpu FFI call failed");
    }

    // 3. Return the results.
    interpolated_values
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
                    &coset[..(1 << log_height)].iter().map(|&x| z - x).collect_vec(),
                ),
            )
        })
        .collect()
}

//V1 is ok but it seems too compolicated.
pub fn compute_inverse_denominators_gpu(
    mats_and_points: &[(Vec<&GpuMatrix<Val>>, &Vec<Vec<Challenge>>)],
    coset: &GpuMatrix<Val>, // Coset is now on the GPU
) -> LinearMap<Challenge, GpuMatrix<Challenge>> {
    // Returns a map to GPU matrices
    // Step 1: Find max_log_height_for_point (remains on CPU)
    let mut max_log_height_for_point: LinearMap<Challenge, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in izip!(mats, *points) {
            let log_height = log2_strict_usize(mat.height);
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    // Step 2: Prepare data for a single batched FFI call
    //let map_iter = (&max_log_height_for_point).into_iter();

    // Get the length from the iterator.
    let num_unique_points = max_log_height_for_point.values().count();
    let mut unique_points: Vec<Challenge> = Vec::with_capacity(num_unique_points);
    let mut max_log_heights: Vec<i32> = Vec::with_capacity(num_unique_points);

    // Use a `for` loop to iterate over the LinearMap.
    // This calls `.into_iter()` implicitly on a reference to the map.
    for (point_ref, log_height_ref) in max_log_height_for_point {
        unique_points.push(point_ref);
        max_log_heights.push(log_height_ref as i32);
    }

    // Allocate host memory to receive the array of device pointers for the results.
    let mut d_inv_denoms_ptrs: Vec<*mut Challenge> =
        vec![std::ptr::null_mut(); unique_points.len()];

    // Step 3: Call the FFI orchestrator
    unsafe {
        let _ = compute_inverse_denominators_for_points_gpu(
            unique_points.as_ptr(),
            unique_points.len() as i32,
            max_log_heights.as_ptr(),
            coset.as_ptr(),
            d_inv_denoms_ptrs.as_mut_ptr(),
        )
        .check("compute_inverse_denominators_for_points_gpu failed");
    }

    // Step 4: Wrap the returned device pointers in GpuMatrix handles
    unique_points
        .into_iter()
        .zip(d_inv_denoms_ptrs.into_iter())
        .zip(max_log_heights.into_iter())
        .map(|((point, ptr), log_height)| {
            let height = 1 << log_height;
            let gpu_matrix = unsafe {
                GpuMatrix::from_raw_parts(ptr as *mut c_void, 1, height) // New GpuMatrix constructor
            };
            (point, gpu_matrix)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baby_bear_poseidon2::{Challenge, MyCompress, MyHash, Val, my_perm};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::dot_product;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_maybe_rayon::prelude::ParallelIterator;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use p3_air::Air;
    use p3_challenger::CanObserve;
    use p3_commit::{ExtensionMmcs, Mmcs, Pcs};
    use p3_field::extension::BinomialExtensionField;

    use p3_matrix::Matrix;
    use p3_symmetric::TruncatedPermutation;

    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DitParallel;

    use p3_fri::{FriConfig, TwoAdicFriPcs};
    use p3_merkle_tree::MerkleTreeMmcs;

    //use p3_baby_bear::BabyBear;
    use p3_commit::PolynomialSpace;

    use p3_challenger::FieldChallenger;

    use crate::baby_bear_poseidon2::StarkConfigCpu;
    use crate::config::StarkGenericConfig;
    use crate::gpu::matrix::GpuMatrix;
    use crate::{BabyBearPoseidon2Inner, StarkConfigGpu};
    use p3_field::{TwoAdicField, batch_multiplicative_inverse};
    use p3_symmetric::PaddingFreeSponge;
    use p3_util::{linear_map::LinearMap, log2_strict_usize};

    //use rand::{thread_rng, };
    use itertools::Itertools;

    // Redefine types for testing

    type Perm = Poseidon2BabyBear<16>;

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    type F = BabyBear;

    use rand::distributions::{Distribution, Standard};
    use rand::thread_rng;

    // Helper function to generate a random matrix.
    fn generate_random_matrix(h: usize, w: usize) -> RowMajorMatrix<BabyBear> {
        let mut rng = thread_rng();
        let values = (0..h * w).map(|_| rng.sample(Standard)).collect();
        RowMajorMatrix::new(values, w)
    }

    #[test]
    fn test_pcs_commit_data_in_gpu() {
        //let cpu_config = BabyBearPoseidon2Inner::default(); //pass
        //let gpu_config = StarkConfigGpu::default();

        let cpu_config = StarkConfigCpu::compressed(); //
        let gpu_config = StarkConfigGpu::compressed();

        const H: usize = 8192 * 2 * 2 * 2 * 2 * 2 * 2 * 2;
        const W: usize = 4;
        let log_h = log2_strict_usize(H);

        //const ADDED_BITS: usize = 1; // blowup_factor = 4

        println!("Generating random matrix of size {}x{}", H, W);
        let main_trace = generate_random_matrix(H, W);

        let log_trace_rows = log2_strict_usize(main_trace.height());
        let trace_rows = main_trace.height();

        let log_quotient_degree = 1; //fix it according the inputs
        let quotient_degree = 1 << log_quotient_degree;

        let trace_domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            cpu_config.pcs(),
            main_trace.height(),
        );
        let quotient_domain =
            trace_domain.create_disjoint_domain(1 << (log_trace_rows + log_quotient_degree));

        println!(
            "--trace.height={}, width={}, trace_domain.size={}, qd_size={}",
            main_trace.height(),
            main_trace.width(),
            trace_domain.size(),
            quotient_domain.size()
        );

        let (cpu_commit, cpu_commit_data) = <MyPcs as Pcs<Challenge, Challenger>>::commit(
            cpu_config.pcs(),
            [(trace_domain.clone(), main_trace.clone())],
        );

        let gpu_mem_blk = GpuMemBlk::new(1000 * 1024 * 1024) //100M ?
            .expect("Failed to create GPU memory block");

        let gpu_trace = GpuMatrix::from_vec(
            &main_trace.values,
            main_trace.height(),
            main_trace.width(),
            &gpu_mem_blk,
        );
        //let log_blowup = &self.inner_pcs.fri.log_blowup;
        let (gpu_commit, gpu_commit_data) =
            GpuPcs::commit_gpu(gpu_config.pcs(), vec![(trace_domain, gpu_trace)], &gpu_mem_blk);

        assert_eq!(cpu_commit, gpu_commit, "commit is not equal!");
        println!("---commit is equal.");

        let main_trace_on_quotient_domains =
            <MyPcs as Pcs<Challenge, Challenger>>::get_evaluations_on_domain(
                cpu_config.pcs(),
                &cpu_commit_data,
                0,
                quotient_domain,
            )
            .to_row_major_matrix();
        let gpu_main_lde = GpuPcs::get_evaluations_on_domain_gpu(
            gpu_config.pcs(),
            &gpu_commit_data,
            0,
            quotient_domain.size(),
            &gpu_mem_blk,
        );
        let gpu_main_lde_values = GpuMatrix::to_host(&gpu_main_lde);

        assert_eq!(
            main_trace_on_quotient_domains.values, gpu_main_lde_values,
            "commit lde is not equal!"
        );

        //println!("RowMajorMatrix， main_trace_on_quotient_domains,val={:?}", main_trace_on_quotient_domains.values);
        //println!("RowMajorMatrix， gpu_main_lde_values,val={:?}", gpu_main_lde_values);

        //
        let cpu_mats =
            ValMmcs::get_matrices::<RowMajorMatrix<Val>>(&cpu_config.pcs().mmcs, &cpu_commit_data)
                .into_iter()
                .map(|m| m.as_view())
                .collect_vec();

        let gpu_mats = GpuPcs::get_all_ldes_gpu(gpu_config.pcs(), &gpu_commit_data);
        assert_eq!(cpu_mats.len(), gpu_mats.len(), "mats len is not equal!");

        let _: Vec<_> = cpu_mats
            .iter()
            .zip(gpu_mats)
            .enumerate()
            .map(|(i, (cpu, gpu))| {
                let gpu_mat_valuse = GpuMatrix::to_host(&gpu);
                assert_eq!(cpu.values, gpu_mat_valuse, "mat[{i}] values is not equal!");
                //println!("--i={}, cpu.values:{:?}",i, cpu.values);
                //println!("--i={}, gpu.values:{:?}",i, gpu_mat_valuse);
                0
            })
            .collect();

        println!("---test is ok.");
    }

    #[test]
    fn test_mat_compress() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        const H: usize = 16;
        const W: usize = 8;

        let lde_matrix = RowMajorMatrix::<BabyBear>::rand(&mut rng, H, W);
        let alpha: Challenge = rng.r#gen();
        let alpha_powers: Vec<Challenge> = alpha.powers().take(W).collect();

        // 1. CPU Reference
        let cpu_result: Vec<Challenge> = lde_matrix
            .par_rows()
            .map(|row| {
                dot_product(
                    alpha_powers.iter().copied(),
                    row.map(|x| Challenge::from_prime_subfield(x)),
                )
            })
            .collect();

        // 2. GPU Call
        let mut gpu_result = vec![Challenge::ZERO; H];
        let res = unsafe {
            stark_test_mat_compress_gpu(
                lde_matrix.values.as_ptr(),
                H as i32,
                W as i32,
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
        let cpu_result: Vec<Challenge> =
            coset.iter().map(|&x| (z - Challenge::from_prime_subfield(x)).inverse()).collect();

        // 2. GPU Call
        let mut gpu_result = vec![Challenge::ZERO; H];
        let res = unsafe {
            stark_test_inv_denoms_gpu(z, coset.as_ptr(), H as i32, gpu_result.as_mut_ptr())
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
                mat_compressed.as_ptr(),
                inv_denoms.as_ptr(),
                y_mat,
                alpha_pow_offset,
                H as i32,
                gpu_result.as_mut_ptr(),
            )
        };
        assert_eq!(res, 0);

        // 4. Compare
        assert_eq!(cpu_result, gpu_result);
        println!("SUCCESS: compute_quotient_main_loop_kernel is correct.");
    }

    fn compute_inverse_denominators_cpu(
        max_log_height_for_point: LinearMap<Challenge, usize>,
        coset: &[Val],
    ) -> LinearMap<Challenge, Vec<Challenge>> {
        max_log_height_for_point
            .into_iter()
            .map(|(z, log_height)| {
                let denoms = &coset[..(1 << log_height)]
                    .iter()
                    .map(|&x| z - Challenge::from(x))
                    .collect_vec();
                (z, batch_multiplicative_inverse(denoms))
            })
            .collect()
    }

    /*
    //TBD
    #[test]
    fn test_compute_inverse_denominators_gpu() {
        // 1. SETUP
        println!("--- Setting up test for compute_inverse_denominators_gpu ---");
        let mut rng = thread_rng();

        // Define parameters for the test
        let max_log_coset_size = 20;
        let coset_size = 1 << max_log_coset_size;
        let num_unique_points = 100;

        // Create a random coset on the CPU
        let root = Val::two_adic_generator(max_log_coset_size);
        let coset_cpu: Vec<Val> = root.powers().take(coset_size).collect();

        // Create a map of unique query points to different log_heights
        let mut max_log_height_for_point_cpu: LinearMap<Challenge, usize> = LinearMap::new();
        let mut max_log_height_for_point: LinearMap<Challenge, usize> = LinearMap::new();
        //let log_height = 4;
        let mut mats_vecs = Vec::new();

        for _ in 0..num_unique_points {
            let point: Challenge = rng.r#gen();
            // Use varying log_heights to test that part of the logic
            let log_height = rng.r#gen_range(4..=max_log_coset_size);
           // println!("--org-input, point={:?}, log_height={}", point, log_height);
            max_log_height_for_point.insert(point, log_height);
            max_log_height_for_point_cpu.insert(point, log_height);

            let mat = GpuMatrix::<Val>::new(1<<log_height, 64);
            mats_vecs.push(mat);
        }
         println!("--input--, cpu_len:{} , gpu_len:{}", max_log_height_for_point_cpu.values().count(),
            max_log_height_for_point.values().count());
        // Create dummy `mats_and_points` to pass to the GPU wrapper.
        // In this test, we don't actually need the matrices, just the points.
        // We pass an empty Vec of GpuMatrix.
        let mats_and_points: Vec<(Vec<GpuMatrix<Val>>, &Vec<Vec<Challenge>>)> = vec![];

        println!("Testing with {} unique points and max coset size of {}", num_unique_points, coset_size);

        // 2. CPU REFERENCE COMPUTATION
        println!("Computing reference inverse denominators on CPU...");
        let start_cpu = std::time::Instant::now();
        let expected_results_map = compute_inverse_denominators_cpu(
            max_log_height_for_point_cpu,
            &coset_cpu,
        );
        println!("CPU computation finished in {:?}", start_cpu.elapsed());


        // 3. GPU EXECUTION
        println!("Computing inverse denominators on GPU...");
        // a) Copy coset data to the GPU
        let coset_gpu = GpuMatrix::from_vec(&coset_cpu, coset_size, 1);

        // b) Create the necessary inputs for the GPU wrapper from our map
        let mut mats_and_points_for_gpu: Vec<(Vec<GpuMatrix<Val>>, Vec<Vec<Challenge>>)> = Vec::new();
        let mut points_vecs = Vec::new();
        for (point, _log_height) in max_log_height_for_point {
            points_vecs.push(vec![point]);
        }



        // We create a structure that matches what the wrapper expects, even if it's a bit artificial for this test.
        let points_refs: Vec<&Vec<Challenge>> = points_vecs.iter().collect();
        mats_and_points_for_gpu.push((mats_vecs, points_vecs));

        // c) Call the GPU wrapper function. This is the main function under test.
        let start_gpu = std::time::Instant::now();
        // mats_and_points: &[(Vec<&GpuMatrix<Val>>, &Vec<Vec<Challenge>>)],
        let actual_results_map_gpu = compute_inverse_denominators_gpu(
            &mats_and_points_for_gpu.iter().map(|(m,p)| (m, p)).collect_vec(), // Reconstruct the expected type
            &coset_gpu
        );
        println!("GPU computation finished in {:?}", start_gpu.elapsed());

        // 4. COMPARE RESULTS
        println!("Comparing CPU and GPU results...");
        assert_eq!(expected_results_map.values().count(), actual_results_map_gpu.values().count(), "Mismatch in number of points processed");

        for (point, expected_inv_denoms_vec) in expected_results_map {
            // Find the corresponding result from the GPU map
            let actual_inv_denoms_gpu_matrix = actual_results_map_gpu.get(&point)
                .expect("Point missing from GPU results");

            // Download the data from the GPU
            let actual_inv_denoms_vec = actual_inv_denoms_gpu_matrix.to_host();

            // Compare the vectors
            assert_eq!(
                *expected_inv_denoms_vec,
                actual_inv_denoms_vec,
                "Inverse denominators do not match for point {:?}", point
            );
        }

        println!("\nSUCCESS: compute_inverse_denominators_gpu is correct.");
    }*/

    #[test]
    fn test_interpolate_coset_gpu() {
        // 1. SETUP
        println!("--- Setting up test for interpolate_coset_with_precomputation_gpu ---");
        let mut rng = thread_rng();

        // Define parameters for a non-trivial test case
        let log_height = 8;
        let height = 1 << log_height;
        let width = 12; // Test with multiple polynomials at once

        // Generate random CPU-side data
        let coset_evals_cpu = RowMajorMatrix::<Val>::rand(&mut rng, height, width);
        let shift: Val = rng.r#gen();
        let point: Challenge = rng.r#gen();

        // Generate the coset elements on the CPU
        let root = Val::two_adic_generator(log_height);
        let coset_cpu: Vec<Val> = root.powers().take(height).collect();

        println!(
            "Testing interpolation of {} polynomials of degree < {} at a random point.",
            width, height
        );

        // 2. PRECOMPUTATION (done on CPU, as it would be in the prover)
        let diffs: Vec<Challenge> = coset_cpu.iter().map(|&x| point - Challenge::from(x)).collect();
        let diff_invs_cpu: Vec<Challenge> = batch_multiplicative_inverse(&diffs);

        // 3. CPU REFERENCE COMPUTATION
        println!("Computing interpolation on CPU...");
        let start_cpu = std::time::Instant::now();
        let expected_values: Vec<Challenge> = interpolate_coset_with_precomputation(
            &coset_evals_cpu,
            shift,
            point,
            &coset_cpu,
            &diff_invs_cpu,
        );
        println!("CPU computation finished in {:?}", start_cpu.elapsed());

        // 4. GPU EXECUTION
        println!("Computing interpolation on GPU...");
        // a) Copy all necessary data to GPU buffers
        let gpu_mem_blk = GpuMemBlk::new(100 * 1024 * 1024) //100M ?
            .expect("Failed to create GPU memory block");

        let coset_evals_gpu =
            GpuMatrix::from_vec(&coset_evals_cpu.values, height, width, &gpu_mem_blk);
        let coset_gpu = GpuMatrix::from_vec(&coset_cpu, height, 1, &gpu_mem_blk);
        let diff_invs_gpu = GpuMatrix::from_vec(&diff_invs_cpu, height, 1, &gpu_mem_blk);

        // b) Call the GPU wrapper function
        let start_gpu = std::time::Instant::now();
        let actual_values_gpu = interpolate_coset_with_precomputation_gpu(
            &coset_evals_gpu,
            shift,
            point,
            &coset_gpu,
            &diff_invs_gpu,
        );
        println!("GPU computation finished in {:?}", start_gpu.elapsed());

        // 5. COMPARE RESULTS
        println!("Comparing results...");
        assert_eq!(
            expected_values.len(),
            actual_values_gpu.len(),
            "Result vectors have different lengths!"
        );

        // Check if the results are close enough, accounting for potential minor floating point differences
        // if a different field were used. For exact fields, direct equality is fine.
        let tolerance = Val::ZERO; // For exact fields, we expect perfect equality
        let mut mismatch_found = false;
        for i in 0..expected_values.len() {
            if expected_values[i] != actual_values_gpu[i] {
                println!("\n==================== MISMATCH FOUND ====================");
                println!("Mismatch at result index {}", i);
                println!("  Expected (CPU): {:?}", expected_values[i]);
                println!("  Actual (GPU):   {:?}", actual_values_gpu[i]);
                println!("========================================================");
                mismatch_found = true;
                // Don't break, let's see all mismatches if there are a few
            }
        }

        if mismatch_found {
            panic!("GPU interpolation results do not match CPU reference.");
        }

        println!("\nSUCCESS: interpolate_coset_with_precomputation_gpu is correct.");
    }
}
