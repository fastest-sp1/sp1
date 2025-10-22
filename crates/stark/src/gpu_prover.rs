//! A GPU-accelerated implementation of the MachineProver trait.
//!
//! This prover manages a "GPU-first" data pipeline, where large datasets
//! (traces, LDEs) are generated on and remain in GPU memory to minimize
//! costly host-device data transfers.

use crate::gpu::matrix::{GpuMatrix, GpuMatrixC};

use crate::{
    AirOpenedValues, ChipOpenedValues, CudaResultCheck, GpuMemBlk, GpuMerkleProverData, GpuPcs,
    MachineProvingKey, ShardCommitment, ShardMainData, ShardOpenedValues, ShardProof,
    StarkVerifyingKey, Val,
    air::{MachineAir, MachineProgram},
};
use crate::{
    DebugConstraintBuilder, MachineChip, MachineProof,
    air::InteractionScope,
    baby_bear_poseidon2::{Challenge, Challenger},
    generate_perm_trace,
    record::MachineRecord,
};

use super::{
    Com, InnerChallenge, InnerVal, StarkGenericConfig, StarkMachine, StarkProvingKey,
    quotient_values_data_in_gpu,
};

use crate::{
    PROOF_MAX_NUM_PVS, count_permutation_constraints, septic_digest::SepticDigest, split_matrix_gpu,
};
use hashbrown::HashMap;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{
    BasedVectorSpace, PrimeCharacteristicRing, PrimeField32, coset::TwoAdicMultiplicativeCoset,
    extension::BinomialExtensionField,
};
use p3_matrix::{Dimensions, Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{SymbolicAirBuilder, get_symbolic_constraints};
use p3_util::log2_strict_usize;

use crate::DIGEST_SIZE;
use std::{cmp::Reverse, error::Error, fmt::Debug};

use itertools::Itertools;

pub trait AbstractMatrix: Send + Sync {
    // We can add common methods here later if needed, e.g., dimensions().
}
impl<T: Send + Sync> AbstractMatrix for RowMajorMatrix<T> {}
impl<T: Send + Sync> AbstractMatrix for GpuMatrix<T> {}

/// An algorithmic & hardware independent prover implementation for any [`MachineAir`].
pub trait GpuMachineProver<SC: StarkGenericConfig, A: MachineAir<SC::Val>>:
    'static + Send + Sync
{
    /// The type used to store the traces.
    type DeviceMatrix;

    /// The type used to store the polynomial commitment schemes data.
    type DeviceProverData;

    /// The type used to store the proving key.
    type DeviceProvingKey: MachineProvingKey<SC>;

    /// The type used for error handling.
    type Error: Error + Send + Sync;

    /// Create a new prover from a given machine.
    fn new(machine: StarkMachine<SC, A>) -> Self;

    /// A reference to the machine that this prover is using.
    fn machine(&self) -> &StarkMachine<SC, A>;

    /// Setup the preprocessed data into a proving and verifying key.
    fn setup(
        &self,
        program: &A::Program,
        gpu_mem_blk: &GpuMemBlk,
    ) -> (Self::DeviceProvingKey, StarkVerifyingKey<SC>);

    /// Setup the proving key given a verifying key. This is similar to `setup` but faster since
    /// some computed information is already in the verifying key.
    fn pk_from_vk(
        &self,
        program: &A::Program,
        vk: &StarkVerifyingKey<SC>,
    ) -> Self::DeviceProvingKey;

    /// Copy the proving key from the host to the device.
    fn pk_to_device(
        &self,
        pk: &StarkProvingKey<SC>,
        gpu_mem_blk: &GpuMemBlk,
    ) -> Self::DeviceProvingKey;

    /// Copy the proving key from the device to the host.
    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> StarkProvingKey<SC>;

    /// Generate the main traces.
    //fn generate_traces(&self, record: &A::Record) -> Vec<(String, GpuMatrix<Val<SC>>)>;
    /// Generates the main execution traces for each chip directly on the GPU.
    fn generate_traces(
        &self,
        record: &A::Record,
        gpu_mem_blk: &GpuMemBlk,
    ) -> Vec<(String, GpuMatrix<Val<SC>>)> {
        let shard_chips = self.shard_chips(record).collect::<Vec<_>>();
        // For each chip, generate the trace on the GPU.
        let mut named_gpu_traces: Vec<(String, GpuMatrix<Val<SC>>)> = shard_chips
            .par_iter() // Can still use Rayon for launching GPU work in parallel
            //.iter()
            .map(|chip| {
                let chip_name = chip.name();
                // This new method must be implemented on the MachineChip trait.
                let trace = chip.generate_trace_gpu(record, &mut A::Record::default());
                let gpu_trace = gpu_mem_blk.alloc_matrix::<SC::Val>(trace.height(), trace.width());
                gpu_trace.copy_from_host(&trace.values);

                (chip_name, gpu_trace)
            })
            .collect();

        //    We sort by height in descending order (biggest first), and then by chip name
        //    alphabetically as a tie-breaker.
        named_gpu_traces.sort_by_key(|(name, trace)| (Reverse(trace.height), name.clone()));

        named_gpu_traces
    }

    /// Commit to the main traces.
    fn commit(
        &self,
        record: &A::Record,
        traces: Vec<(String, GpuMatrix<Val<SC>>)>,
        gpu_mem_blk: &GpuMemBlk,
    ) -> ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>;

    /// Observe the main commitment and public values and update the challenger.
    fn observe(
        &self,
        challenger: &mut SC::Challenger,
        commitment: Com<SC>,
        public_values: &[SC::Val],
    ) {
        // Observe the commitment.
        challenger.observe(commitment);

        // Observe the public values.
        challenger.observe_slice(public_values);
    }

    /// Compute the openings of the traces.
    fn open(
        &self,
        pk: &Self::DeviceProvingKey,
        data: ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>,
        challenger: &mut SC::Challenger,
        gpu_mem_blk: &GpuMemBlk,
    ) -> Result<ShardProof<SC>, Self::Error>;

    /// Generate a proof for the given records.
    fn prove(
        &self,
        pk: &Self::DeviceProvingKey,
        records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
        opts: <A::Record as MachineRecord>::Config,
        gpu_mem_blk: &GpuMemBlk,
    ) -> Result<MachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>;

    /// The stark config for the machine.
    fn config(&self) -> &SC {
        self.machine().config()
    }

    /// The number of public values elements.
    fn num_pv_elts(&self) -> usize {
        self.machine().num_pv_elts()
    }

    /// The chips that will be necessary to prove this record.
    fn shard_chips<'a, 'b>(
        &'a self,
        record: &'b A::Record,
    ) -> impl Iterator<Item = &'b MachineChip<SC, A>>
    where
        'a: 'b,
        SC: 'b,
    {
        self.machine().shard_chips(record)
    }

    /// Debug the constraints for the given inputs.
    fn debug_constraints(
        &self,
        pk: &GpuProvingKey<SC>,
        records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
    ) where
        SC::Val: PrimeField32,
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>;
}

/// A proving key for a STARK where preprocessed traces are stored on the GPU.
// NOTE: We derive Clone, but this will perform a device-to-device copy of the GPU matrices,

pub struct GpuProvingKey<SC: StarkGenericConfig> {
    /// The commitment to the preprocessed traces (remains on CPU).
    pub commit: Com<SC>,
    /// The start pc of the program (CPU).
    pub pc_start: Val<SC>,
    /// The starting global digest of the program (CPU).
    pub initial_global_cumulative_sum: SepticDigest<Val<SC>>,

    /// The preprocessed traces, stored as handles to GPU memory.
    pub traces: Vec<GpuMatrix<Val<SC>>>,

    /// The PCS data for the preprocessed traces (contains GPU MMCS handles).
    pub data: GpuMerkleProverData<GpuMatrix<SC::Val>>, //PcsProverData<SC>,

    // Metadata remains on the CPU.
    pub chip_ordering: HashMap<String, usize>,
    pub local_only: Vec<bool>,
    pub constraints_map: HashMap<String, usize>,
}

/// Implements the generic `MachineProvingKey` trait for our GPU-specific key.
/// This allows it to be used by higher-level generic components.
impl<SC: StarkGenericConfig> MachineProvingKey<SC> for GpuProvingKey<SC>
where
    // Add necessary bounds for trait methods
    Com<SC>: Send + Sync,
    //PcsProverData<SC>: Send + Sync,
{
    fn preprocessed_commit(&self) -> Com<SC> {
        self.commit.clone()
    }

    fn pc_start(&self) -> Val<SC> {
        self.pc_start
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<Val<SC>> {
        self.initial_global_cumulative_sum
    }

    fn observe_into(&self, challenger: &mut SC::Challenger) {
        // This logic is identical to the CPU version as it only deals with CPU data.
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        challenger.observe(Val::<SC>::ZERO);
    }
}

// An enum to pass to the GPU, matching the one in your CUDA code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuChipId {
    BaseAlu = 0,
    ExtAlu = 1,
    BatchFRI = 2,
    ExpReverseBitsLen = 3,
    FriFold = 4,
    PublicValues = 5,
    Select = 6,
    Poseidon2WideDeg3 = 7,
    Poseidon2SkinnyDeg9 = 8,
    MemoryConst = 9,
    MemoryVar = 10,
    // Add any others if needed
}

// Helper function to map name to ID
fn map_chip_name_to_gpu_id(name: &str) -> Option<GpuChipId> {
    match name {
        "Poseidon2SkinnyDeg9" => Some(GpuChipId::Poseidon2SkinnyDeg9),
        "Poseidon2WideDeg3" => Some(GpuChipId::Poseidon2WideDeg3),
        "BaseAlu" => Some(GpuChipId::BaseAlu),
        "ExtAlu" => Some(GpuChipId::ExtAlu),
        "BatchFRI" => Some(GpuChipId::BatchFRI),
        "ExpReverseBitsLen" => Some(GpuChipId::ExpReverseBitsLen),
        "FriFold" => Some(GpuChipId::FriFold),
        "PublicValues" => Some(GpuChipId::PublicValues),
        "Select" => Some(GpuChipId::Select),
        "MemoryConst" => Some(GpuChipId::MemoryConst),
        "MemoryVar" => Some(GpuChipId::MemoryVar),
        _ => {
            println!(" does not support chip:{:?}", name);
            None
        }
    }
}

// A new error type for the GPU prover.
#[derive(Debug)]
pub enum GpuProverError {
    // ... define specific error types if needed, e.g., FFIError(i32)
}
impl std::error::Error for GpuProverError {}
impl std::fmt::Display for GpuProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU Prover Error")
    }
}

/// The GPU-accelerated prover.
pub struct GpuProver<SC: StarkGenericConfig, A> {
    pub machine: StarkMachine<SC, A>,
}

// Main implementation of the MachineProver trait for our GpuProver.
impl<SC, A> GpuMachineProver<SC, A> for GpuProver<SC, A>
where
    SC: StarkGenericConfig<
            Val = InnerVal,
            Challenge = Challenge,
            Challenger = Challenger,
            Pcs = GpuPcs,
            Domain = <GpuPcs as Pcs<Challenge, Challenger>>::Domain,
        >,
    SC::Val: PrimeField32,
    Com<SC>: Send + Sync,
    A: MachineAir<InnerVal> + Air<SymbolicAirBuilder<InnerVal>>,
{
    // Define the associated types to use our new GPU-centric structs.
    type DeviceMatrix = GpuMatrix<SC::Val>;
    type DeviceProverData = GpuMerkleProverData<GpuMatrix<SC::Val>>; //GpuPcs::ProverData; // This will be  GpuMmcsProverData
    type DeviceProvingKey = GpuProvingKey<SC>;
    type Error = GpuProverError;

    fn new(machine: StarkMachine<SC, A>) -> Self {
        Self {
            machine,
            //  _phantom_sc: PhantomData,
        }
    }

    fn machine(&self) -> &StarkMachine<SC, A> {
        &self.machine
    }

    /// Setup generates the preprocessed traces and commits to them, keeping data on the GPU.
    fn setup(
        &self,
        program: &A::Program,
        gpu_mem_blk: &GpuMemBlk,
    ) -> (GpuProvingKey<SC>, StarkVerifyingKey<SC>) {
        // 1. Generate preprocessed traces directly on the GPU.
        let (mut named_preprocessed_traces, num_constraints): (Vec<_>, Vec<_>) = self
            .machine
            .chips()
            .par_iter()
            //.iter()
            .map(|chip| {
                let trace = chip.generate_preprocessed_trace_gpu(program);
                let prep_trace = trace.unwrap();

                let gpu_trace = GpuMatrix::<SC::Val>::from_vec(
                    &prep_trace.values,
                    prep_trace.height(),
                    prep_trace.width(),
                    gpu_mem_blk,
                );

                // Count the number of constraints.
                let num_main_constraints = get_symbolic_constraints(
                    &chip.air,
                    chip.preprocessed_width(),
                    PROOF_MAX_NUM_PVS,
                )
                .len();

                let num_permutation_constraints = count_permutation_constraints(
                    &chip.sends,
                    &chip.receives,
                    chip.logup_batch_size(),
                    chip.air.commit_scope(),
                );

                (
                    //gpu_trace.map(move |t| (chip.name(), chip.local_only(), t)),
                    (chip.name(), chip.local_only(), gpu_trace),
                    (chip.name(), num_main_constraints + num_permutation_constraints),
                )
            })
            .unzip();

        // 2. Sort by height (descending) for the PCS.

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_preprocessed_traces
            .sort_by_key(|(name, _, trace)| (Reverse(trace.height), name.clone()));

        // 3. Create domains and commit on the GPU.
        let pcs = self.machine.config().pcs();
        let (chip_information, domains_and_traces): (Vec<_>, Vec<_>) = named_preprocessed_traces
            .iter()
            .map(|(name, _, trace)| {
                let domain = pcs.natural_domain_for_degree(trace.height);
                let dimensions = Dimensions { width: trace.width, height: trace.height };
                let vk_info = (name.to_owned(), domain, dimensions);
                let commit_info = (domain, trace.clone()); // Clone GpuMatrix handle
                //let commit_info = (domain, trace);
                (vk_info, commit_info)
            })
            .unzip();

        let (commit, data) = GpuPcs::commit_gpu(&pcs, domains_and_traces, gpu_mem_blk);

        // 4. Collect metadata for the keys.
        // Get the chip ordering.
        let chip_ordering = named_preprocessed_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _, _))| (name.to_owned(), i))
            .collect::<HashMap<_, _>>();

        let local_only = named_preprocessed_traces
            .iter()
            .map(|(_, local_only, _)| local_only.to_owned())
            .collect::<Vec<_>>();

        let constraints_map: HashMap<_, _> = num_constraints.into_iter().collect();

        // Get the preprocessed traces
        let traces =
            named_preprocessed_traces.into_iter().map(|(_, _, trace)| trace).collect::<Vec<_>>();

        let gpu_pk = GpuProvingKey {
            commit: commit.clone(),
            pc_start: program.pc_start(),
            initial_global_cumulative_sum: program.initial_global_cumulative_sum(),
            traces,
            data,
            chip_ordering: chip_ordering.clone(),
            local_only,
            constraints_map,
        };

        let vk = StarkVerifyingKey {
            commit,
            pc_start: program.pc_start(),
            initial_global_cumulative_sum: program.initial_global_cumulative_sum(),
            chip_information,
            chip_ordering,
        };

        (gpu_pk, vk)
    }

    /// Converts a CPU-based StarkProvingKey to a GPU-based GpuProvingKey.
    fn pk_to_device(&self, pk: &StarkProvingKey<SC>, gpu_mem_blk: &GpuMemBlk) -> GpuProvingKey<SC> {
        // 1. Transfer preprocessed traces from CPU Vecs to GPU GpuMatrices.
        let gpu_traces: Vec<GpuMatrix<SC::Val>> = pk
            .traces
            .par_iter()
            .map(|cpu_trace| {
                GpuMatrix::from_vec(
                    &cpu_trace.values,
                    cpu_trace.height(),
                    cpu_trace.width(),
                    gpu_mem_blk,
                )
            })
            .collect();

        // 2. The `data` field (PcsProverData) from the CPU key contains CPU MMCS data.
        // We need to re-commit on the GPU to get a GPU MMCS handle.
        // This is essentially a subset of the `setup` logic.
        let pcs = self.machine.config().pcs();
        let domains_and_gpu_traces = pk
            .traces
            .iter()
            .zip(&gpu_traces)
            .map(|(cpu_trace, gpu_trace): (&RowMajorMatrix<SC::Val>, _)| {
                let domain = pcs.natural_domain_for_degree(cpu_trace.height());
                (domain, gpu_trace.clone())
            })
            .collect();

        let (_commit, gpu_data) = GpuPcs::commit_gpu(&pcs, domains_and_gpu_traces, gpu_mem_blk); //pcs.commit_gpu(domains_and_gpu_traces);

        // 3. Assemble the GpuProvingKey.
        GpuProvingKey {
            commit: pk.commit.clone(),
            pc_start: pk.pc_start,
            initial_global_cumulative_sum: pk.initial_global_cumulative_sum,
            traces: gpu_traces,
            data: gpu_data,
            chip_ordering: pk.chip_ordering.clone(),
            local_only: pk.local_only.clone(),
            constraints_map: pk.constraints_map.clone(),
        }
    }

    /// Converts a GPU-based GpuProvingKey back to a CPU-based StarkProvingKey.
    /// This is an expensive operation as it involves downloading all preprocessed traces.
    fn pk_to_host(&self, pk: &GpuProvingKey<SC>) -> StarkProvingKey<SC> {
        // 1. Download preprocessed traces from GPU to CPU.
        let cpu_traces: Vec<RowMajorMatrix<SC::Val>> = pk
            .traces
            .par_iter()
            .map(|gpu_trace| {
                let values = gpu_trace.to_host();
                RowMajorMatrix::<SC::Val>::new(values, gpu_trace.width)
            })
            .collect();

        // 2. Convert the GPU PcsProverData back to a CPU version. This is complex
        // and might not be possible without re-committing on the CPU side.
        // For now, let's assume we can create a dummy or placeholder CPU data object.
        // The correct way would be to commit `cpu_traces` with a CPU-based PCS.
        let pcs = self.machine.config().pcs();
        let domains_and_cpu_traces: Vec<_> = cpu_traces
            .iter()
            .map(|trace| (pcs.natural_domain_for_degree(trace.height()), trace.clone()))
            .collect();
        let (_commit, cpu_data) = pcs.commit(domains_and_cpu_traces);

        // 3. Assemble the StarkProvingKey.
        StarkProvingKey {
            commit: pk.commit.clone(),
            pc_start: pk.pc_start,
            initial_global_cumulative_sum: pk.initial_global_cumulative_sum,
            traces: cpu_traces,
            data: cpu_data,
            chip_ordering: pk.chip_ordering.clone(),
            local_only: pk.local_only.clone(),
            constraints_map: pk.constraints_map.clone(),
        }
    }

    /// Commits to the main traces which are already on the GPU.
    fn commit(
        &self,
        record: &A::Record,
        named_gpu_traces: Vec<(String, GpuMatrix<Val<SC>>)>,
        gpu_mem_blk: &GpuMemBlk,
    ) -> ShardMainData<SC, GpuMatrix<SC::Val>, GpuMerkleProverData<GpuMatrix<SC::Val>>> {
        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        //named_gpu_traces.sort_by_key(|(name, trace)| (Reverse(trace.height()), name.clone()));

        let pcs = self.machine.config().pcs();

        // The domains are calculated on the CPU, as they are small.
        let domains_and_gpu_traces = named_gpu_traces
            .iter()
            .map(|(_, gpu_trace)| {
                let domain = pcs.natural_domain_for_degree(gpu_trace.height);
                (domain, gpu_trace.clone())
            })
            .collect::<Vec<_>>();

        // This `commit_gpu` method needs to be added to your GpuPcs.
        // It takes GPU matrices as input.
        let (main_commit, main_data) =
            GpuPcs::commit_gpu(&pcs, domains_and_gpu_traces, gpu_mem_blk); //pcs.commit_gpu(domains_and_gpu_traces);

        let chip_ordering =
            named_gpu_traces.iter().enumerate().map(|(i, (name, _))| (name.clone(), i)).collect();

        let traces = named_gpu_traces.into_iter().map(|(_, trace)| trace.clone()).collect(); //must-have-clone-gpumatrix?

        ShardMainData {
            traces,
            main_commit,
            main_data,
            chip_ordering,
            public_values: record.public_values(),
        }
    }

    /// Orchestrates the entire opening phase on the GPU.
    fn open(
        &self,
        pk: &GpuProvingKey<SC>,
        data: ShardMainData<SC, GpuMatrix<SC::Val>, GpuMerkleProverData<GpuMatrix<SC::Val>>>,
        challenger: &mut SC::Challenger,
        gpu_mem_blk: &GpuMemBlk,
    ) -> Result<ShardProof<SC>, GpuProverError> {
        let chips = self.machine().shard_chips_ordered(&data.chip_ordering).collect::<Vec<_>>();
        let traces = data.traces; //GpuMatrix<Val>
        let config = self.machine().config();

        let degrees = traces.iter().map(|trace| trace.height).collect::<Vec<_>>();

        let log_degrees =
            degrees.iter().map(|degree| log2_strict_usize(*degree)).collect::<Vec<_>>();

        let log_quotient_degrees =
            chips.iter().map(|chip| chip.log_quotient_degree()).collect::<Vec<_>>();

        let pcs = config.pcs();
        //let pcs = self.config().pcs();
        let trace_domains =
            degrees.iter().map(|degree| pcs.natural_domain_for_degree(*degree)).collect::<Vec<_>>();

        // Observe the public values and the main commitment.
        challenger.observe_slice(&data.public_values[0..self.machine.num_pv_elts()]);
        challenger.observe(data.main_commit.clone());

        // Obtain the challenges used for the local permutation argument.
        let mut local_permutation_challenges: Vec<SC::Challenge> = Vec::new();
        for _ in 0..2 {
            local_permutation_challenges.push(challenger.sample_algebra_element());
        }

        // 1. Generate permutation traces & commit
        let ((gpu_perm_traces, _gpu_prep_traces), (local_cumulative_sums, global_cumulative_sums)): (
            (Vec<_>, Vec<_>),
            (Vec<_>, Vec<_>),
        )  = chips
        .par_iter() 
        //.iter()//debug
        .zip(traces)
        .map(|(chip, main_trace)| {
                let chip_name = chip.name();
                let preprocessed_trace =
                        pk.chip_ordering.get(&chip_name).map(|&index| &pk.traces[index]);
                // FFI call to generate permutation trace for one chip on GPU
         
                 let gpu_chip_id = map_chip_name_to_gpu_id(&chip_name)
                                    .expect("Chip name in GPU set but not in ID map");
                 let width = chip.permutation_width();
                 let batch_size = chip.logup_batch_size();
                 let (perm_trace, local_sum) = 
                    generate_perm_trace(gpu_chip_id as i32, width as i32, batch_size as i32,preprocessed_trace.unwrap(), &main_trace, &local_permutation_challenges, gpu_mem_blk);
                 let global_sum = if chip.commit_scope() == InteractionScope::Local {
                        SepticDigest::<Val<SC>>::zero()
                    } else {
                       
                        //SepticDigest::<Val<SC>>::zero()
                        println!("NOT SURPPORT CORE CHIPS!");
                        unimplemented!()
                    };
                ((perm_trace, preprocessed_trace), (local_sum, global_sum))
        }).collect();

        let domains_and_perm_traces = gpu_perm_traces
            .into_iter()
            .zip(trace_domains.iter())
            .map(|(perm_trace, domain)| {
                let flatten_trace = perm_trace.flatten_to_base(gpu_mem_blk);
                (*domain, flatten_trace)
            })
            .collect::<Vec<_>>();

        let pcs = config.pcs(); //?
        // Commit to all permutation traces at once
        let (permutation_commit, perm_data) = pcs.commit_gpu(domains_and_perm_traces, gpu_mem_blk);

        challenger.observe(permutation_commit.clone());
        for (local_sum, global_sum) in
            local_cumulative_sums.iter().zip(global_cumulative_sums.iter())
        {
            challenger.observe_slice(<BinomialExtensionField<SC::Val, 4> as BasedVectorSpace<SC::Val>>::as_basis_coefficients_slice(local_sum)); //(local_sum.as_basis_coefficients_slice());
            challenger.observe_slice(&global_sum.0.x.0);
            challenger.observe_slice(&global_sum.0.y.0);
        }

        // Compute the quotient polynomial for all chips.
        let quotient_domains = trace_domains
            .iter()
            .zip_eq(log_degrees.iter())
            .zip_eq(log_quotient_degrees.iter())
            .map(|((domain, log_degree), log_quotient_degree)| {
                domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree))
            })
            .collect::<Vec<_>>();

        let alpha: SC::Challenge = challenger.sample_algebra_element::<SC::Challenge>();
        // Compute quotient values
        let gpu_quotient_values: Vec<GpuMatrix<SC::Challenge>> = quotient_domains
            .clone()
            .into_par_iter() //debug
            //.into_iter()
            .enumerate()
            .map(|(i, quotient_domain)| {
                // All matrix pointers are now device pointers.
                let chip_name = chips[i].name();
                let qd_size = quotient_domain.size();
                let gpu_chip_id = map_chip_name_to_gpu_id(&chip_name)
                    .expect("Chip name in GPU set but not in ID map");
                let qdb = log2_strict_usize(qd_size) - log2_strict_usize(trace_domains[i].size());
                let next_step = 1 << qdb;

                let batch_size = chips[i].logup_batch_size();

                let chip_num_constraints = pk.constraints_map.get(&chip_name).unwrap();

                // Calculate powers of alpha for constraint evaluation:
                // 1. Generate sequence [α⁰, α¹, ..., α^(n-1)] where n = chip_num_constraints.
                // 2. Reverse to [α^(n-1), ..., α¹, α⁰] to align with Horner's method in the verifier.
                let powers_of_alpha =
                    alpha.powers().take(*chip_num_constraints).collect::<Vec<_>>();
                let mut powers_of_alpha_rev = powers_of_alpha.clone();
                powers_of_alpha_rev.reverse();

                let mut public_values_digest_slice: &[SC::Val] = &[];

                if gpu_chip_id == GpuChipId::PublicValues {
                    //public values chip
                    let pv_len = data.public_values.len();
                    public_values_digest_slice = &data.public_values[pv_len - DIGEST_SIZE..];
                }

                let prep_lde_gpu = pk
                    .chip_ordering
                    .get(&chip_name)
                    .map(|&index| pcs.get_lde_on_domain_gpu(&pk.data, index));

                let main_lde_gpu = pcs.get_lde_on_domain_gpu(&data.main_data, i);
                let perm_lde_gpu = pcs.get_lde_on_domain_gpu(&perm_data, i);

                let trace_gen = trace_domains[i].subgroup_generator();
                let coset_shift = quotient_domain.shift();
                let coset_gen = quotient_domain.subgroup_generator();

                let alpha_offset = 0; //?

                // Allocate output buffer for GPU results
                let gpu_quotients = vec![SC::Challenge::ZERO; qd_size];
                let gpu_matrix_quotient =
                    GpuMatrix::<SC::Challenge>::from_vec(&gpu_quotients, qd_size, 1, gpu_mem_blk);

                let trace_domain_coset_log_size = trace_domains[i].log_size();
                let quotient_domain_coset_log_size = quotient_domain.log_size();

                //Call the GPU FFI function
                //Notice: if supporting core chips, some prep trace is none!
                unsafe {
                    // Correctly handle pointers to single items passed by reference
                    let local_sum_ptr = &local_cumulative_sums[i] as *const SC::Challenge;
                    let global_sum_ptr = &global_cumulative_sums[i] as *const SepticDigest<SC::Val>;

                    let main_lde_gpu_c: GpuMatrixC = (&main_lde_gpu).into();

                    // must clone prep_lde_gpu, otherwise prep_lde_gpu will release the gpu mem  at once
                    // and  this will cause  prep_lde_gpu_c.ptr is invalid.
                    let prep_lde_gpu_c: GpuMatrixC = (&prep_lde_gpu.clone().unwrap()).into();
                    let perm_lde_gpu_c: GpuMatrixC = (&perm_lde_gpu).into();

                    let _ = quotient_values_data_in_gpu(
                        gpu_chip_id as i32,
                        &main_lde_gpu_c, //  as *const GpuMatrix<InnerVal>,
                        &prep_lde_gpu_c, //  as *const GpuMatrix<InnerVal>,
                        &perm_lde_gpu_c, // as *const GpuMatrix<InnerVal>,
                        powers_of_alpha_rev.as_ptr() as *const InnerChallenge,
                        *chip_num_constraints as i32,
                        qd_size as i32,
                        next_step as i32,
                        batch_size as i32,
                        local_permutation_challenges.as_ptr() as *const InnerChallenge,
                        local_sum_ptr as *const InnerChallenge,
                        public_values_digest_slice.as_ptr() as *const InnerVal,
                        public_values_digest_slice.len() as i32,
                        global_sum_ptr as *const SepticDigest<InnerVal>,
                        alpha_offset as i32,
                        trace_domain_coset_log_size as i32,
                        quotient_domain_coset_log_size as i32,
                        trace_gen.into(),
                        coset_shift.into(),
                        coset_gen.into(),
                        gpu_matrix_quotient.as_mut_ptr(),
                    )
                    .check("cpu_prover quotient_values_data_in_gpu() failed.");
                }

                gpu_matrix_quotient
            })
            .collect();

        // Split the quotient values and commit to them.
        let quotient_domains_and_chunks = quotient_domains
            .into_iter()
            .zip(gpu_quotient_values) // `gpu_quotient_values` is Vec<GpuMatrix<Challenge>>
            .zip(log_quotient_degrees.iter())
            .flat_map(
                |((quotient_domain, q_vals_chall), log_quotient_degree): (
                    (TwoAdicMultiplicativeCoset<SC::Val>, GpuMatrix<SC::Challenge>),
                    &usize,
                )| {
                    let quotient_degree = 1 << *log_quotient_degree;

                    // 1. Flatten from Challenge to Val on GPU
                    let q_vals_flat: GpuMatrix<SC::Val> = q_vals_chall.flatten_to_base(gpu_mem_blk);

                    // 2. Split the domains on the CPU (fast)
                    let qc_domains = quotient_domain.split_domains(quotient_degree);

                    // 3. Split the evaluations on the GPU
                    // a) Allocate output GpuMatrix handles for the chunks
                    let chunk_height = q_vals_flat.height / quotient_degree;
                    let chunk_width = q_vals_flat.width;
                    let output_chunks: Vec<GpuMatrix<SC::Val>> = (0..quotient_degree)
                        .map(|_| {
                            GpuMatrix::<SC::Val>::new(chunk_height, chunk_width, gpu_mem_blk).into()
                        })
                        .collect();

                    let output_chunks_c: Vec<GpuMatrixC> = output_chunks
                        .iter()
                        .map(|gpu_matrix| gpu_matrix.into()) // `into()` calls the `From<&GpuMatrix>` impl
                        .collect();

                    // c) Call the FFI function to perform the split on the GPU
                    unsafe {
                        let q_vals_flat_c: GpuMatrixC = (&q_vals_flat).into();
                        //let q_vals_flat_c: GpuMatrixC = (q_vals_flat).into();
                        let _ = split_matrix_gpu(
                            &q_vals_flat_c,           // as *const GpuMatrix<SC::Val>,
                            output_chunks_c.as_ptr(), // as *const GpuMatrix<SC::Val>,
                            quotient_degree as i32,
                        )
                        .check("split_matrix_gpu failed");
                    }

                    // 4. Zip the CPU domains with the GPU matrix handles
                    qc_domains.into_iter().zip(output_chunks)
                },
            )
            .collect::<Vec<_>>();

        let num_quotient_chunks = quotient_domains_and_chunks.len();
        assert_eq!(
            num_quotient_chunks,
            chips.iter().map(|c| 1 << c.log_quotient_degree()).sum::<usize>()
        );

        // Now `quotient_domains_and_chunks` is a Vec<(Domain, GpuMatrix<Val>)>
        // and can be passed to `pcs.commit_gpu`.
        let (quotient_commit, quotient_data) =
            pcs.commit_gpu(quotient_domains_and_chunks, gpu_mem_blk);
        challenger.observe(quotient_commit.clone());

        // Compute the quotient argument.
        let zeta: SC::Challenge = challenger.sample_algebra_element();

        let preprocessed_opening_points = pk
            .traces
            .iter()
            .zip(pk.local_only.iter())
            .map(|(trace, local_only)| {
                let domain = pcs.natural_domain_for_degree(trace.height);
                if !local_only { vec![zeta, domain.next_point(zeta).unwrap()] } else { vec![zeta] }
            })
            .collect::<Vec<_>>();

        let main_trace_opening_points = trace_domains
            .iter()
            .zip(chips.iter())
            .map(|(domain, chip)| {
                if !chip.local_only() {
                    vec![zeta, domain.next_point(zeta).unwrap()]
                } else {
                    vec![zeta]
                }
            })
            .collect::<Vec<_>>();

        let permutation_trace_opening_points = trace_domains
            .iter()
            .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
            .collect::<Vec<_>>();

        // Compute quotient opening points, open every chunk at zeta.
        let quotient_opening_points =
            (0..num_quotient_chunks).map(|_| vec![zeta]).collect::<Vec<_>>();

        let (openings, opening_proof) = pcs.open_gpu(
            vec![
                (&pk.data, preprocessed_opening_points),
                (&data.main_data, main_trace_opening_points.clone()),
                (&perm_data, permutation_trace_opening_points.clone()),
                (&quotient_data, quotient_opening_points),
            ],
            challenger,
            gpu_mem_blk,
        );

        // Collect the opened values for each chip.
        let [preprocessed_values, main_values, permutation_values, mut quotient_values] =
            openings.try_into().unwrap();
        assert!(main_values.len() == chips.len());
        let preprocessed_opened_values = preprocessed_values
            .into_iter()
            .zip(pk.local_only.iter())
            .map(|(op, local_only)| {
                if !local_only {
                    let [local, next] = op.try_into().unwrap();
                    AirOpenedValues { local, next }
                } else {
                    let [local] = op.try_into().unwrap();
                    let width = local.len();
                    AirOpenedValues { local, next: vec![SC::Challenge::ZERO; width] }
                }
            })
            .collect::<Vec<_>>();

        let main_opened_values = main_values
            .into_iter()
            .zip(chips.iter())
            .map(|(op, chip)| {
                if !chip.local_only() {
                    let [local, next] = op.try_into().unwrap();
                    AirOpenedValues { local, next }
                } else {
                    let [local] = op.try_into().unwrap();
                    let width = local.len();
                    AirOpenedValues { local, next: vec![SC::Challenge::ZERO; width] }
                }
            })
            .collect::<Vec<_>>();
        let permutation_opened_values = permutation_values
            .into_iter()
            .map(|op| {
                let [local, next] = op.try_into().unwrap();
                AirOpenedValues { local, next }
            })
            .collect::<Vec<_>>();
        let mut quotient_opened_values = Vec::with_capacity(log_quotient_degrees.len());
        for log_quotient_degree in log_quotient_degrees.iter() {
            let degree = 1 << *log_quotient_degree;
            let slice = quotient_values.drain(0..degree);
            quotient_opened_values.push(slice.map(|mut op| op.pop().unwrap()).collect::<Vec<_>>());
        }

        let opened_values = main_opened_values
            .into_iter()
            .zip_eq(permutation_opened_values)
            .zip_eq(quotient_opened_values)
            .zip_eq(local_cumulative_sums)
            .zip_eq(global_cumulative_sums)
            .zip_eq(log_degrees.iter())
            .enumerate()
            .map(
                |(
                    i,
                    (
                        (
                            (((main, permutation), quotient), local_cumulative_sum),
                            global_cumulative_sum,
                        ),
                        log_degree,
                    ),
                )| {
                    let preprocessed = pk
                        .chip_ordering
                        .get(&chips[i].name())
                        .map(|&index| preprocessed_opened_values[index].clone())
                        .unwrap_or(AirOpenedValues { local: vec![], next: vec![] });
                    ChipOpenedValues {
                        preprocessed,
                        main,
                        permutation,
                        quotient,
                        global_cumulative_sum,
                        local_cumulative_sum,
                        log_degree: *log_degree,
                    }
                },
            )
            .collect::<Vec<_>>();

        Ok(ShardProof::<SC> {
            commitment: ShardCommitment {
                main_commit: data.main_commit.clone(),
                permutation_commit,
                quotient_commit,
            },
            opened_values: ShardOpenedValues { chips: opened_values },
            opening_proof,
            chip_ordering: data.chip_ordering,
            public_values: data.public_values,
        })
    }

    fn prove(
        &self,
        pk: &GpuProvingKey<SC>,
        mut records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
        opts: <A::Record as MachineRecord>::Config,
        gpu_mem_blk: &GpuMemBlk,
    ) -> Result<MachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        // Generate dependencies.
        self.machine().generate_dependencies(&mut records, &opts, None);

        // Observe the preprocessed commitment.
        pk.observe_into(challenger);

        let shard_proofs = tracing::info_span!("prove_shards").in_scope(|| {
            records
                .into_par_iter()
                .map(|record| {
                    let named_traces = self.generate_traces(&record, gpu_mem_blk);
                    let shard_data = self.commit(&record, named_traces, gpu_mem_blk);
                    self.open(pk, shard_data, &mut challenger.clone(), gpu_mem_blk)
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        Ok(MachineProof { shard_proofs })
    }

    fn pk_from_vk(
        &self,
        _program: &A::Program,
        _vk: &StarkVerifyingKey<SC>,
    ) -> Self::DeviceProvingKey {
        unimplemented!()
    }

    fn debug_constraints(
        &self,
        _pk: &GpuProvingKey<SC>,
        _records: Vec<A::Record>,
        _challenger: &mut SC::Challenger,
    ) where
        SC::Val: PrimeField32,
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        unimplemented!()
    }
}
