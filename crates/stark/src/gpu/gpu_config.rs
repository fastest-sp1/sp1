// Use statements should be updated to include what's needed.
use crate::baby_bear_poseidon2::{
    BabyBearPoseidon2Type, Challenge, Challenger, DigestHash, MyCompress, MyHash, Perm,
    StarkConfigCpu, Val, compressed_fri_config, default_fri_config, my_perm,
};
use crate::gpu::{merkle::GpuMerkleTreeMmcs, pcs::GpuFriPcs};

use crate::{
    DIGEST_SIZE, InnerDft, StarkGenericConfig,
    config::{Com, ZeroCommitment},
};
use p3_commit::ExtensionMmcs;

use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::FriConfig;

use serde::{Deserialize, Serialize};

// Helper types remain mostly the same
pub type GpuValMmcs =
    GpuMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
pub type GpuChallengeMmcs = ExtensionMmcs<Val, Challenge, GpuValMmcs>;

//pub type GpuPcs = GpuFriPcs<Val, GpuDft, GpuValMmcs, GpuChallengeMmcs>; //GpuDft performace
pub type GpuPcs = GpuFriPcs<Val, InnerDft, GpuValMmcs, GpuChallengeMmcs>;

#[derive(Deserialize)]
#[serde(from = "std::marker::PhantomData<StarkConfigGpu>")]
pub struct StarkConfigGpu {
    pub perm: Perm,

    #[serde(skip)]
    pub pcs: GpuPcs,

    pub challenger: Challenger,
    pub config_type: BabyBearPoseidon2Type,
}

/// Implement serialization manually instead of using serde to avoid cloing the config.
impl Serialize for StarkConfigGpu {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        std::marker::PhantomData::<StarkConfigGpu>.serialize(serializer)
    }
}

impl From<std::marker::PhantomData<StarkConfigGpu>> for StarkConfigGpu {
    fn from(_: std::marker::PhantomData<StarkConfigGpu>) -> Self {
        Self::new()
    }
}

impl Clone for StarkConfigGpu {
    fn clone(&self) -> Self {
        match self.config_type {
            BabyBearPoseidon2Type::Default => Self::new(),
            BabyBearPoseidon2Type::Compressed => Self::compressed(),
        }
    }
}

impl StarkConfigGpu {
    pub fn new() -> Self {
        let perm = my_perm();
        let challenger = Challenger::new(perm.clone());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        //let dft = GpuDft::default(); //for performace
        let dft = InnerDft::default();
        let val_mmcs = GpuValMmcs::new(hash, compress);
        let challenge_mmcs = GpuChallengeMmcs::new(val_mmcs.clone());

        let fri_config = default_fri_config();
        let gpu_fri_config = FriConfig {
            log_blowup: fri_config.log_blowup,
            log_final_poly_len: fri_config.log_final_poly_len,
            num_queries: fri_config.num_queries,
            proof_of_work_bits: fri_config.proof_of_work_bits,
            mmcs: challenge_mmcs,
        };

        let pcs = GpuPcs::new(dft, val_mmcs, gpu_fri_config);

        Self { perm, pcs, challenger, config_type: BabyBearPoseidon2Type::Default }
    }

    #[must_use]
    pub fn compressed() -> Self {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenger = Challenger::new(perm.clone());

        //let dft = GpuDft::default(); //for performace
        let dft = InnerDft::default();
        let val_mmcs = GpuValMmcs::new(hash, compress);
        let challenge_mmcs = GpuChallengeMmcs::new(val_mmcs.clone());

        let fri_config = compressed_fri_config();
        let gpu_fri_config = FriConfig {
            log_blowup: fri_config.log_blowup,
            log_final_poly_len: fri_config.log_final_poly_len,
            num_queries: fri_config.num_queries,
            proof_of_work_bits: fri_config.proof_of_work_bits,
            mmcs: challenge_mmcs,
        };

        let pcs = GpuPcs::new(dft, val_mmcs, gpu_fri_config);
        Self { perm, pcs, challenger, config_type: BabyBearPoseidon2Type::Compressed }
    }
}

impl Default for StarkConfigGpu {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroCommitment<StarkConfigGpu> for GpuPcs {
    fn zero_commitment(&self) -> Com<StarkConfigGpu> {
        DigestHash::from([Val::ZERO; DIGEST_SIZE])
    }
}

impl StarkGenericConfig for StarkConfigGpu {
    type Val = Val;
    type Challenge = Challenge;
    type Pcs = GpuPcs;
    type Challenger = Challenger;

    // We can just reuse the one from the CPU config.
    type Domain = <StarkConfigCpu as StarkGenericConfig>::Domain;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Self::Challenger {
        self.challenger.clone()
    }
}
