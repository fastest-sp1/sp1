#![allow(missing_docs)]

use crate::{Com, StarkGenericConfig, ZeroCommitment};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::{BatchOpening, ExtensionMmcs};
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing, Field};
use p3_fri::{
    CommitPhaseProofStep, FriConfig, FriProof, QueryProof, TwoAdicFriPcs,
    TwoAdicFriPcsProof,
};
use p3_merkle_tree::MerkleTreeMmcs;
//use p3_poseidon2::Poseidon2;
use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};
use serde::{Deserialize, Serialize};
use sp1_primitives::poseidon2_init;

pub const DIGEST_SIZE: usize = 8;

/// A configuration for inner recursion.
pub type InnerVal = BabyBear;
pub type InnerChallenge = BinomialExtensionField<InnerVal, 4>;
pub type InnerPerm = Poseidon2BabyBear<16>;
pub type InnerHash = PaddingFreeSponge<InnerPerm, 16, 8, DIGEST_SIZE>;
pub type InnerDigestHash = Hash<InnerVal, InnerVal, DIGEST_SIZE>;
pub type InnerDigest = [InnerVal; DIGEST_SIZE];
pub type InnerCompress = TruncatedPermutation<InnerPerm, 2, 8, 16>;
pub type InnerValMmcs = MerkleTreeMmcs<
    <InnerVal as Field>::Packing,
    <InnerVal as Field>::Packing,
    InnerHash,
    InnerCompress,
    8,
>;
pub type InnerChallengeMmcs = ExtensionMmcs<InnerVal, InnerChallenge, InnerValMmcs>;
pub type InnerChallenger = DuplexChallenger<InnerVal, InnerPerm, 16, 8>;
pub type InnerDft = Radix2DitParallel<InnerVal>;
pub type InnerPcs = TwoAdicFriPcs<InnerVal, InnerDft, InnerValMmcs, InnerChallengeMmcs>;
pub type InnerQueryProof = QueryProof<InnerChallenge, InnerChallengeMmcs>;
pub type InnerCommitPhaseStep = CommitPhaseProofStep<InnerChallenge, InnerChallengeMmcs>;
pub type InnerFriProof = FriProof<InnerChallenge, InnerChallengeMmcs, InnerVal>;
pub type InnerBatchOpening = BatchOpening<InnerVal, InnerValMmcs>;
pub type InnerPcsProof =
    TwoAdicFriPcsProof<InnerVal, InnerChallenge, InnerValMmcs, InnerChallengeMmcs>;

/// The permutation for inner recursion.
#[must_use]
pub fn inner_perm() -> InnerPerm {
    poseidon2_init()
}

/// The FRI config for sp1 proofs.
#[must_use]
pub fn sp1_fri_config() -> FriConfig<InnerChallengeMmcs> {
    let perm = inner_perm();
    let hash = InnerHash::new(perm.clone());
    let compress = InnerCompress::new(perm.clone());
    let challenge_mmcs = InnerChallengeMmcs::new(InnerValMmcs::new(hash, compress));
    let num_queries = match std::env::var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => 100,
    };
    FriConfig { log_blowup: 1, log_final_poly_len:0, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
}

/// The FRI config for inner recursion.
#[must_use]
pub fn inner_fri_config() -> FriConfig<InnerChallengeMmcs> {
    let perm = inner_perm();
    let hash = InnerHash::new(perm.clone());
    let compress = InnerCompress::new(perm.clone());
    let challenge_mmcs = InnerChallengeMmcs::new(InnerValMmcs::new(hash, compress));
    let num_queries = match std::env::var("FRI_QUERIES") {
        Ok(value) => value.parse().unwrap(),
        Err(_) => 100,
    };
    FriConfig { log_blowup: 1, log_final_poly_len:0, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
}

/// The recursion config used for recursive reduce circuit.
#[derive(Deserialize)]
#[serde(from = "std::marker::PhantomData<BabyBearPoseidon2Inner>")]
pub struct BabyBearPoseidon2Inner {
    pub perm: InnerPerm,
    pub pcs: InnerPcs,
    pub challenger: InnerChallenger,
}

impl Clone for BabyBearPoseidon2Inner {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl Serialize for BabyBearPoseidon2Inner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        std::marker::PhantomData::<BabyBearPoseidon2Inner>.serialize(serializer)
    }
}

impl From<std::marker::PhantomData<BabyBearPoseidon2Inner>> for BabyBearPoseidon2Inner {
    fn from(_: std::marker::PhantomData<BabyBearPoseidon2Inner>) -> Self {
        Self::new()
    }
}

impl BabyBearPoseidon2Inner {
    #[must_use]
    pub fn new() -> Self {
        let perm = inner_perm();
        let hash = InnerHash::new(perm.clone());
        let compress = InnerCompress::new(perm.clone());
        let val_mmcs = InnerValMmcs::new(hash, compress);
        let dft = InnerDft::default();
        let fri_config = inner_fri_config();
        let pcs = InnerPcs::new(dft, val_mmcs, fri_config);
        let challenger = InnerChallenger::new(perm.clone());
        Self {perm, pcs,  challenger}
    }
}

impl Default for BabyBearPoseidon2Inner {
    fn default() -> Self {
        Self::new()
    }
}

impl StarkGenericConfig for BabyBearPoseidon2Inner {
    type Val = InnerVal;
    type Domain = <InnerPcs as p3_commit::Pcs<InnerChallenge, InnerChallenger>>::Domain;
    type Pcs = InnerPcs;
    type Challenge = InnerChallenge;
    type Challenger = InnerChallenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Self::Challenger {
        self.challenger.clone()
    }
}

impl ZeroCommitment<BabyBearPoseidon2Inner> for InnerPcs {
    fn zero_commitment(&self) -> Com<BabyBearPoseidon2Inner> {
        InnerDigestHash::from([InnerVal::ZERO; DIGEST_SIZE])
    }
}

pub mod baby_bear_poseidon2 {

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing, Field};
    use p3_fri::{FriConfig, TwoAdicFriPcs};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};
    use serde::{Deserialize, Serialize};
    use sp1_primitives::RC_16_30;

    use crate::{Com, StarkGenericConfig, ZeroCommitment, DIGEST_SIZE};

    pub type Val = BabyBear;
    pub type Challenge = BinomialExtensionField<Val, 4>;

    pub type Perm = Poseidon2BabyBear<16>;
    pub type MyHash = PaddingFreeSponge<Perm, 16, 8, DIGEST_SIZE>;
    pub type DigestHash = Hash<Val, Val, DIGEST_SIZE>;
    pub type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    pub type ValMmcs = MerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    pub type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    pub type Dft = Radix2DitParallel<Val>;
    pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    #[must_use]
    pub fn my_perm() -> Perm {
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;
        let mut round_constants = RC_16_30.to_vec();
        let internal_start = ROUNDS_F / 2;
        let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
        let internal_round_constants = round_constants
            .drain(internal_start..internal_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();
        let external_round_constants = round_constants;
        Perm::new(
                ExternalLayerConstants::new(
                    external_round_constants[..4].to_vec(),
                    external_round_constants[4..].to_vec(),
                ),
                internal_round_constants.to_vec(),
            )
    }

    #[must_use]
    pub fn default_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 100,
        };
        FriConfig { log_blowup: 1, log_final_poly_len:0, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
    }

    #[must_use]
    pub fn compressed_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 50,
        };
        FriConfig { log_blowup: 2, log_final_poly_len:0, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
    }

    #[must_use]
    pub fn ultra_compressed_fri_config() -> FriConfig<ChallengeMmcs> {
        let perm = my_perm();
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
        let num_queries = match std::env::var("FRI_QUERIES") {
            Ok(value) => value.parse().unwrap(),
            Err(_) => 33,
        };
        FriConfig { log_blowup: 3, log_final_poly_len:0, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
    }

    enum BabyBearPoseidon2Type {
        Default,
        Compressed,
    }

    #[derive(Deserialize)]
    #[serde(from = "std::marker::PhantomData<BabyBearPoseidon2>")]
    pub struct BabyBearPoseidon2 {
        pub perm: Perm,
        pcs: Pcs,
        challenger: Challenger,
        config_type: BabyBearPoseidon2Type,
    }

    impl BabyBearPoseidon2 {
        #[must_use]
        pub fn new() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft::default();
            let fri_config = default_fri_config();
            let pcs = Pcs::new(dft, val_mmcs, fri_config);
            let challenger = Challenger::new(perm.clone());
            Self {perm, pcs, challenger, config_type: BabyBearPoseidon2Type::Default }
        }

        #[must_use]
        pub fn compressed() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft::default();
            let fri_config = compressed_fri_config();
            let pcs = Pcs::new(dft, val_mmcs, fri_config);
            let challenger = Challenger::new(perm.clone());
            Self {perm, pcs, challenger, config_type: BabyBearPoseidon2Type::Compressed }
        }

        #[must_use]
        pub fn ultra_compressed() -> Self {
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = ValMmcs::new(hash, compress);
            let dft = Dft::default();
            let fri_config = ultra_compressed_fri_config();
            let pcs = Pcs::new(dft, val_mmcs, fri_config);
            let challenger = Challenger::new(perm.clone());
            Self {perm, pcs, challenger, config_type: BabyBearPoseidon2Type::Compressed }
        }
    }

    impl Clone for BabyBearPoseidon2 {
        fn clone(&self) -> Self {
            match self.config_type {
                BabyBearPoseidon2Type::Default => Self::new(),
                BabyBearPoseidon2Type::Compressed => Self::compressed(),
            }
        }
    }

    impl Default for BabyBearPoseidon2 {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Implement serialization manually instead of using serde to avoid cloing the config.
    impl Serialize for BabyBearPoseidon2 {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            std::marker::PhantomData::<BabyBearPoseidon2>.serialize(serializer)
        }
    }

    impl From<std::marker::PhantomData<BabyBearPoseidon2>> for BabyBearPoseidon2 {
        fn from(_: std::marker::PhantomData<BabyBearPoseidon2>) -> Self {
            Self::new()
        }
    }

    impl StarkGenericConfig for BabyBearPoseidon2 {
        type Val = BabyBear;
        type Domain = <Pcs as p3_commit::Pcs<Challenge, Challenger>>::Domain;
        type Pcs = Pcs;
        type Challenge = Challenge;
        type Challenger = Challenger;

        fn pcs(&self) -> &Self::Pcs {
            &self.pcs
        }

        fn initialise_challenger(&self) -> Self::Challenger {
            self.challenger.clone()
        }
    }

    impl ZeroCommitment<BabyBearPoseidon2> for Pcs {
        fn zero_commitment(&self) -> Com<BabyBearPoseidon2> {
            DigestHash::from([Val::ZERO; DIGEST_SIZE])
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::BasedVectorSpace;
    use p3_challenger::{CanSample, CanObserve, FieldChallenger};
    use p3_symmetric::CryptographicHasher;
    use p3_symmetric::PseudoCompressionFunction;

    use p3_commit::Mmcs;

    //use p3_baby_bear::PackedBabyBearAVX2;
    //use crate::baby_bear_poseidon2::{Val, MyHash, MyCompress, ChallengeMmcs, ValMmcs, Dft, my_perm,default_fri_config};
    type SC = baby_bear_poseidon2::BabyBearPoseidon2;
    type F = <SC as StarkGenericConfig>::Val;
    type EF = <SC as StarkGenericConfig>::Challenge;
   
    #[test]
    fn test_challenger() {
        let config = SC::default();
        //let mut challenger = config.challenger();
        let mut challenger = config.initialise_challenger();
        challenger.observe(F::ONE);
        challenger.observe(F::TWO);
        challenger.observe(F::TWO);
        challenger.observe(F::TWO);
        let expect_F: F = BabyBear::from_u32(1573547511);

        let coefficients = vec![BabyBear::from_u32(974685597), BabyBear::from_u32(1115323271),
                               BabyBear::from_u32(1101470026), BabyBear::from_u32(1992477023)];

        let expect_EF: EF = EF::from_basis_coefficients_slice(&coefficients).unwrap();

        let result: F = challenger.sample();
        assert_eq!(expect_F, result); 

        let result_ef: EF = challenger.sample_algebra_element();
    
        assert_eq!(expect_EF, result_ef);
    }

    #[test]
    fn commit_single_1x8() {
        let perm = inner_perm();
        let hash = InnerHash::new(perm.clone());
        let compress = InnerCompress::new(perm.clone());
        
        let mmcs = InnerValMmcs::new(hash.clone(), compress.clone());

        // v = [2, 1, 2, 2, 0, 0, 1, 0]
        let v = vec![
            F::TWO,
            F::ONE,
            F::TWO,
            F::TWO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        let (commit, _) = mmcs.commit_vec(v.clone());

        let expected_result = compress.compress([
            compress.compress([
                compress.compress([hash.hash_item(v[0]), hash.hash_item(v[1])]),
                compress.compress([hash.hash_item(v[2]), hash.hash_item(v[3])]),
            ]),
            compress.compress([
                compress.compress([hash.hash_item(v[4]), hash.hash_item(v[5])]),
                compress.compress([hash.hash_item(v[6]), hash.hash_item(v[7])]),
            ]),
        ]);

        let expected : [BabyBear; 8] = BabyBear::new_array([844319689, 1405767669, 1040991291, 853408487, 340576855, 395735542, 1410456727, 100300798]);
        assert_eq!(expected, expected_result);
        assert_eq!(commit, expected);
    }

    #[test]
    fn test_truncated_permutation_compress() {
        let perm = inner_perm();
        //let hash = InnerHash::new(perm.clone());
        //InnerCompress = TruncatedPermutation<InnerPerm, 2, 8, 16>;
        let compress = InnerCompress::new(perm.clone());
        const N: usize = 2;
        const CHUNK: usize = 8;
        const WIDTH: usize = 8;

        let input: [[F; CHUNK]; N] = [F::new_array([1, 2, 3, 4,5,6,7,8]), 
                                    F::new_array([545148404, 1306564358, 637903742, 106526026, 997551990, 1145772050, 25820965, 1036805604
                                    ])];
        let output = compress.compress(input);
        let expected : [BabyBear; 8] = BabyBear::new_array([1000522935, 1359720241, 1095213472, 1777007570, 585380650, 1943112261, 1058326616, 205684059]);

        assert_eq!(output, expected);
    }

    
   /*
    //test avx2
    #[test]
    #[ignore]  
    fn test_avx2_poseidon2_width_16() {
        // Our Poseidon2 implementation.
        let poseidon2 = inner_perm();

        //let input: [F; 16] = rng.r#gen();
        let input: [F; 16] = BabyBear::new_array([2, 1, 2, 2, 0, 0, 1, 0,2, 1, 2, 2, 0, 0, 1, 0]);

        let mut out_no_avx = input;
        poseidon2.permute_mut(&mut out_no_avx);

        let expected : [BabyBear; 16] = BabyBear::new_array([1622298962, 1536623934, 247570392, 689896596, 217611166, 1742770036, 794609491, 430147729, 
        1723236467, 1367877742, 267378299, 1403615570, 208553167, 427630706, 1702616973, 1631677790]);

        let mut avx2_input = input.map(Into::<PackedBabyBearAVX2>::into);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(out_no_avx, expected);
        assert_eq!(avx2_output, expected);
    }
*/

}
