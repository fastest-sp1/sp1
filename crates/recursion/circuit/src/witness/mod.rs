mod outer;
mod stark;

use p3_matrix::dense::RowMajorMatrix;
use p3_baby_bear::BabyBear;
use p3_commit::{Mmcs, Pcs};
use p3_challenger::{CanObserve, GrindingChallenger, FieldChallenger, CanSample};

use serde::{Deserialize, Serialize};
use sp1_recursion_compiler::ir::{Builder, Ext, Felt};

pub use outer::*;
use sp1_stark::{
    septic_curve::SepticCurve, septic_digest::SepticDigest, septic_extension::SepticExtension,
    ChipOpenedValues, Com, InnerChallenge, InnerVal, OpeningProof, ShardCommitment,
    ShardOpenedValues, ShardProof, Challenge, Val,
};
pub use stark::*;

use crate::{
    hash::{FieldHasherVariable, FieldHasher}, stark::ShardProofVariable, BabyBearFriConfigVariable, CircuitConfig,
    TwoAdicPcsProofVariable,  EF, FriMmcs, 
};

pub trait WitnessWriter<C: CircuitConfig>: Sized {
    fn write_bit(&mut self, value: bool);

    fn write_var(&mut self, value: C::N);

    fn write_felt(&mut self, value: C::F);

    fn write_ext(&mut self, value: C::EF);
}

/// TODO change the name. For now, the name is unique to prevent confusion.
pub trait Witnessable<C: CircuitConfig> {
    type WitnessVariable;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable;

    fn write(&self, witness: &mut impl WitnessWriter<C>);
}

impl<C: CircuitConfig> Witnessable<C> for bool {
    type WitnessVariable = C::Bit;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_bit(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_bit(*self);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for &T {
    type WitnessVariable = T::WitnessVariable;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        (*self).read(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        (*self).write(witness)
    }
}

impl<C: CircuitConfig, T: Witnessable<C>, U: Witnessable<C>> Witnessable<C> for (T, U) {
    type WitnessVariable = (T::WitnessVariable, U::WitnessVariable);

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        (self.0.read(builder), self.1.read(builder))
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.0.write(witness);
        self.1.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal>> Witnessable<C> for InnerVal {
    type WitnessVariable = Felt<InnerVal>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_felt(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_felt(*self);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C> for InnerChallenge {
    type WitnessVariable = Ext<InnerVal, InnerChallenge>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_ext(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        // vec![Block::from(self.as_basis_coefficients_slice())]
        witness.write_ext(*self);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>, const N: usize> Witnessable<C> for [T; N] {
    type WitnessVariable = [T::WitnessVariable; N];

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|x| x.read(builder)).collect::<Vec<_>>().try_into().unwrap_or_else(
            |x: Vec<_>| {
                // Cannot just `.unwrap()` without requiring Debug bounds.
                panic!("could not coerce vec of len {} into array of len {N}", x.len())
            },
        )
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Vec<T> {
    type WitnessVariable = Vec<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|x| x.read(builder)).collect()
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>, SC: BabyBearFriConfigVariable<C>  + FieldHasher<BabyBear>>
    Witnessable<C> for ShardProof<SC>
where
    SC: BabyBearFriConfigVariable<C>,
    Com<SC>: Witnessable<C, WitnessVariable = <SC as FieldHasherVariable<C>>::DigestVariable>,
    OpeningProof<SC>: Witnessable<C, WitnessVariable = TwoAdicPcsProofVariable<C, SC>>,
    SC::Challenger: CanObserve<<SC::ValMmcs as Mmcs<BabyBear>>::Commitment>
        + CanObserve<<FriMmcs<SC> as Mmcs<EF>>::Commitment>
        + CanSample<EF>
        + GrindingChallenger<Witness = BabyBear>
        + FieldChallenger<BabyBear>,
    SC::Pcs: Pcs<
        EF,
        SC::Challenger,
        ProverData: Clone + Send + Sync,
        Proof: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    >,
    SC::ValMmcs: Mmcs<BabyBear, ProverData<RowMajorMatrix<BabyBear>> = SC::RowMajorProverData>,
    //ShardCommitment<Com<SC>>: Witnessable<C>,
    ShardCommitment<Com<SC>>: Witnessable<C, 
        WitnessVariable = ShardCommitment<<SC as FieldHasherVariable<C>>::DigestVariable>>,
    //ShardOpenedValues<Val<SC>, Challenge<SC>>: Witnessable<C>,
    ShardOpenedValues<Val<SC>, Challenge<SC>>: Witnessable<C, 
        WitnessVariable = ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>>,
    Vec<Val<SC>>: Witnessable<C, WitnessVariable = Vec<Felt<C::F>>>,
{
    type WitnessVariable = ShardProofVariable<C, SC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        //let commitment: <ShardCommitment<Com<SC>> as Witnessable<C>>::WitnessVariable = self.commitment.read(builder);
        //let opened_values: <ShardOpenedValues<Val<SC>, Challenge<SC>> as Witnessable<C>>::WitnessVariable = self.opened_values.read(builder);
        let commitment = self.commitment.read(builder);
        let opened_values = self.opened_values.read(builder);
        let opening_proof = self.opening_proof.read(builder);
        let public_values = self.public_values.read(builder);
        let chip_ordering = self.chip_ordering.clone();

        ShardProofVariable {
            commitment,
            opened_values,
            opening_proof,
            public_values,
            chip_ordering,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.commitment.write(witness);
        self.opened_values.write(witness);
        self.opening_proof.write(witness);
        self.public_values.write(witness);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for ShardCommitment<T> {
    type WitnessVariable = ShardCommitment<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let main_commit = self.main_commit.read(builder);
        let permutation_commit = self.permutation_commit.read(builder);
        let quotient_commit = self.quotient_commit.read(builder);
        Self::WitnessVariable { main_commit, permutation_commit, quotient_commit }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.main_commit.write(witness);
        self.permutation_commit.write(witness);
        self.quotient_commit.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for ShardOpenedValues<InnerVal, InnerChallenge>
{
    type WitnessVariable = ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let chips = self.chips.read(builder);
        Self::WitnessVariable { chips }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.chips.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for SepticDigest<InnerVal>
{
    type WitnessVariable = SepticDigest<Felt<C::F>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let x = self.0.x.0.read(builder);
        let y = self.0.y.0.read(builder);
        SepticDigest(SepticCurve { x: SepticExtension(x), y: SepticExtension(y) })
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.0.x.0.write(witness);
        self.0.y.0.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for ChipOpenedValues<InnerVal, InnerChallenge>
{
    type WitnessVariable = ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let preprocessed = self.preprocessed.read(builder);
        let main = self.main.read(builder);
        let permutation = self.permutation.read(builder);
        let quotient = self.quotient.read(builder);
        let global_cumulative_sum = self.global_cumulative_sum.read(builder);
        let local_cumulative_sum = self.local_cumulative_sum.read(builder);
        let log_degree = self.log_degree;
        Self::WitnessVariable {
            preprocessed,
            main,
            permutation,
            quotient,
            global_cumulative_sum,
            local_cumulative_sum,
            log_degree,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.preprocessed.write(witness);
        self.main.write(witness);
        self.permutation.write(witness);
        self.quotient.write(witness);
        self.global_cumulative_sum.write(witness);
        self.local_cumulative_sum.write(witness);
    }
}

/*
impl<C, F, P> Witnessable<C> for Hash<F, P, DIGEST_SIZE>
where
    C: CircuitConfig<F = F>, // Ensure circuit field matches the hash field
    F: PrimeCharacteristicRing,
    P: Witnessable<C, WitnessVariable = C::Felt>, // The elements of the hash must be witnessable
    [P; DIGEST_SIZE]: Witnessable<C, WitnessVariable = [C::Felt; DIGEST_SIZE]>, // The inner array must be witnessable
{
    // The variable for a hash is just an array of field elements in the circuit.
    type WitnessVariable = [C::Felt; DIGEST_SIZE];

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        // We read the inner array `self.0`.
        self.value.read(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        // We write the inner array `self.0`.
        self.value.write(witness);
    }
}*/
