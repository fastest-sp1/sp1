use p3_air::Air;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Proof;
use sp1_stark::{StarkGenericConfig, UniConfig, PcsError};

#[cfg(debug_assertions)]
#[cfg(not(doctest))]
pub fn uni_stark_prove<SC, A>(
    config: &SC,
    air: &A,
    trace: RowMajorMatrix<SC::Val>,
) -> Proof<UniConfig<SC>>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<SC::Val>>
        + for<'a> Air<p3_uni_stark::ProverConstraintFolder<'a, UniConfig<SC>>>
        + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, SC::Val>>,
{
    p3_uni_stark::prove(&UniConfig(config.clone()), air, trace, &vec![])
}

#[cfg(not(debug_assertions))]
pub fn uni_stark_prove<SC, A>(
    config: &SC,
    air: &A,
    trace: RowMajorMatrix<SC::Val>,
) -> Proof<UniConfig<SC>>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<SC::Val>>
        + for<'a> Air<p3_uni_stark::ProverConstraintFolder<'a, UniConfig<SC>>>,
{
    p3_uni_stark::prove(&UniConfig(config.clone()), air, trace, &vec![])
}

#[cfg(debug_assertions)]
#[cfg(not(doctest))]
pub fn uni_stark_verify<SC, A>(
    config: &SC,
    air: &A,
    proof: &Proof<UniConfig<SC>>,
) -> Result<(), p3_uni_stark::VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<SC::Val>>
        + for<'a> Air<p3_uni_stark::VerifierConstraintFolder<'a, UniConfig<SC>>>
        + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, SC::Val>>,
{
    p3_uni_stark::verify(&UniConfig(config.clone()), air, proof, &vec![])
}

#[cfg(not(debug_assertions))]
pub fn uni_stark_verify<SC, A>(
    config: &SC,
    air: &A,
    proof: &Proof<UniConfig<SC>>,
) -> Result<(), p3_uni_stark::VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<SC::Val>>
        + for<'a> Air<p3_uni_stark::VerifierConstraintFolder<'a, UniConfig<SC>>>,
{
    p3_uni_stark::verify(&UniConfig(config.clone()), air, proof, &vec![])
}
