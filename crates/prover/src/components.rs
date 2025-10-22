use sp1_core_machine::riscv::RiscvAir;

use crate::{CompressAir, CoreSC, InnerSC, OuterSC, ShrinkAir, WrapAir};

#[cfg(feature = "recursion_cuda")]
use sp1_stark::{CpuProver, GpuMachineProver, GpuProver, MachineProver, StarkGenericConfig};

#[cfg(not(feature = "recursion_cuda"))]
use sp1_stark::{CpuProver, MachineProver, StarkGenericConfig};

pub trait SP1ProverComponents: Send + Sync {
    /// The prover for making SP1 core proofs.
    type CoreProver: MachineProver<CoreSC, RiscvAir<<CoreSC as StarkGenericConfig>::Val>>
        + Send
        + Sync;

    /// The prover for making SP1 recursive proofs.
    #[cfg(not(feature = "recursion_cuda"))]
    type CompressProver: MachineProver<InnerSC, CompressAir<<InnerSC as StarkGenericConfig>::Val>>
        + Send
        + Sync;
    #[cfg(feature = "recursion_cuda")]
    type CompressProver: GpuMachineProver<InnerSC, CompressAir<<InnerSC as StarkGenericConfig>::Val>>
        + Send
        + Sync;

    /// The prover for shrinking compressed proofs.
    #[cfg(not(feature = "recursion_cuda"))]
    type ShrinkProver: MachineProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>
        + Send
        + Sync;
    #[cfg(feature = "recursion_cuda")]
    type ShrinkProver: GpuMachineProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>
        + Send
        + Sync;

    /// The prover for wrapping compressed proofs into SNARK-friendly field elements.
    type WrapProver: MachineProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>
        + Send
        + Sync;
}

#[cfg(not(feature = "recursion_cuda"))]
pub type CpuProverComponents = ProverComponents;

#[cfg(feature = "recursion_cuda")]
pub type CpuProverComponents = GpuProverComponents;

pub struct ProverComponents;

#[cfg(not(feature = "recursion_cuda"))]
impl SP1ProverComponents for ProverComponents {
    type CoreProver = CpuProver<CoreSC, RiscvAir<<CoreSC as StarkGenericConfig>::Val>>;

    type CompressProver = CpuProver<InnerSC, CompressAir<<InnerSC as StarkGenericConfig>::Val>>;

    type ShrinkProver = CpuProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>;
    type WrapProver = CpuProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>;
}

pub struct GpuProverComponents;

#[cfg(feature = "recursion_cuda")]
impl SP1ProverComponents for GpuProverComponents {
    // For Core, we still use CpuProver .
    type CoreProver = CpuProver<CoreSC, RiscvAir<<CoreSC as StarkGenericConfig>::Val>>;

    // For recursive stages, use the GpuProver.
    type CompressProver = GpuProver<InnerSC, CompressAir<<InnerSC as StarkGenericConfig>::Val>>;

    //type ShrinkProver = CpuProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>;
    type ShrinkProver = GpuProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>;

    type WrapProver = CpuProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>;
}
