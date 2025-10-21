//! GPU-accelerated implementation for generating the permutation trace.
//!
//! This module contains the logic to flatten the complex Rust `Interaction` structs
//! into a C-compatible format, call a CUDA kernel to compute the trace on the device,
//! and manage the associated GPU resources.


use crate::air::MultiTableAirBuilder;
use crate::chip::Chip;
use crate::gpu::ffi::*;
use crate::{CudaResultCheck, GpuMatrix, GpuMatrixC,  GpuMemBlk};
use crate::lookup::{Interaction, InteractionBuilder};
use crate::{local_permutation_trace_width, };
use p3_air::{PairCol, VirtualPairCol};
use p3_field::{ExtensionField, PrimeField, PrimeCharacteristicRing};
use std::ffi::c_void;
use std::marker::PhantomData;
use crate::baby_bear_poseidon2::{Val, Challenge};
use crate::air::MachineAir;

// --- FFI Structs ---
// These structs must exactly match the layout of the `struct`s defined in the C++/CUDA header.
// They are "Plain Old Data" (POD) and safe to send across the FFI boundary.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiVpcTerm {
    col_type: i32, // 0 for Preprocessed, 1 for Main
    col_index: i32,
    weight: Val,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiVirtualPairCol {
    terms_ptr: *const FfiVpcTerm,
    num_terms: i32,
    constant: Val,
}

// Define a default for initialization in arrays
impl Default for FfiVirtualPairCol {
    fn default() -> Self {
        Self {
            terms_ptr: std::ptr::null(),
            num_terms: 0,
            constant: Val::ZERO,
        }
    }
}

const MAX_INTERACTION_VALUES: usize = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiInteraction {
    pub values: [FfiVirtualPairCol; MAX_INTERACTION_VALUES],
    pub num_values: i32,
    pub multiplicity: FfiVirtualPairCol,
    //argument_index: i32,
    pub kind: i32,
    /// The scope of the interaction.
    pub scope: i32, 
    pub is_send: bool,
}

/// A helper struct to own the flattened data and ensure its lifetime
/// is long enough for the GPU to use the pointers.
struct FfiInteractionData {
    ffi_sends: Vec<FfiInteraction>,
    ffi_receives: Vec<FfiInteraction>,
    // These vectors own all the nested data that the pointers inside FfiInteraction point to.
    all_vpc_terms: Vec<FfiVpcTerm>,
    // We don't need a separate `all_vpcs` because `FfiVirtualPairCol` can be copied.
}

/// Flattens a slice of Rust `Interaction` objects into a C-compatible format.
///
/// This is the core data marshalling step. It iterates through the nested Rust structs
/// and packs all their data into contiguous vectors, which can then be safely copied to the GPU.
fn convert_interactions_to_ffi(
    sends: &[Interaction<Val>],
    receives: &[Interaction<Val>],
) -> FfiInteractionData {
    let mut all_vpc_terms = Vec::new();

    // Helper closure to convert a single VirtualPairCol into its FFI representation.
    // It appends the term data to the `all_vpc_terms` vector and returns a struct
    // containing a pointer into that vector.
    let mut convert_vpc = |vpc: &VirtualPairCol<Val>| -> FfiVirtualPairCol {
        let terms_start_idx = all_vpc_terms.len();
        for (col, weight) in &vpc.column_weights {
            all_vpc_terms.push(FfiVpcTerm {
                col_type: match col {
                    PairCol::Preprocessed(_) => 0,
                    PairCol::Main(_) => 1,
                },
                col_index: match col {
                    PairCol::Preprocessed(i) | PairCol::Main(i) => *i as i32,
                },
                weight: *weight,
            });
        }
        FfiVirtualPairCol {
            // This pointer will be invalid once `all_vpc_terms` is moved or reallocated,
            // so we must be careful to get the final pointer only after all appends are done.
            terms_ptr: std::ptr::null(), // Placeholder
            num_terms: vpc.column_weights.len() as i32,
            constant: vpc.constant,
        }
    };

    let mut ffi_sends_intermediate = Vec::new();
    for interaction in sends {
        let mut ffi_values = [FfiVirtualPairCol::default(); MAX_INTERACTION_VALUES];
        for (i, v) in interaction.values.iter().enumerate() {
            ffi_values[i] = convert_vpc(v);
        }
        ffi_sends_intermediate.push(FfiInteraction {
            values: ffi_values,
            num_values: interaction.values.len() as i32,
            multiplicity: convert_vpc(&interaction.multiplicity),
            kind: interaction.kind as i32,
            scope: interaction.scope as i32,
            is_send: true,
        });
    }

    let mut ffi_receives_intermediate = Vec::new();
    for interaction in receives {
        let mut ffi_values = [FfiVirtualPairCol::default(); MAX_INTERACTION_VALUES];
        for (i, v) in interaction.values.iter().enumerate() {
            ffi_values[i] = convert_vpc(v);
        }
        ffi_receives_intermediate.push(FfiInteraction {
            values: ffi_values,
            num_values: interaction.values.len() as i32,
            multiplicity: convert_vpc(&interaction.multiplicity),
            kind: interaction.kind as i32,
            scope: interaction.scope as i32,
            is_send: false,
        });
    }

    // Now that `all_vpc_terms` has its final size, we can fix the pointers.
    let base_terms_ptr = all_vpc_terms.as_ptr();
    let mut term_cursor = 0;

    let ffi_sends = ffi_sends_intermediate.into_iter().map(|mut ffi_int| {
        for i in 0..ffi_int.num_values as usize {
            let num_terms = ffi_int.values[i].num_terms as usize;
            ffi_int.values[i].terms_ptr = unsafe { base_terms_ptr.add(term_cursor) };
            term_cursor += num_terms;
        }
        let num_terms = ffi_int.multiplicity.num_terms as usize;
        ffi_int.multiplicity.terms_ptr = unsafe { base_terms_ptr.add(term_cursor) };
        term_cursor += num_terms;
        ffi_int
    }).collect();
    
    let ffi_receives = ffi_receives_intermediate.into_iter().map(|mut ffi_int| {
        for i in 0..ffi_int.num_values as usize {
            let num_terms = ffi_int.values[i].num_terms as usize;
            ffi_int.values[i].terms_ptr = unsafe { base_terms_ptr.add(term_cursor) };
            term_cursor += num_terms;
        }
        let num_terms = ffi_int.multiplicity.num_terms as usize;
        ffi_int.multiplicity.terms_ptr = unsafe { base_terms_ptr.add(term_cursor) };
        term_cursor += num_terms;
        ffi_int
    }).collect();

    FfiInteractionData {
        ffi_sends,
        ffi_receives,
        all_vpc_terms,
    }
}


/// Generates the permutation trace for a chip entirely on the GPU.
pub fn generate_perm_trace(
    chip_id: i32,
    perm_width: i32,
    batch_size: i32,
    preprocessed_lde: &GpuMatrix<Val>,
    main_lde: &GpuMatrix<Val>,
    random_elements: &[Challenge],
    gpu_mem_blk: &GpuMemBlk,
) -> (GpuMatrix<Challenge>, Challenge) {

    let height = main_lde.height;

    if perm_width == 0 {
        //return (GpuMatrix::new(height, 0), Challenge::ZERO);
        panic!("perm_width is 0!" );
    }
    
    // 1. Allocate the output permutation trace on the GPU.
    let mut perm_trace_gpu = GpuMatrix::<Challenge>::new(height, perm_width as usize, gpu_mem_blk);
     
    // 4. Call the main FFI orchestration function.
    let mut local_cumulative_sum = Challenge::ZERO;
    unsafe {
        let main_lde_c: GpuMatrixC = (main_lde).into();
        let preprocessed_lde_c: GpuMatrixC = (preprocessed_lde).into();
        let mut perm_trace_gpu_c: GpuMatrixC = (&perm_trace_gpu).into();
        generate_permutation_trace_gpu(
            chip_id,
            &main_lde_c,// as *const GpuMatrix<Val>,
            &preprocessed_lde_c,// as *const GpuMatrix<Val>,
            &mut perm_trace_gpu_c,// as *mut GpuMatrix<Challenge>,
            random_elements.as_ptr(),
            batch_size,
            &mut local_cumulative_sum,
        )
        .check("generate_permutation_trace_gpu failed");
    }

    (perm_trace_gpu, local_cumulative_sum)
}

