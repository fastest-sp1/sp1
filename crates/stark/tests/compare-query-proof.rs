// In your test file, e.g., tests/fri_query_consistency.rs
use std::path::Path;
use std::path::PathBuf;
use serde::Deserialize;
use p3_field::extension::BinomialExtensionField;
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
//use crate::baby_bear_poseidon2::Challenge; // Your concrete Challenge type

const DIGEST_SIZE: usize = 8;
type Challenge = BinomialExtensionField<BabyBear, 4>;

mod challenge_serde {
    use super::{BabyBear, Challenge}; 
    use p3_field::PrimeCharacteristicRing;
    use serde::{self, Deserialize, Deserializer, Serializer};
    use p3_field::PackedValue;
    use p3_field::PackedFieldExtension;
    use p3_field::BasedVectorSpace;
    use p3_field::PrimeField32;
    use serde::{Serialize};

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
/*
#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct DeserializableCommitPhaseProofStep {
    #[serde(with = "challenge_serde")]
    pub sibling_value: Challenge,
    pub opening_proof: Vec<[BabyBear; DIGEST_SIZE]>,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct DeserializableQueryProof {
    pub commit_phase_openings: Vec<DeserializableCommitPhaseProofStep>,
}

// The top-level structure of the JSON file
pub type QueryProofsFile = Vec<DeserializableQueryProof>;
*/

///////new
#[derive(Deserialize, Debug, PartialEq, Eq)]
struct ChallengeWrapper {
    #[serde(with = "challenge_serde")]
    value: Challenge,
    _phantom: serde_json::Value,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct DeserializableCommitPhaseProofStep {
    // The sibling_value field is now this wrapper struct
    sibling_value: ChallengeWrapper,
    opening_proof: Vec<[BabyBear; DIGEST_SIZE]>,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct DeserializableQueryProof {
    commit_phase_openings: Vec<DeserializableCommitPhaseProofStep>,
}

pub type QueryProofsFile = Vec<DeserializableQueryProof>;

// In tests/fri_query_consistency.rs

use std::fs::File;
use std::io::BufReader;
use rayon::prelude::*; // For parallel comparison

// --- Paste the Deserializable structs and challenge_serde module here ---

#[test]
fn compare_fri_query_proofs() {
    println!("\n--- Comparing FRI Query Proofs from cpu_query_proofs.json and gpu_query_proofs.json ---");

    let test_data_dir = PathBuf::from("test_data");

    // 检查目录是否存在
    if !test_data_dir.is_dir() {
        panic!(
            "Test data directory not found at '{}'. Please create it and place your json files there.",
            test_data_dir.display()
        );
    }

     let cpu_filepath = test_data_dir.join("cpu_query_proofs.json");
    let gpu_filepath = test_data_dir.join("gpu_query_proofs.json");

    // 1. Load the data from files
    let cpu_proofs: QueryProofsFile = {
        let file = File::open(cpu_filepath).expect("Failed to open cpu_query_proofs.json");
        serde_json::from_reader(BufReader::new(file)).expect("Failed to parse cpu_query_proofs.json")
    };
    let gpu_proofs: QueryProofsFile = {
        let file = File::open(gpu_filepath).expect("Failed to open gpu_query_proofs.json");
        serde_json::from_reader(BufReader::new(file)).expect("Failed to parse gpu_query_proofs.json")
    };

    // 2. Perform comparisons
    assert_eq!(
        cpu_proofs.len(),
        gpu_proofs.len(),
        "Top-level QueryProof vectors have different lengths! CPU: {}, GPU: {}",
        cpu_proofs.len(),
        gpu_proofs.len()
    );
    println!("Total number of QueryProofs match: {}", cpu_proofs.len());

    // Use rayon to compare all QueryProof pairs in parallel
    let mismatches: Vec<String> = cpu_proofs
        .par_iter()
        .zip(gpu_proofs.par_iter())
        .enumerate()
        .filter_map(|(i, (cpu_proof, gpu_proof))| {
            // This closure will run in parallel for each query proof
            if cpu_proof != gpu_proof {
                // If they are not equal, find the specific difference
                let mut error_msg = format!("\n[MISMATCH] QueryProof at index {} differs.\n", i);

                if cpu_proof.commit_phase_openings.len() != gpu_proof.commit_phase_openings.len() {
                    error_msg.push_str(&format!(
                        "  - Different number of layers (commit_phase_openings): CPU={}, GPU={}\n",
                        cpu_proof.commit_phase_openings.len(), gpu_proof.commit_phase_openings.len()
                    ));
                  //  return Some(error_msg);
                  println!("NO.：{}, len is defferent:{}", i, error_msg);
                }

                // Find the first differing layer
                for j in 0..cpu_proof.commit_phase_openings.len() {
                    let cpu_step = &cpu_proof.commit_phase_openings[j];
                    let gpu_step = &gpu_proof.commit_phase_openings[j];
                    
                    if cpu_step != gpu_step {
                        error_msg.push_str(&format!("  - Difference found at layer {}:\n", j));
                        if cpu_step.sibling_value != gpu_step.sibling_value {
                            error_msg.push_str(&format!("    - Sibling values mismatch:\n"));
                            error_msg.push_str(&format!("      CPU: {:?}\n", cpu_step.sibling_value));
                            error_msg.push_str(&format!("      GPU: {:?}\n", gpu_step.sibling_value));
                        }
                        if cpu_step.opening_proof != gpu_step.opening_proof {
                             error_msg.push_str(&format!("    - Opening proofs (Merkle paths) mismatch.\n"));
                             // Optionally print the first differing sibling hash
                        }
                        println!("NO.：{}, content is defferent:{}", i, error_msg);
                        //break; // Stop at the first differing layer
                    }
                }
                Some(error_msg)
            } else {
                None // This pair matches
            }
        })
        .collect();

    // 3. Report final result
    if mismatches.is_empty() {
        println!("\n==================== SUCCESS ====================");
        println!("All {} FRI Query Proofs are identical.", cpu_proofs.len());
        println!("==============================================");
    } else {
        println!("\n==================== FAILURE ====================");
        println!("Found mismatches in {} out of {} Query Proofs:", mismatches.len(), cpu_proofs.len());
        for msg in mismatches {
            print!("{}", msg); // `msg` already contains newlines
        }
        println!("==============================================");
        panic!("FRI Query Proofs do not match. See logs for details.");
    }
}