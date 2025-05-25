use p3_bn254_fr::{Bn254Fr, FFBn254Fr};
use zkhash::{
    ark_ff::{BigInteger, PrimeField},
    fields::bn256::FpBN256 as ark_FpBN256,
    poseidon2::poseidon2_instance_bn256::RC3,
};

fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254Fr {
    let mut full_bytes = [0; 32];
    let bytes = input.into_bigint().to_bytes_le();
    full_bytes[..bytes.len()].copy_from_slice(&bytes);
    let value = FFBn254Fr::from_bytes(&full_bytes);

    if value.is_some().into() {
        Bn254Fr {
            value: value.unwrap(),
        }
    } else {
        panic!("Invalid field element")
    }
}

pub fn bn254_poseidon2_rc3() -> Vec<[Bn254Fr; 3]> {
    RC3.iter()
        .map(|vec| {
            vec.iter().cloned().map(bn254_from_ark_ff).collect::<Vec<_>>().try_into().unwrap()
        })
        .collect()
}

pub fn bn254_poseidon2_rc4() -> Vec<[Bn254Fr; 4]> {
    RC3.iter()
        .map(|vec| {
            let result: [Bn254Fr; 3] =
                vec.iter().cloned().map(bn254_from_ark_ff).collect::<Vec<_>>().try_into().unwrap();
            [result[0], result[1], result[2], result[2]]
        })
        .collect()
}
