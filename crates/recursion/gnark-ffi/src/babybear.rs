use p3_baby_bear::BabyBear;
use p3_field::{
    extension::BinomialExtensionField, BasedVectorSpace, PrimeCharacteristicRing, Field, PrimeField32,
};

#[unsafe(no_mangle)]
pub extern "C" fn babybearextinv(a: u32, b: u32, c: u32, d: u32, i: u32) -> u32 {
    let a = BabyBear::from_u32(a);
    let b = BabyBear::from_u32(b);
    let c = BabyBear::from_u32(c);
    let d = BabyBear::from_u32(d);
    let inv = BinomialExtensionField::<BabyBear, 4>::from_basis_coefficients_slice(&[a, b, c, d]).unwrap().inverse();
    let inv: &[BabyBear] = inv.as_basis_coefficients_slice();
    inv[i as usize].as_canonical_u32()
}

#[unsafe(no_mangle)]
pub extern "C" fn babybearinv(a: u32) -> u32 {
    let a = BabyBear::from_u32(a);
    a.inverse().as_canonical_u32()
}

#[cfg(test)]
pub mod test {
    use super::babybearextinv;

    #[test]
    fn test_babybearextinv() {
        babybearextinv(1, 2, 3, 4, 0);
    }
}
