//! A packed septic extension field based on p3_field PackedBinomialExtensionField.
use p3_field::{
    Algebra, BasedVectorSpace, Field, PackedField, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing, PrimeField, field_to_array,
};
use p3_util::{flatten_to_base, reconstitute_from_base};
use serde::{Deserialize, Serialize};
use std::array;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::septic_extension::{SepticExtension, septic_mul, vector_add, vector_sub};
use itertools::Itertools;

const D: usize = 7;
/// A septic extension with an irreducible polynomial `z^7 - 2z - 5`.
///
/// The field can be constructed as `F_{p^7} = F_p[z]/(z^7 - 2z - 5)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct PackedSepticExtension<F: Field, PF: PackedField<Scalar = F>>(pub [PF; D]);

impl<F: Field, PF: PackedField<Scalar = F>> PackedSepticExtension<F, PF> {
    const fn new(value: [PF; D]) -> Self {
        Self(value) //?
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> Default for PackedSepticExtension<F, PF> {
    #[inline]
    fn default() -> Self {
        Self(array::from_fn(|_| PF::ZERO))
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> From<SepticExtension<F>>
    for PackedSepticExtension<F, PF>
{
    #[inline]
    fn from(x: SepticExtension<F>) -> Self {
        Self(x.0.map(Into::<PF>::into))
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> From<PF> for PackedSepticExtension<F, PF> {
    #[inline]
    fn from(x: PF) -> Self {
        Self(field_to_array::<PF, D>(x))
    }
}

//The following two are ok? G
impl<F: Field, PF: PackedField<Scalar = F>> Algebra<SepticExtension<F>>
    for PackedSepticExtension<F, PF>
{
}

impl<F: Field, PF: PackedField<Scalar = F>> Algebra<PF> for PackedSepticExtension<F, PF> {}

impl<F, PF> PrimeCharacteristicRing for PackedSepticExtension<F, PF>
where
    //F: Field<D>,
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type PrimeSubfield = PF::PrimeSubfield;

    const ZERO: Self = Self([PF::ZERO; D]);

    const ONE: Self = Self(field_to_array::<PF, D>(PF::ONE));

    const TWO: Self = Self(field_to_array::<PF, D>(PF::TWO));

    const NEG_ONE: Self = Self(field_to_array::<PF, D>(PF::NEG_ONE));

    #[inline]
    fn from_prime_subfield(val: Self::PrimeSubfield) -> Self {
        PF::from_prime_subfield(val).into()
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        PF::from_bool(b).into()
    }
    /*
        #[inline(always)]
        fn square(&self) -> Self {
            match D {
                2 => {
                    let a = self.value;
                    let mut res = Self::default();
                    res.value[0] = a[0].square() + a[1].square() * F::W;
                    res.value[1] = a[0] * a[1].double();
                    res
                }
                3 => {
                    let mut res = Self::default();
                    cubic_square(&self.value, &mut res.value);
                    res
                }
                _ => <Self as Mul<Self>>::mul(*self, *self),
            }
        }
    */
    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(PF::zero_vec(len * D)) }
    }
}

impl<F, PF> BasedVectorSpace<PF> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    const DIMENSION: usize = D;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[PF] {
        &self.0
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> PF>(f: Fn) -> Self {
        Self(array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = PF>>(mut iter: I) -> Option<Self> {
        (iter.len() == D).then(|| Self::new(array::from_fn(|_| iter.next().unwrap()))) // The unwrap is safe as we just checked the length of iter.
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<PF> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[PF; D]`
            flatten_to_base::<PF, Self>(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<PF>) -> Vec<Self> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[PF; D]`
            reconstitute_from_base::<PF, Self>(vec)
        }
    }
}

impl<F: PrimeField> PackedFieldExtension<F, SepticExtension<F>>
    for PackedSepticExtension<F, F::Packing>
//where
//    F: Field<D>,
{
    #[inline]
    fn from_ext_slice(ext_slice: &[SepticExtension<F>]) -> Self {
        let width = F::Packing::WIDTH;
        assert_eq!(ext_slice.len(), width);

        let res = array::from_fn(|i| F::Packing::from_fn(|j| ext_slice[j].0[i]));
        Self::new(res)
    }

    #[inline]
    fn to_ext_iter(
        iter: impl IntoIterator<Item = Self>,
    ) -> impl Iterator<Item = SepticExtension<F>> {
        let width = F::Packing::WIDTH;
        iter.into_iter().flat_map(move |x| {
            (0..width).map(move |i| {
                let values = array::from_fn(|j| x.0[j].as_slice()[i]);
                SepticExtension::new(values)
            })
        })
    }

    #[inline]
    fn packed_ext_powers(base: SepticExtension<F>) -> Powers<Self> {
        let width = F::Packing::WIDTH;
        let powers = base.powers().take(width + 1).collect_vec();
        // Transpose first WIDTH powers
        let current = Self::from_ext_slice(&powers[..width]);

        // Broadcast self^WIDTH
        let multiplier = powers[width].into();

        Powers { base: multiplier, current }
    }
}

impl<F, PF> Neg for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(self.0.map(PF::neg))
    }
}

impl<F, PF> Add for PackedSepticExtension<F, PF>
where
    //F: Field<D>,
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = vector_add(&self.0, &rhs.0);
        Self(value)
    }
}

impl<F, PF> Add<SepticExtension<F>> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: SepticExtension<F>) -> Self {
        let value = vector_add(&self.0, &rhs.0);
        //Self { value }
        Self(value)
    }
}

impl<F, PF> Add<PF> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: PF) -> Self {
        self.0[0] += rhs;
        self
    }
}

impl<F, PF> AddAssign for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<F, PF> AddAssign<SepticExtension<F>> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: SepticExtension<F>) {
        for i in 0..D {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<F, PF> AddAssign<PF> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: PF) {
        self.0[0] += rhs;
    }
}

impl<F, PF> Sum for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, PF> Sub for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = vector_sub(&self.0, &rhs.0);
        Self(value)
    }
}

impl<F, PF> Sub<SepticExtension<F>> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SepticExtension<F>) -> Self {
        let value = vector_sub(&self.0, &rhs.0);
        Self(value)
    }
}

impl<F, PF> Sub<PF> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: PF) -> Self {
        let mut res = self.0;
        res[0] -= rhs;
        Self(res)
    }
}

impl<F, PF> SubAssign for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, PF> SubAssign<SepticExtension<F>> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: SepticExtension<F>) {
        *self = *self - rhs;
    }
}

impl<F, PF> SubAssign<PF> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: PF) {
        *self = *self - rhs;
    }
}

impl<F, PF> Mul for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        let mut res = Self::default();

        septic_mul(&a, &b, &mut res.0); //?
        res
    }
}

impl<F, PF> Mul<SepticExtension<F>> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: SepticExtension<F>) -> Self {
        let a = self.0;
        let b = rhs.0;
        let mut res = Self::default();

        septic_mul(&a, &b, &mut res.0); //?
        res
    }
}

impl<F, PF> Mul<PF> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: PF) -> Self {
        Self(self.0.map(|x| x * rhs))
    }
}

impl<F, PF> Product for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ZERO)
    }
}

impl<F, PF> MulAssign for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, PF> MulAssign<SepticExtension<F>> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: SepticExtension<F>) {
        *self = *self * rhs;
    }
}

impl<F, PF> MulAssign<PF> for PackedSepticExtension<F, PF>
where
    F: Field,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: PF) {
        *self = *self * rhs;
    }
}
