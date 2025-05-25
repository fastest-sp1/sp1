use p3_field::{
    extension::{BinomialExtensionField, BinomiallyExtendable},
    BasedVectorSpace, Field,
};
use sp1_stark::air::BinomialExtension;

use super::Block;

use crate::runtime::D;

pub trait BinomialExtensionUtils<T> {
    fn from_block(block: Block<T>) -> Self;

    fn as_block(&self) -> Block<T>;
}

impl<T: Clone> BinomialExtensionUtils<T> for BinomialExtension<T> {
    fn from_block(block: Block<T>) -> Self {
        Self(block.0)
    }

    fn as_block(&self) -> Block<T> {
        Block(self.0.clone())
    }
}

impl<F> BinomialExtensionUtils<F> for BinomialExtensionField<F, D>
where
    F: Field + BinomiallyExtendable<D>,
{
    fn from_block(block: Block<F>) -> Self {
        Self::from_basis_coefficients_slice(&block.0).unwrap()
    }

    fn as_block(&self) -> Block<F> {
        Block(self.as_basis_coefficients_slice().try_into().unwrap())
    }
}
