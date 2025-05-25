use std::array;

use p3_air::PairBuilder;
use p3_baby_bear::INTERNAL_DIAG_MONTY_16;
use p3_field::{PrimeCharacteristicRing, PrimeField32};

use p3_poseidon2::matmul_internal_sp1;
use sp1_primitives::RC_16_30_U32;
use sp1_stark::air::MachineAirBuilder;

use super::{permutation::Poseidon2Cols, NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS, WIDTH};

pub fn apply_m_4_mut<AF>(x: &mut [AF])
where
    AF: PrimeCharacteristicRing,
{
    let t01 = x[0].clone() + x[1].clone();
    let t23 = x[2].clone() + x[3].clone();
    let t0123 = t01.clone() + t23.clone();
    let t01123 = t0123.clone() + x[1].clone();
    let t01233 = t0123.clone() + x[3].clone();
    x[3] = t01233.clone() + x[0].double();
    x[1] = t01123.clone() + x[2].double();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

pub fn external_linear_layer_mut<AF: PrimeCharacteristicRing>(state: &mut [AF; WIDTH]) {
    for j in (0..WIDTH).step_by(4) {
        apply_m_4_mut(&mut state[j..j + 4]);
    }
    let sums: [AF; 4] =
        core::array::from_fn(|k| (0..WIDTH).step_by(4).map(|j| state[j + k].clone()).sum::<AF>());

    for j in 0..WIDTH {
        state[j] = state[j].clone() + sums[j % 4].clone();
    }
}

pub fn external_linear_layer<AF: PrimeCharacteristicRing + Copy>(state: &[AF; WIDTH]) -> [AF; WIDTH] {
    let mut state = *state;
    external_linear_layer_mut(&mut state);
    state
}

//pub fn internal_linear_layer_mut<AF: PrimeCharacteristicRing<PrimeSubfield = F> + Field>(state: &mut [AF; WIDTH]) {
pub fn internal_linear_layer_mut<AF>(state: &mut [AF; WIDTH]) 
where
    AF: PrimeCharacteristicRing,
{
    //let matmul_constants: [<AF as PrimeCharacteristicRing>::PrimeSubfield; WIDTH] =
    let matmul_constants: [AF; WIDTH] =
        INTERNAL_DIAG_MONTY_16
            .iter()
            .map(|x| AF::from_u32(x.as_canonical_u32()))
            //.map(|x| <AF as PrimeCharacteristicRing>::PrimeSubfield::from_u32(x.as_canonical_u32()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(); 
    matmul_internal_sp1(state, matmul_constants);

    //let monty_inverse = AF::from_u32(MONTY_INVERSE.as_canonical_u32());
    //state.iter_mut().for_each(|i| *i = i.clone() * monty_inverse.clone());
}

/// Eval the constraints for the external rounds.
pub fn eval_external_round<AB>(builder: &mut AB, local_row: &dyn Poseidon2Cols<AB::Var>, r: usize)
where
    AB: MachineAirBuilder + PairBuilder,
{
    let mut local_state: [AB::Expr; WIDTH] =
        array::from_fn(|i| local_row.external_rounds_state()[r][i].into());

    // For the first round, apply the linear layer.
    if r == 0 {
        external_linear_layer_mut(&mut local_state);
    }

    // Add the round constants.
    let round = if r < NUM_EXTERNAL_ROUNDS / 2 { r } else { r + NUM_INTERNAL_ROUNDS };
    let add_rc: [AB::Expr; WIDTH] = array::from_fn(|i| {
        local_state[i].clone() + AB::F::from_u32(RC_16_30_U32[round][i])
    });

    // Apply the sboxes.
    // See `populate_external_round` for why we don't have columns for the sbox output here.
    let mut sbox_deg_7: [AB::Expr; WIDTH] = core::array::from_fn(|_| AB::Expr::ZERO);
    let mut sbox_deg_3: [AB::Expr; WIDTH] = core::array::from_fn(|_| AB::Expr::ZERO);
    for i in 0..WIDTH {
        let calculated_sbox_deg_3 = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();

        if let Some(external_sbox) = local_row.external_rounds_sbox() {
            builder.assert_eq(external_sbox[r][i].into(), calculated_sbox_deg_3);
            sbox_deg_3[i] = external_sbox[r][i].into();
        } else {
            sbox_deg_3[i] = calculated_sbox_deg_3;
        }

        sbox_deg_7[i] = sbox_deg_3[i].clone() * sbox_deg_3[i].clone() * add_rc[i].clone();
    }

    // Apply the linear layer.
    let mut state = sbox_deg_7;
    external_linear_layer_mut(&mut state);

    let next_state = if r == (NUM_EXTERNAL_ROUNDS / 2) - 1 {
        local_row.internal_rounds_state()
    } else if r == NUM_EXTERNAL_ROUNDS - 1 {
        local_row.perm_output()
    } else {
        &local_row.external_rounds_state()[r + 1]
    };

    for i in 0..WIDTH {
        builder.assert_eq(next_state[i], state[i].clone());
    }
}

/// Eval the constraints for the internal rounds.
pub fn eval_internal_rounds<AB>(builder: &mut AB, local_row: &dyn Poseidon2Cols<AB::Var>)
where
    AB: MachineAirBuilder + PairBuilder,
{
    let state = &local_row.internal_rounds_state();
    let s0 = local_row.internal_rounds_s0();
    let mut state: [AB::Expr; WIDTH] = core::array::from_fn(|i| state[i].into());
    for r in 0..NUM_INTERNAL_ROUNDS {
        // Add the round constant.
        let round = r + NUM_EXTERNAL_ROUNDS / 2;
        let add_rc = if r == 0 { state[0].clone() } else { s0[r - 1].into() } +
            AB::Expr::from_u32(RC_16_30_U32[round][0]);

        let mut sbox_deg_3 = add_rc.clone() * add_rc.clone() * add_rc.clone();
        if let Some(internal_sbox) = local_row.internal_rounds_sbox() {
            builder.assert_eq(internal_sbox[r], sbox_deg_3);
            sbox_deg_3 = internal_sbox[r].into();
        }

        // See `populate_internal_rounds` for why we don't have columns for the sbox output
        // here.
        let sbox_deg_7 = sbox_deg_3.clone() * sbox_deg_3.clone() * add_rc.clone();

        // Apply the linear layer.
        // See `populate_internal_rounds` for why we don't have columns for the new state here.
        state[0] = sbox_deg_7.clone();
        internal_linear_layer_mut(&mut state);

        if r < NUM_INTERNAL_ROUNDS - 1 {
            builder.assert_eq(s0[r], state[0].clone());
        }
    }

    let external_state = local_row.external_rounds_state()[NUM_EXTERNAL_ROUNDS / 2];
    for i in 0..WIDTH {
        builder.assert_eq(external_state[i], state[i].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //use p3_field::PrimeCharacteristicRing;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    #[test]
    fn test_local_external_linear_layer() {
        let mut input: [BabyBear; 16] = BabyBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);

        //sp1
        let expected: [BabyBear; 16] = BabyBear::new_array([
            1977003202, 275423229, 149669171, 119092734, 212141362, 57409241, 2003712403, 1377784331, 
            1746930307, 1564946509, 272867965, 839322770, 1247853484, 474497709, 1648233987, 1050217978
            ]);

         let ret = external_linear_layer(&input);
         
        assert_eq!(ret, expected); 
    }

     #[test]
    fn test_local_internal_linear_layer() {
        let mut input: [BabyBear; 16] = BabyBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);

        let expected: [BabyBear; 16] = BabyBear::new_array([
            925407769, 1551681711, 722526423, 315099541, 1203865770, 431153091, 867214325, 872835009, 
            1290679418, 1417814860, 1721490076, 833828430, 1180381619, 422160376, 344105810, 524430026
            ]);

        internal_linear_layer_mut(&mut input);
        assert_eq!(input, expected); 
    }
}


