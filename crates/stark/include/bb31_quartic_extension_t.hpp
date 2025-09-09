#pragma once
#include "bb31_t.hpp"
//#include "sp1_core_types.hpp"

// Equivalent to sp1_core::air::Word<T>
template <typename T>
struct Word {
    T words[4];
};

// Equivalent to sp1_core::air::Block<T>
template <typename T>
struct Block {
    T _0[4]; // In SP1 recursion, D=4
};


// Performs (a0, a1) * (b0, b1) in F[y]/(y^2-w)
__host__ __device__ inline void quadratic_mul(
    const bb31_t a[2], const bb31_t b[2], bb31_t out[2], bb31_t w)
{
    // out0 = a0*b0 + a1*b1*w
    // out1 = a0*b1 + a1*b0
    out[0] = a[0] * b[0] + a[1] * b[1] * w;
    out[1] = a[0] * b[1] + a[1] * b[0];
}

// Performs 1 / (a0, a1) in F[y]/(y^2-w)
__host__ __device__ inline void quadratic_inv(const bb31_t a[2], bb31_t out[2], bb31_t w) {
    bb31_t neg_a1 = bb31_t(0) - a[1];
    bb31_t denominator_inv = (a[0] * a[0] - a[1] * neg_a1 * w).reciprocal();
    out[0] = a[0] * denominator_inv;
    out[1] = neg_a1 * denominator_inv;
}


// Represents an element of the 4th-degree binomial extension field over bb31_t.
// Mathematically equivalent to plonky3's BinomialExtensionField<BabyBear, 4>.
struct bb31_quartic_extension_t {
     bb31_t coeffs[4];
    
    __host__ __device__ static bb31_t W() { return bb31_t(11); }
    
    // Generator for the entire extension field.
    __host__ __device__ static bb31_quartic_extension_t generator() {
        bb31_quartic_extension_t g;
        g.coeffs[0] = bb31_t::from_canonical_u32(8);
        g.coeffs[1] = bb31_t::from_canonical_u32(1);
        g.coeffs[2] = bb31_t::from_canonical_u32(0);
        g.coeffs[3] = bb31_t::from_canonical_u32(0);
        return g;
    }

    // --- Two-Adic Field Trait Implementation ---
    __host__ __device__ static bb31_quartic_extension_t two_adic_generator(int bits) {
        if (bits == 29) {
            // Special case for the largest subgroup
            bb31_quartic_extension_t g;
            g.coeffs[0] = bb31_t(0);
            g.coeffs[1] = bb31_t(0);
            g.coeffs[2] = bb31_t(1996171314);
            g.coeffs[3] = bb31_t(0);
            return g;
        }
        // For other subgroups, it's derived from DTH_ROOT.
        // DTH_ROOT = 1728404513
        bb31_t dth_root = bb31_t(1728404513);
        // The generator for 2^k subgroup is (DTH_ROOT)^(2^(29-k))
        // This requires a `pow` method on the base field element.
        // Let's use the existing `exp_power_of_2` from bb31_t.hpp
        bb31_t base = dth_root;
        int power = 29 - bits;
        if (power > 0) {
#ifdef __CUDA_ARCH__
            base = base.exp_power_of_two(power); // Device 
#else
            base = base.exp_power_of_2(power);   // Host 
#endif
        }
        return bb31_quartic_extension_t(base);
    }

    // --- Constructors ---
    __host__ __device__ bb31_quartic_extension_t() {
        for (int i=0; i<4; ++i) coeffs[i] = bb31_t(0);
    }
    __host__ __device__ bb31_quartic_extension_t(const bb31_t& base_val) {
        coeffs[0] = base_val;
        for (int i=1; i<4; ++i) coeffs[i] = bb31_t(0);
    }
    __host__ __device__ static bb31_quartic_extension_t one() {
        return bb31_quartic_extension_t(bb31_t::one());
    }

    __host__ __device__ static bb31_quartic_extension_t zero() {
        return bb31_quartic_extension_t(bb31_t::zero());
    }

    __host__ __device__ static bb31_quartic_extension_t two() {
        return bb31_quartic_extension_t(bb31_t::two());
    }
    
    /// Constructs a quartic extension element from its four base field coefficients.
    __host__ __device__ static bb31_quartic_extension_t from_base_slice(const bb31_t* base_coeffs) {
        bb31_quartic_extension_t result;
        for (int i = 0; i < 4; ++i) {
            result.coeffs[i] = base_coeffs[i];
        }
        return result;
    }

    /// A convenience helper to construct an extension element from a Block<Val>.
    __host__ __device__ static bb31_quartic_extension_t from_block(const Block<bb31_t>& block) {
        // block.words is already an array of 4 `Val` (which is bb31_t).
        return from_base_slice(block._0);
    }

    __host__ __device__ bb31_quartic_extension_t(
        const bb31_t& c0,
        const bb31_t& c1,
        const bb31_t& c2,
        const bb31_t& c3
    ) {
        coeffs[0] = c0;
        coeffs[1] = c1;
        coeffs[2] = c2;
        coeffs[3] = c3;
   }

    // --- Operators ---
    __host__ __device__ bb31_quartic_extension_t operator+(const bb31_quartic_extension_t& other) const {
        bb31_quartic_extension_t result;
        for (int i=0; i<4; ++i) result.coeffs[i] = this->coeffs[i] + other.coeffs[i];
        return result;
    }
    __host__ __device__ bb31_quartic_extension_t& operator+=(const bb31_quartic_extension_t& other) {
        for (int i=0; i<4; ++i) this->coeffs[i] += other.coeffs[i];
        return *this;
    }

    __host__ __device__ bb31_quartic_extension_t operator-(const bb31_quartic_extension_t& other) const {
        bb31_quartic_extension_t result;
        for (int i=0; i<4; ++i) result.coeffs[i] = this->coeffs[i] - other.coeffs[i];
        return result;
    }

     __host__ __device__ bb31_quartic_extension_t& operator/=(const bb31_quartic_extension_t& other) {
        *this *= other.reciprocal();
        return *this;
    }

    __host__ __device__ bb31_quartic_extension_t operator*(const bb31_t& scalar) const {
        bb31_quartic_extension_t result;
        for (int i=0; i<4; ++i) result.coeffs[i] = this->coeffs[i] * scalar;
        return result;
    }

    __host__ __device__ bb31_quartic_extension_t operator*(const bb31_quartic_extension_t& b) const {
        const bb31_t* a = this->coeffs;
        bb31_t w = W();
        bb31_t c[7] = {bb31_t(0)}; // c = a * b before reduction
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                c[i+j] += a[i] * b.coeffs[j];
            }
        }
        
        // Reduction using irreducible polynomial x^4 - w = 0  => x^4 = w
        bb31_quartic_extension_t res;
        res.coeffs[0] = c[0] + c[4] * w;
        res.coeffs[1] = c[1] + c[5] * w;
        res.coeffs[2] = c[2] + c[6] * w;
        res.coeffs[3] = c[3];
        
        return res;
    }

     __host__ __device__ bb31_quartic_extension_t& operator*=(const bb31_quartic_extension_t& other) {
        // The `*` operator returns a new object. We assign it back to `this`.
        *this = *this * other;
        return *this;
    }

    __host__ __device__ bb31_quartic_extension_t pow(uint64_t n) const {
        bb31_quartic_extension_t result = bb31_quartic_extension_t::one();
        bb31_quartic_extension_t base = *this;

        if (n == 0) {
            return result;
        }

        while (n > 0) {
            // If n is odd, multiply result with base
            if (n % 2 == 1) {
                result *= base;
            }
            // n must be even now
            n /= 2;
            // base becomes base^2
            base *= base;
        }
        return result;
    }

    // Reciprocal (Inverse).
    __host__ __device__ bb31_quartic_extension_t reciprocal() const {
        const bb31_t* a = this->coeffs;
        bb31_t w = W();

        // Step 1: Compute the norm wrt the quadratic subfield F[y]/(y^2-w)
        // This logic is from plonky3's quartic_inv and is correct.
        // It requires a temporary quadratic field `y = X^2`, but the irreducible
        // polynomial for this is NOT x^2-W. It's more complex.
        // The `quartic_inv` logic provided by plonky3 relies on this structure.
        // Let's re-implement it carefully.
        bb31_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
        
        // Let A = (a0, a2) and B = (a1, a3) be elements in F[y]/(y^2-w).
        // We want to compute 1 / (A + B*X). This is (A - B*X) / (A^2 - B^2*X^2).
        // X^2 is not a base field element. We need to be careful.
        // The irreducible polynomial is x^4 - W = 0.
        
        // Let's use a more direct method: Extended Euclidean Algorithm or Fermat's Little Theorem.
        // For a field of size p^4, inverse is pow(p^4 - 2). This is too slow.
        // `plonky3`'s `quartic_inv` is the way to go. Let's trust its math.
        
        // Re-transcribing `quartic_inv` logic:
        // norm = (a0 + a2 X^2)^2 - (a1 + a3 X^2)^2 * X^2
        // X^2 in the quadratic extension is just a variable `y`.
        // The irreducible polynomial for the quadratic extension is y^2 - w.
        
        // norm_quad = (a0,a2)^2 - (a1,a3)^2 * (w,0)
        bb31_t a0_sq = a0*a0;
        bb31_t a1_sq = a1*a1;
        bb31_t a2_sq = a2*a2;
        bb31_t a3_sq = a3*a3;

        bb31_t two_a0a2 = (a0*a2) + (a0*a2);
        bb31_t two_a1a3 = (a1*a3) + (a1*a3);

        bb31_t norm_c0 = a0_sq + w*a2_sq - w*two_a1a3;
        bb31_t norm_c1 = two_a0a2 - a1_sq - w*a3_sq;

        // Now we need inv(norm_c0 + norm_c1 * X^2) in F[X^2]
        // This is a quadratic extension inverse.
        bb31_t norm_det_inv = (norm_c0*norm_c0 - norm_c1*norm_c1*w).reciprocal();
        bb31_t inv_norm_c0 = norm_c0 * norm_det_inv;
        bb31_t inv_norm_c1 = -norm_c1 * norm_det_inv;

        // Final result = ( (a0,a2) - (a1,a3)*X ) * (inv_norm_c0, inv_norm_c1)
        bb31_quartic_extension_t res;
        bb31_t t0 = a0*inv_norm_c0 + a2*inv_norm_c1*w;
        bb31_t t1 = a1*inv_norm_c0 + a3*inv_norm_c1*w;
        bb31_t t2 = a0*inv_norm_c1 + a2*inv_norm_c0;
        bb31_t t3 = a1*inv_norm_c1 + a3*inv_norm_c0;

        res.coeffs[0] = t0;//- w*t3;
        res.coeffs[1] = -t1;
        res.coeffs[2] = t2;
        res.coeffs[3] = -t3;

        return res;
    }
};

// Allow `scalar * ext`
__host__ __device__ inline bb31_quartic_extension_t operator*(const bb31_t& scalar, const bb31_quartic_extension_t& ext) {
    return ext * scalar;
}

__host__ __device__ inline bb31_quartic_extension_t operator/(
    bb31_quartic_extension_t lhs, 
    const bb31_quartic_extension_t& rhs
) {
    lhs /= rhs; 
    return lhs;
}