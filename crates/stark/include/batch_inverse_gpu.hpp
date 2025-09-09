// file: batch_inverse_gpu.hpp
#pragma once
#include "gpu_types.hpp"
#include <vector>

/**
 * @brief Performs a batch multiplicative inverse on an array of field elements on the GPU.
 * @param d_in A device pointer to the input array of `Val`s.
 * @param n The number of elements in the array.
 * @return A new device pointer to the array of inverses. The caller is responsible for freeing this memory.
 */
Val* batch_multiplicative_inverse_gpu(const Val* d_in, int n);

//From plonky3: batch_multiplicative_inverse_general
std::vector<Val> batch_multiplicative_inverse_cpu(const std::vector<Val>& x) {
    int n = x.size();
    if (n == 0) {
        return std::vector<Val>();
    }

    // This is a direct, line-by-line translation of the Rust `batch_multiplicative_inverse_general`.
    // The vector `products` here serves the same purpose as `result` in the Rust code.
    std::vector<Val> products_then_inverses(n);
    
    products_then_inverses[0] = Val::one();
    for (int i = 1; i < n; ++i) {
        products_then_inverses[i] = products_then_inverses[i - 1] * x[i - 1];
    }
   

    // 2. Compute the inverse of the total product.
    Val total_product = products_then_inverses[n - 1] * x[n - 1];
  
    if (total_product.is_zero()) {
        printf("ERROR in batch_multiplicative_inverse_cpu: trying to invert a zero element.\n");
        return std::vector<Val>(n, Val::zero());
    }
    Val inv = total_product.reciprocal();
 
    // 3. Suffix scan to compute individual inverses.
    // We will now modify the `products` vector in place, from back to front,
    // to transform it from prefix products into the final inverses.
   for (int i = n - 1; i >= 0; --i) {
        products_then_inverses[i] *= inv;
        inv = inv * x[i];
    }
    
    return products_then_inverses;
}