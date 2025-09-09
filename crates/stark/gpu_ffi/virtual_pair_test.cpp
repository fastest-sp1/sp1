// file: gpu_ffi/virtual_col_logic_test.cu

#include "virtual_pair_col.hpp" // Our C++ implementation of VirtualPairCol
#include <cstdio>
#include "gpu_types.hpp"


// --- Test Kernels ---

__global__ void test_vpc_constant_kernel(const Val* d_main_row, const Val* d_prep_row, Val* d_result) {
    // This kernel is simple: it only has one thread.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 1. Construct the VPC object on the GPU stack using the C++ API.
        VirtualPairCol vpc = VirtualPairCol::from_constant(Val::from_canonical_u32(42));
        // 2. Call the C++ `apply` method.
        *d_result = vpc.apply(d_main_row, d_prep_row);
    }
}

__global__ void test_vpc_single_main_kernel(const Val* d_main_row, const Val* d_prep_row, Val* d_result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        VirtualPairCol vpc = VirtualPairCol::single_main(5); // Test getting main_row[5]
        *d_result = vpc.apply(d_main_row, d_prep_row);
    }
}

__global__ void test_vpc_complex_kernel(const Val* d_main_row, const Val* d_prep_row, Val* d_result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Emulates: 2 * main[1] + 5 * prep[2] + 100
        VirtualPairCol term1 = VirtualPairCol::single_main(1);
        term1.column_weights[0].second = Val::from_canonical_u32(2); // Inefficient, but works for test

        VirtualPairCol term2 = VirtualPairCol::single_preprocessed(2);
        term2.column_weights[0].second = Val::from_canonical_u32(5);

        VirtualPairCol term3 = VirtualPairCol::from_constant(Val::from_canonical_u32(100));

        VirtualPairCol vpc = term1 + term2 + term3;
        
        *d_result = vpc.apply(d_main_row, d_prep_row);
    }
}

#define BASE_ALU_PREP_COLS_ACCESSES_0_IS_ADD_OFFSET (offsetof(BaseAluPreprocessedCols<Val>, accesses[0].is_add))
#define BASE_ALU_PREP_COLS_ACCESSES_0_IS_SUB_OFFSET (offsetof(BaseAluPreprocessedCols<Val>, accesses[0].is_sub))
#define BASE_ALU_PREP_COLS_ACCESSES_0_IS_MUL_OFFSET (offsetof(BaseAluPreprocessedCols<Val>, accesses[0].is_mul))
#define BASE_ALU_PREP_COLS_ACCESSES_0_IS_DIV_OFFSET (offsetof(BaseAluPreprocessedCols<Val>, accesses[0].is_div))

__global__ void test_vpc_is_real_kernel(const Val* d_main_row, const Val* d_prep_row, Val* d_result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // This kernel tests if our C++ `apply` function can correctly evaluate
        // a VirtualPairCol with multiple terms, which is what is_real becomes.

        // Manually construct the VirtualPairCol representing `is_add + is_sub + is_mul + is_div`
        // for the first operation slot (i=0).
        VirtualPairCol vpc_is_real;
        vpc_is_real.num_weights = 4;
        vpc_is_real.constant = Val::zero();

        // Hardcoded column indices for BaseAluPreprocessedCols.accesses[0]
        int is_add_idx = 3; 
        int is_sub_idx = 4;
        int is_mul_idx = 5;
        int is_div_idx = 6;

        vpc_is_real.column_weights[0] = {{PairColType::Preprocessed, is_add_idx}, Val::one()};
        vpc_is_real.column_weights[1] = {{PairColType::Preprocessed, is_sub_idx}, Val::one()};
        vpc_is_real.column_weights[2] = {{PairColType::Preprocessed, is_mul_idx}, Val::one()};
        vpc_is_real.column_weights[3] = {{PairColType::Preprocessed, is_div_idx}, Val::one()};
        
        *d_result = vpc_is_real.apply(d_main_row, d_prep_row);
    }
}

// --- Helper to run a test case ---
Val run_test(const void* kernel, const Val* h_main_row, const Val* h_prep_row, int main_width, int prep_width) {
    Val* d_main_row;
    Val* d_prep_row;
    Val* d_result;

    cudaMalloc(&d_main_row, main_width * sizeof(Val));
    cudaMalloc(&d_prep_row, prep_width * sizeof(Val));
    cudaMalloc(&d_result, sizeof(Val));

    cudaMemcpy(d_main_row, h_main_row, main_width * sizeof(Val), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prep_row, h_prep_row, prep_width * sizeof(Val), cudaMemcpyHostToDevice);

    void* args[] = {&d_main_row, &d_prep_row, &d_result};
    cudaLaunchKernel(kernel, 1, 1, args, 0, 0);
    
    cudaDeviceSynchronize();

    Val h_result;
    cudaMemcpy(&h_result, d_result, sizeof(Val), cudaMemcpyDeviceToHost);

    cudaFree(d_main_row);
    cudaFree(d_prep_row);
    cudaFree(d_result);

    return h_result;
}


// --- FFI Functions Exposed to Rust ---
extern "C" {
    Val test_vpc_case_constant(const Val* h_main, const Val* h_prep, int mw, int pw) {
        return run_test((void*)test_vpc_constant_kernel, h_main, h_prep, mw, pw);
    }
    Val test_vpc_case_single_main(const Val* h_main, const Val* h_prep, int mw, int pw) {
        return run_test((void*)test_vpc_single_main_kernel, h_main, h_prep, mw, pw);
    }
    Val test_vpc_case_complex(const Val* h_main, const Val* h_prep, int mw, int pw) {
        return run_test((void*)test_vpc_complex_kernel, h_main, h_prep, mw, pw);
    }

    Val test_vpc_case_is_real(const Val* h_main, const Val* h_prep, int mw, int pw) {
        return run_test((void*)test_vpc_is_real_kernel, h_main, h_prep, mw, pw);
    }
}