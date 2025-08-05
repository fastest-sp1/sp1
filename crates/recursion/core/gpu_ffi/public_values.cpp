#include <cstdint>
#include <cstdio>

#include "bb31_t.hpp" // For BabyBear type
#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "public_values.hpp"

// Assuming CUDA runtime headers are available
#include <cuda_runtime.h>


using namespace sp1_recursion_core_sys;
typedef bb31_t BabyBear;

// Kernel for processing main trace events
__global__ void kernel_process_public_values_events_gpu(
                const CommitPublicValuesEvent<BabyBear>* events_d,  
                BabyBear* output_d,        
                uint32_t digit_size,            
                uint32_t num_value_cols ) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one event per thread if within bounds
    if (idx < digit_size) {
        CommitPublicValuesEvent<BabyBear> event = events_d[0];        
        BabyBear* target_output_ptr = output_d + idx * num_value_cols;

        PublicValuesCols<BabyBear>& cols = *reinterpret_cast<PublicValuesCols<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::public_values::event_to_row<BabyBear>(event, idx, cols);

    }
}

// Kernel for processing preprocessed instructions
__global__ void kernel_process_public_values_instructions_gpu(
                const CommitPublicValuesInstr<BabyBear>* instrs_d,  
                BabyBear* output_d, 
                uint32_t digit_size,              
                uint32_t num_value_cols ) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one instruction per thread if within bounds
    if (idx < digit_size) {
        CommitPublicValuesInstr<BabyBear> instr = instrs_d[0];        
        BabyBear* target_output_ptr = output_d + idx * num_value_cols;

        PublicValuesPreprocessedCols<BabyBear>& cols = *reinterpret_cast<PublicValuesPreprocessedCols<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::public_values::instr_to_row<BabyBear>(instr, idx, cols);
    }
}


extern "C" void process_public_values_events_gpu(
        const CommitPublicValuesEvent<BabyBear>* events_h, 
        size_t events_len,  
        BabyBear* output_h, 
        size_t output_len, 
        size_t digit_size,             
        size_t num_value_cols                        
) {
    if (events_len == 0) return; // Nothing to process
    CommitPublicValuesEvent<BabyBear> *events_d = nullptr; 
    BabyBear *output_d = nullptr; // Declare at top
    cudaError_t err = cudaSuccess; // Initialize err at top

    // Declare all other variables that might be bypassed by goto
    size_t input_size_bytes;
    size_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks=0;

    // 1. Allocate device memory for input
    input_size_bytes = (size_t)events_len * sizeof(CommitPublicValuesEvent<BabyBear>); 
    err = cudaMalloc(&events_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_events: cudaMalloc events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(events_d, events_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_events: cudaMemcpy events_h to events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    output_size_bytes = (size_t)digit_size * num_value_cols * sizeof(BabyBear); 
    err = cudaMalloc(&output_d, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    // Optional: Initialize device output memory to zero
    err = cudaMemset(output_d, 0, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_events: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (digit_size + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_public_values_events_gpu<<<num_blocks, threads_per_block>>>(
        events_d, output_d, digit_size, num_value_cols);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_alu_base_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_events: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (events_d) cudaFree(events_d);
    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_public_values_events: CUDA error in process_public_values_events_gpu_ffi: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void process_public_values_instructions_gpu(
        const CommitPublicValuesInstr<BabyBear>* instrs_h, 
        size_t instrs_len,  
        BabyBear* output_h, 
        size_t output_len,
        size_t digit_size,             
        int num_value_cols                        
) {
    if (instrs_len == 0) return; 
    CommitPublicValuesInstr<BabyBear> *instructions_d = nullptr;
    BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;

    // Declare all other variables that might be bypassed by goto
    size_t input_size_bytes;
    size_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (size_t)instrs_len * sizeof(CommitPublicValuesInstr<BabyBear>); 
    err = cudaMalloc(&instructions_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_instrs: cudaMalloc instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(instructions_d, instrs_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_instrs: cudaMemcpy instructions_h to instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    output_size_bytes = digit_size * num_value_cols * sizeof(BabyBear); // Assignment here
    err = cudaMalloc(&output_d, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_instrs: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Optional: Initialize device output memory to zero
    err = cudaMemset(output_d, 0, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_instrs: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (digit_size + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_public_values_instructions_gpu<<<num_blocks, threads_per_block>>>(
        instructions_d, output_d, digit_size, num_value_cols);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_public_values_instructions_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_instrs: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_public_values_instrs: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (instructions_d) cudaFree(instructions_d);
    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_public_values_instrs: CUDA error : %s\n", cudaGetErrorString(err));
        // Depending on error handling strategy, propagate this error
    }
}