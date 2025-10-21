#include <cstdint>
#include <cstdio>

#include "bb31_t.hpp" // For BabyBear type
#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "alu_ext.hpp"

// Assuming CUDA runtime headers are available
#include <cuda_runtime.h>

using namespace sp1_recursion_core_sys;
typedef bb31_t BabyBear;

// Kernel for processing main trace events
__global__ void kernel_process_alu_ext_events_gpu(
                const ExtAluIo<Block<BabyBear>>* events_d,  
                BabyBear* output_d,        
                uint32_t num_events,            
                uint32_t num_cols_per_row ) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one event per thread if within bounds
    if (idx < num_events) {
        ExtAluIo<Block<BabyBear>> event = events_d[idx];        
        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row;

        ExtAluValueCols<BabyBear>& cols = *reinterpret_cast<ExtAluValueCols<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::alu_ext::event_to_row<BabyBear>(event, cols);

    }
}

// Kernel for processing preprocessed instructions
__global__ void kernel_process_alu_ext_instructions_gpu(
                const ExtAluInstr<BabyBear>* instrs_d,  
                BabyBear* output_d, 
                uint32_t num_instrs,              
                uint32_t num_cols_per_row ) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one instruction per thread if within bounds
    if (idx < num_instrs) {
        ExtAluInstr<BabyBear> instr = instrs_d[idx];        
        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row;

        ExtAluAccessCols<BabyBear>& cols = *reinterpret_cast<ExtAluAccessCols<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::alu_ext::instr_to_row<BabyBear>(instr, cols);
    }
}


extern "C" void process_alu_ext_events_gpu(
        const ExtAluIo<Block<BabyBear>>* events_h,
        uint32_t events_len,  
        BabyBear* output_d, 
        uint32_t output_len,              
        int num_cols_per_row                        
) {
    if (events_len == 0) return; 
    ExtAluIo<Block<BabyBear>> *events_d = nullptr; 
    //BabyBear *output_d = nullptr; 
    cudaError_t err = cudaSuccess; 

    uint32_t input_size_bytes;
    uint32_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (uint32_t)events_len * sizeof(ExtAluIo<Block<BabyBear>>); // Assignment here
    err = cudaMalloc(&events_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_events: cudaMalloc events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(events_d, events_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_events: cudaMemcpy events_h to events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    //output_size_bytes = (uint32_t)events_len * num_cols_per_row * sizeof(BabyBear);
    //err = cudaMalloc(&output_d, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    // Optional: Initialize device output memory to zero
    //err = cudaMemset(output_d, 0, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_events: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (events_len + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_alu_ext_events_gpu<<<num_blocks, threads_per_block>>>(
        events_d, output_d, events_len, num_cols_per_row);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_alu_ext_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    //err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_events: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (events_d) cudaFree(events_d);
    //if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_alu_ext_events: CUDA error : %s\n", cudaGetErrorString(err));
    }
}

extern "C" void process_alu_ext_instructions_gpu(
        const ExtAluInstr<BabyBear>* instrs_h, 
        uint32_t instrs_len,  
        BabyBear* output_d, 
        uint32_t output_len,              
        int num_cols_per_row                        
) {
    if (instrs_len == 0) return;

    ExtAluInstr<BabyBear> *instructions_d = nullptr;
    //BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;


    uint32_t input_size_bytes;
    uint32_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (uint32_t)instrs_len * sizeof(ExtAluInstr<BabyBear>); 
    err = cudaMalloc(&instructions_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_instructions: cudaMalloc instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(instructions_d, instrs_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_instructions: cudaMemcpy instructions_h to instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    //output_size_bytes = instrs_len * num_cols_per_row * sizeof(BabyBear); // Assignment here
    //err = cudaMalloc(&output_d, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_instructions: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Optional: Initialize device output memory to zero
    //err = cudaMemset(output_d, 0, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_instructions: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (instrs_len + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_alu_ext_instructions_gpu<<<num_blocks, threads_per_block>>>(
        instructions_d, output_d, instrs_len, num_cols_per_row);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_alu_ext_instructions_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_instructions: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    //err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_ext_instructions: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (instructions_d) cudaFree(instructions_d);
    //if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_alu_ext_instructions: CUDA error: %s\n", cudaGetErrorString(err));
    }
}