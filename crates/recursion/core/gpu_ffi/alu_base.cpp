#include <cstdint>
#include <cstdio>

#include "bb31_t.hpp" // For BabyBear type
#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "alu_base.hpp"

// Assuming CUDA runtime headers are available
#include <cuda_runtime.h>

using namespace sp1_recursion_core_sys;
typedef bb31_t BabyBear;

// Kernel for processing main trace events
__global__ void kernel_process_alu_base_events_gpu(
                const BaseAluIo<BabyBear>* events_d,  
                BabyBear* output_d,        
                uint32_t num_events,            
                uint32_t num_cols_per_row ) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one event per thread if within bounds
    if (idx < num_events) {
        //int row_idx = idx ;
        BaseAluIo<BabyBear> event = events_d[idx];        
        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row;

        BaseAluValueCols<BabyBear>& cols = *reinterpret_cast<BaseAluValueCols<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::alu_base::event_to_row<BabyBear>(event, cols);

    }
}

// Kernel for processing preprocessed instructions
__global__ void kernel_process_alu_base_instructions_gpu(
                const BaseAluInstr<BabyBear>* instrs_d,  
                BabyBear* output_d, 
                uint32_t num_instrs,              
                uint32_t num_cols_per_row ) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one instruction per thread if within bounds
    if (idx < num_instrs) {
        BaseAluInstr<BabyBear> instr = instrs_d[idx];        
        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row;

        //if (idx==0){
        //    printf("****alu_base_kernl, op_code:%u, mult:%u, out:%u", instr.opcode, instr.mult,instr.addrs.out);

       // }

        BaseAluAccessCols<BabyBear>& cols = *reinterpret_cast<BaseAluAccessCols<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::alu_base::instr_to_row<BabyBear>(instr, cols);
    }
}


extern "C" void process_alu_base_events_gpu(
        const BaseAluIo<BabyBear>* events_h, 
        size_t events_len,  
        BabyBear* output_d, 
        size_t output_len,              
        int num_cols_per_row                        
) {
    if (events_len == 0) return; // Nothing to process
    BaseAluIo<BabyBear> *events_d = nullptr; // Declare at top
    //BabyBear *output_d = nullptr; // Declare at top
    cudaError_t err = cudaSuccess; // Initialize err at top

    // Declare all other variables that might be bypassed by goto
    size_t input_size_bytes;
    size_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (size_t)events_len * sizeof(BaseAluIo<BabyBear>); // Assignment here
    err = cudaMalloc(&events_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_alu_base_events: cudaMalloc events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(events_d, events_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_alu_base_events: cudaMemcpy events_h to events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    //output_size_bytes = (size_t)events_len * num_cols_per_row * sizeof(BabyBear); // Assignment here
    //err = cudaMalloc(&output_d, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_base_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    // Optional: Initialize device output memory to zero
    //err = cudaMemset(output_d, 0, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_base_events: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (events_len + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_alu_base_events_gpu<<<num_blocks, threads_per_block>>>(
        events_d, output_d, events_len, num_cols_per_row);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_alu_base_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_alu_base_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    //err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_base_events: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (events_d) cudaFree(events_d);
    //if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_alu_base_events: CUDA error : %s\n", cudaGetErrorString(err));
    }
}

extern "C" void process_alu_base_instructions_gpu(
        const BaseAluInstr<BabyBear>* instrs_h, 
        size_t instrs_len,  
        BabyBear* output_d, 
        size_t output_len,              
        int num_cols_per_row                        
) {
    if (instrs_len == 0) return; // Nothing to process
//printf("----alu_base_cpp, op_code:%u, mult:%u, out:%u", instrs_h[0].opcode, instrs_h[0].mult,instrs_h[0].addrs.out);
    BaseAluInstr<BabyBear> *instructions_d = nullptr;
    //BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;

    // Declare all other variables that might be bypassed by goto
    size_t input_size_bytes;
    size_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (size_t)instrs_len * sizeof(BaseAluInstr<BabyBear>); 
    err = cudaMalloc(&instructions_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_alu_base_instructions: cudaMalloc instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(instructions_d, instrs_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_alu_base_instructions: cudaMemcpy instructions_h to instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    //output_size_bytes = instrs_len * num_cols_per_row * sizeof(BabyBear); // Assignment here
    //err = cudaMalloc(&output_d, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_base_instructions: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Optional: Initialize device output memory to zero
    //err = cudaMemset(output_d, 0, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_base_instructions: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (instrs_len + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_alu_base_instructions_gpu<<<num_blocks, threads_per_block>>>(
        instructions_d, output_d, instrs_len, num_cols_per_row);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_alu_base_instructions_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_alu_base_instructions: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    //err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) { fprintf(stderr, "_alu_base_instructions: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (instructions_d) cudaFree(instructions_d);
    //if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_alu_base_instructions: CUDA error : %s\n", cudaGetErrorString(err));
        // Depending on error handling strategy, propagate this error
    }
}