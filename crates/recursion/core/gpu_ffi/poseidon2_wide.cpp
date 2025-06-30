#include <cstdint>
#include <cstdio>

#include "bb31_t.hpp" // For BabyBear type
#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "poseidon2_wide.hpp"

// Assuming CUDA runtime headers are available
#include <cuda_runtime.h>

using namespace sp1_recursion_core_sys;
typedef bb31_t BabyBear;

// Kernel for processing main trace events
__global__ void kernel_process_poseidon2_wide_events_gpu(
                const Poseidon2Event<BabyBear>* events_d,
                BabyBear* output_d,
                const BabyBear* dummy_event_d,
                uint32_t num_events,
                uint32_t total_events, // padded events
                uint32_t num_cols_per_row,
                bool sbox_state_flag,
                uint32_t width       
) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one event per thread if within bounds
    if (idx < total_events) {
        BabyBear* cols_ptr = output_d + idx * num_cols_per_row;
        
        const BabyBear* input_ptr; 

        if (idx >= num_events){
            input_ptr = dummy_event_d;
        } else {
            const Poseidon2Event<BabyBear>  event = events_d[idx];
            input_ptr = event.input;
        }

        sp1_recursion_core_sys::poseidon2_wide::event_to_row<BabyBear>(reinterpret_cast<const bb31_t*>(input_ptr), reinterpret_cast<bb31_t*>(cols_ptr), 0, 1, sbox_state_flag);
    }
}


// Kernel for processing preprocessed instructions
__global__ void kernel_process_poseidon2_wide_instructions_gpu(
                const Poseidon2SkinnyInstr<BabyBear>* instrs_d,  
                BabyBear* output_d, 
                uint32_t num_instrs,              
                uint32_t num_cols_per_row ) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one instruction per thread if within bounds
    if (idx < num_instrs) {
        Poseidon2SkinnyInstr<BabyBear> instr = instrs_d[idx];
        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row ;

        Poseidon2PreprocessedColsWide<BabyBear>& cols = *reinterpret_cast<Poseidon2PreprocessedColsWide<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::poseidon2_wide::instr_to_row<BabyBear>(instr, cols);
    }
}


extern "C" void process_poseidon2_wide_events_gpu(
        const Poseidon2Event<BabyBear>* events_h, 
        uint32_t events_len,  
        BabyBear* output_h, 
        uint32_t output_len,  
        uint32_t total_rows,
        uint32_t num_cols_per_row,
        bool   sbox_state,
        uint32_t width                      
) {
    if (events_len == 0) return; 
    
    Poseidon2Event<BabyBear> *events_d = nullptr; 
    BabyBear *output_d = nullptr; 
    BabyBear * dummy_event_d = nullptr;
    cudaError_t err = cudaSuccess; 
    //BabyBear dummy_event_h[width];
    //memset(dummy_event_h, 0, sizeof(dummy_event_h));

    uint32_t input_size_bytes;
    uint32_t output_size_bytes;
    uint32_t dummy_event_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (uint32_t)events_len * sizeof(Poseidon2Event<BabyBear>); // Assignment here
    err = cudaMalloc(&events_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaMalloc  events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    dummy_event_size_bytes = width * sizeof(BabyBear); 
    err = cudaMalloc(&dummy_event_d, dummy_event_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaMalloc  dummy_event_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    //err = cudaMemcpy(dummy_event_d, dummy_event_h, dummy_event_size_bytes, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaMemcpy dummy_event_h  failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemset(dummy_event_d, 0, dummy_event_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaMemset  dummy_event_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(events_d, events_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaMemcpy events_h to events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    output_size_bytes = (uint32_t)total_rows * num_cols_per_row * sizeof(BabyBear); // Assignment here
    err = cudaMalloc(&output_d, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    // Optional: Initialize device output memory to zero
   
    err = cudaMemset(output_d, 0, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (total_rows + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_poseidon2_wide_events_gpu<<<num_blocks, threads_per_block>>>(
        events_d, 
        output_d, 
        dummy_event_d, 
        events_len,
        total_rows, 
        num_cols_per_row,
        sbox_state,
        width);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_poseidon2_wide_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_events: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (events_d) cudaFree(events_d);
    if (dummy_event_d) cudaFree(dummy_event_d);
    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_poseidon2_wide_events: CUDA error  : %s\n", cudaGetErrorString(err));
    }
}

extern "C" void process_poseidon2_wide_instructions_gpu(
        const Poseidon2SkinnyInstr<BabyBear>* instrs_h, 
        uint32_t instrs_len,  
        BabyBear* output_h, 
        uint32_t output_len,              
        int num_cols_per_row                        
) {
    if (instrs_len == 0) return; // Nothing to process

    Poseidon2SkinnyInstr<BabyBear> *instructions_d = nullptr;
    BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;

    // Declare all other variables that might be bypassed by goto
    uint32_t input_size_bytes;
    uint32_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (uint32_t)instrs_len * sizeof(Poseidon2SkinnyInstr<BabyBear>); 
    err = cudaMalloc(&instructions_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_instructions: cudaMalloc instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(instructions_d, instrs_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_instructions: cudaMemcpy instructions_h to instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    output_size_bytes = instrs_len * num_cols_per_row * sizeof(BabyBear); // Assignment here
    err = cudaMalloc(&output_d, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_instructions: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Optional: Initialize device output memory to zero
    err = cudaMemset(output_d, 0, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_instructions: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (instrs_len + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_poseidon2_wide_instructions_gpu<<<num_blocks, threads_per_block>>>(
        instructions_d, output_d, instrs_len, num_cols_per_row);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_poseidon2_wide_instructions_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_instructions: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_wide_instructions: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (instructions_d) cudaFree(instructions_d);
    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_poseidon2_wide_instructions: CUDA error : %s\n", cudaGetErrorString(err));
    }
}