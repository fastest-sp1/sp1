#include <cstdint>
#include <cstdio>

#include "bb31_t.hpp" // For BabyBear type
#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "poseidon2_skinny.hpp"

// Assuming CUDA runtime headers are available
#include <cuda_runtime.h>


using namespace sp1_recursion_core_sys;
typedef bb31_t BabyBear;

// Kernel for processing main trace events
__global__ void kernel_process_poseidon2_skinny_events_gpu(
                const Poseidon2Event<BabyBear>* events_d,
                BabyBear* output_d,
                uint32_t num_events,
                uint32_t cols_per_row
) {
    // Global thread index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process one event per thread if within bounds
    if (idx < num_events) {
        const Poseidon2Event<BabyBear>& event = events_d[idx];        
        BabyBear* target_output_ptr = output_d + idx * cols_per_row ;

        //Poseidon2<BabyBear>& cols = *reinterpret_cast<Poseidon2<BabyBear>*>(target_output_ptr);
        Poseidon2<BabyBear>* cols = reinterpret_cast<Poseidon2<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::poseidon2_skinny::event_to_row<BabyBear>(event, cols);
    }
}

// Kernel for processing preprocessed instructions
__global__ void kernel_process_poseidon2_skinny_instructions_gpu(
                const Poseidon2Instr<BabyBear>* instrs_d,
                BabyBear* output_d,
                uint32_t total_rows,
                uint32_t rows_per_instr,  // each instruction generates rows (NUM_EXTERNAL_ROUNDS + 3)
                uint32_t cols_per_row,
                uint32_t original_instrs 
) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_rows) {
        // get the instruction which is relative to the idx
        uint32_t instr_idx = idx / rows_per_instr;

    
        uint32_t internal_row_idx = idx % rows_per_instr;

        if (instr_idx >= original_instrs) {
            return;
        }

        // get the instruction for the idx thread
        const Poseidon2Instr<BabyBear>& instr = instrs_d[instr_idx];
        
        BabyBear* target_output_ptr = output_d + idx * cols_per_row;

        Poseidon2PreprocessedColsSkinny<BabyBear>& cols = 
            *reinterpret_cast<Poseidon2PreprocessedColsSkinny<BabyBear>*>(target_output_ptr);


        sp1_recursion_core_sys::poseidon2_skinny::instr_to_row<BabyBear>(instr, internal_row_idx, cols);
    }
}

extern "C" void process_poseidon2_skinny_events_gpu(
        const Poseidon2Event<BabyBear>* events_h, 
        uint32_t events_len,  
        BabyBear* output_d, 
        uint32_t output_len,  
        uint32_t cols_per_row                    
) {
    if (events_len == 0) return; // Nothing to process
    
    Poseidon2Event<BabyBear> *events_d = nullptr; 
    //BabyBear *output_d = nullptr; 
    cudaError_t err = cudaSuccess; 
  

    uint32_t input_size_bytes;
    uint32_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (uint32_t)events_len * sizeof(Poseidon2Event<BabyBear>); // Assignment here
    err = cudaMalloc(&events_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_events: cudaMalloc  events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(events_d, events_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_events: cudaMemcpy events_h to events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    //output_size_bytes = (uint32_t)events_len * cols_per_row * sizeof(BabyBear); // Assignment here
    //err = cudaMalloc(&output_d, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    // Optional: Initialize device output memory to zero
   
    //err = cudaMemset(output_d, 0, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_events: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (events_len + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_poseidon2_skinny_events_gpu<<<num_blocks, threads_per_block>>>(
        events_d, 
        output_d, 
        events_len, 
        cols_per_row);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_poseidon2_skinny_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    //err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_events: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (events_d) cudaFree(events_d);
    //if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_poseidon2_skinny_events: CUDA error  : %s\n", cudaGetErrorString(err));
    }
}

extern "C" void process_poseidon2_skinny_instructions_gpu(
        const Poseidon2SkinnyInstr<BabyBear>* instrs_h, 
        uint32_t instrs_len,  
        BabyBear* output_d, 
        uint32_t output_len,  
        uint32_t rows_per_instr,            
        uint32_t cols_per_row                        
) {
    if (instrs_len == 0) return; // Nothing to process

    Poseidon2SkinnyInstr<BabyBear> *instructions_d = nullptr;
    //BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;

    // Declare all other variables that might be bypassed by goto
    uint32_t input_size_bytes;
    uint32_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;
    uint32_t total_output_rows = instrs_len * rows_per_instr;

    // 1. Allocate device memory for input
    input_size_bytes = (uint32_t)instrs_len * sizeof(Poseidon2SkinnyInstr<BabyBear>); 
    err = cudaMalloc(&instructions_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_instructions: cudaMalloc instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(instructions_d, instrs_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_instructions: cudaMemcpy instructions_h to instructions_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    //output_size_bytes = total_output_rows * cols_per_row * sizeof(BabyBear); // Assignment here
    //err = cudaMalloc(&output_d, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_instructions: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Optional: Initialize device output memory to zero
    //err = cudaMemset(output_d, 0, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_instructions: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (total_output_rows + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_poseidon2_skinny_instructions_gpu<<<num_blocks, threads_per_block>>>(
        instructions_d, output_d, total_output_rows , rows_per_instr, cols_per_row, instrs_len);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_poseidon2_skinny_instructions_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_instructions: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    //err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) { fprintf(stderr, "_poseidon2_skinny_instructions: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (instructions_d) cudaFree(instructions_d);
    //if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_poseidon2_skinny_instructions: CUDA error : %s\n", cudaGetErrorString(err));
        // Depending on error handling strategy, propagate this error
    }
}