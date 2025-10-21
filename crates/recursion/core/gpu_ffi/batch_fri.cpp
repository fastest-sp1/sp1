#include <cstdint>
#include <cstdio>

#include "bb31_t.hpp" // For BabyBear type
#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "batch_fri.hpp"

// Assuming CUDA runtime headers are available
#include <cuda_runtime.h>


using namespace sp1_recursion_core_sys;
typedef bb31_t BabyBear;

// Kernel for processing main trace events
__global__ void kernel_process_batch_fri_events_gpu(
                    const BatchFRIEvent<BabyBear>* events_d,   
                    BabyBear* output_d,         
                    uint32_t num_events,               
                    uint32_t num_value_cols          
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_events) {
        uint32_t row_idx = idx ;
        BatchFRIEvent<BabyBear> event = events_d[idx];        
        BabyBear* target_output_ptr = output_d + row_idx * num_value_cols ;

        BatchFRICols<BabyBear>& cols = *reinterpret_cast<BatchFRICols<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::batch_fri::event_to_row<BabyBear>(event, cols);
    }
}


//v1 ok
__global__ void kernel_process_batch_fri_instructions_gpu(
                    const BatchFRIInstrFlat<BabyBear>* instrs_d,     
                    const InstrsFlatIdex* instrs_index_info_d,  
                    const Address<BabyBear>* all_base_p_at_x_d, 
                    const Address<BabyBear>* all_ext_p_at_z_d,     
                    const Address<BabyBear>* all_ext_alpha_pow_d,   
                    BabyBear* output_d,
                    uint32_t num_total_output_rows, 
                    uint32_t num_cols_per_row 
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_total_output_rows) {

        const InstrsFlatIdex& index_info = instrs_index_info_d[idx];

        uint32_t instr_idx = index_info.instr_idx;
        uint32_t internal_row_idx = index_info.arr_idx;

        const BatchFRIInstrFlat<BabyBear>& instr = instrs_d[instr_idx]; 

        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row;
        BatchFRIPreprocessedCols<BabyBear>& cols = 
            *reinterpret_cast<BatchFRIPreprocessedCols<BabyBear>*>(target_output_ptr);

        cols.is_real = BabyBear::one();
        cols.is_end = BabyBear(internal_row_idx == instr.ext_p_at_z_len - 1);
        cols.acc_addr = instr.ext_single_addrs_acc_val.acc; 
        cols.alpha_pow_addr = all_ext_alpha_pow_d[instr.ext_alpha_pow_offset + internal_row_idx];
        cols.p_at_z_addr = all_ext_p_at_z_d[instr.ext_p_at_z_offset + internal_row_idx]; 
        cols.p_at_x_addr = all_base_p_at_x_d[instr.base_p_at_x_offset + internal_row_idx]; 

    }
}

//gpu-> 100%
// Child Kernel: Processes rows for a single instruction
/*
__global__ void kernel_process_batch_fri_instructions_child_gpu(
    const BatchFRIInstrFlat<BabyBear>* parent_instr, 
    const Address<BabyBear>* all_base_p_at_x_d,
    const Address<BabyBear>* all_ext_p_at_z_d,
    const Address<BabyBear>* all_ext_alpha_pow_d,
    BabyBear* output_d,
    uint32_t start_output_idx, 
    uint32_t sub_instrs, 
    uint32_t num_cols_per_row
) {
    uint32_t child_idx = blockIdx.x * blockDim.x + threadIdx.x; //child thread idx

    if (child_idx < sub_instrs) {

        uint32_t global_output_idx = start_output_idx + child_idx;

        BabyBear* target_output_ptr = output_d + global_output_idx * num_cols_per_row;
        BatchFRIPreprocessedCols<BabyBear>& cols = 
            *reinterpret_cast<BatchFRIPreprocessedCols<BabyBear>*>(target_output_ptr);

        cols.is_real = BabyBear::one();
        cols.is_end = BabyBear(child_idx == sub_instrs - 1);
        cols.acc_addr = parent_instr->ext_single_addrs_acc_val.acc; // 从父 Kernel 传递的指令访问

        cols.alpha_pow_addr = all_ext_alpha_pow_d[parent_instr->ext_alpha_pow_offset + child_idx];
        cols.p_at_z_addr = all_ext_p_at_z_d[parent_instr->ext_p_at_z_offset + child_idx];
        cols.p_at_x_addr = all_base_p_at_x_d[parent_instr->base_p_at_x_offset + child_idx];
    }
}

//v2 ok
// Parent Kernel: Launches child kernels for each instruction
__global__ void kernel_process_batch_fri_instructions_parent_gpu(
    const BatchFRIInstrFlat<BabyBear>* instrs_d,     
    const Address<BabyBear>* all_base_p_at_x_d,      
    const Address<BabyBear>* all_ext_p_at_z_d,       
    const Address<BabyBear>* all_ext_alpha_pow_d, 
    BabyBear* output_d,
    uint32_t num_original_instrs,   
    uint32_t num_cols_per_row 
) {
    uint32_t parent_idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (parent_idx < num_original_instrs) {

        const BatchFRIInstrFlat<BabyBear>& current_instr = instrs_d[parent_idx];
        uint32_t sub_instrs = current_instr.ext_p_at_z_len; 
       
        uint32_t start_output_idx = current_instr.ext_p_at_z_offset; // This needs to be calculated or passed as argument

        // Launch child kernel
        uint32_t child_threads_per_block = 256;
        uint32_t child_num_blocks = (sub_instrs + child_threads_per_block - 1) / child_threads_per_block;

        kernel_process_batch_fri_instructions_child_gpu<<<child_num_blocks, child_threads_per_block>>>(
            &current_instr, 
            all_base_p_at_x_d,
            all_ext_p_at_z_d,
            all_ext_alpha_pow_d,
            output_d,
            start_output_idx, 
            sub_instrs,
            num_cols_per_row
        );
    }
}*/

extern "C" void process_batch_fri_events_gpu(
                    const BatchFRIEvent<BabyBear>* events_h, 
                    size_t events_len,  
                    BabyBear* output_d, 
                    size_t output_len,              
                    int num_value_cols                        
) {
    if (events_len == 0) return; // Nothing to process
    BatchFRIEvent<BabyBear> *events_d = nullptr; // Declare at top
    //BabyBear *output_d = nullptr; // Declare at top
    cudaError_t err = cudaSuccess; // Initialize err at top

    // Declare all other variables that might be bypassed by goto
    size_t input_size_bytes;
    size_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (size_t)events_len * sizeof(BatchFRIEvent<BabyBear>); // Assignment here
    err = cudaMalloc(&events_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_events: cudaMalloc events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(events_d, events_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_events: cudaMemcpy events_h to events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    //output_size_bytes = (size_t)events_len * num_value_cols * sizeof(BabyBear); // Assignment here
    //err = cudaMalloc(&output_d, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    // Optional: Initialize device output memory to zero
    //err = cudaMemset(output_d, 0, output_size_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_events: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (events_len + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_batch_fri_events_gpu<<<num_blocks, threads_per_block>>>(
        events_d, output_d, events_len, num_value_cols);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_batch_fri_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    //err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_events: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (events_d) cudaFree(events_d);
    //if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_batch_fri_events: CUDA error : %s\n", cudaGetErrorString(err));
    }
}



//v1 ok
extern "C" void process_batch_fri_instructions_gpu(
    const BatchFRIInstrFlat<BabyBear>* instrs_h,
    size_t instrs_len,
     const InstrsFlatIdex* instrs_index_info_h,
    size_t instrs_index_info_len,
    const Address<BabyBear>* all_base_p_at_x_h,
    size_t all_base_p_at_x_len,
    const Address<BabyBear>* all_ext_p_at_z_h,
    size_t all_ext_p_at_z_len,
    const Address<BabyBear>* all_ext_alpha_pow_h,
    size_t all_ext_alpha_pow_len,
    int original_instrs,
    int total_instrs,
     BabyBear* output_d,
    size_t output_len,
    int num_cols_per_row
) {
    if (instrs_len == 0) return;

    // Declare device pointers
    BatchFRIInstrFlat<BabyBear> *instrs_d = nullptr;
    Address<BabyBear> *all_base_p_at_x_d = nullptr;
    Address<BabyBear> *all_ext_p_at_z_d = nullptr;
    Address<BabyBear> *all_ext_alpha_pow_d = nullptr;
    InstrsFlatIdex *instrs_index_info_d = nullptr; 
    //BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;
    int num_blocks = 0;

    // 1. Allocate device memory for all input arrays
    size_t instrs_bytes = instrs_len * sizeof(BatchFRIInstrFlat<BabyBear>);
    size_t base_p_at_x_bytes = all_base_p_at_x_len * sizeof(Address<BabyBear>);
    size_t ext_p_at_z_bytes = all_ext_p_at_z_len * sizeof(Address<BabyBear>);
    size_t ext_alpha_pow_bytes = all_ext_alpha_pow_len * sizeof(Address<BabyBear>);
    size_t output_bytes = output_len * sizeof(BabyBear);
    size_t instrs_index_info_bytes = instrs_index_info_len * sizeof(InstrsFlatIdex);

    // Malloc all inputs
    err = cudaMalloc(&instrs_d, instrs_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMalloc instrs_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMalloc(&all_base_p_at_x_d, base_p_at_x_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMalloc all_base_p_at_x_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMalloc(&all_ext_p_at_z_d, ext_p_at_z_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMalloc all_ext_p_at_z_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMalloc(&all_ext_alpha_pow_d, ext_alpha_pow_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMalloc all_ext_alpha_pow_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    //err = cudaMalloc(&output_d, output_bytes);
    //if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMalloc(&instrs_index_info_d, instrs_index_info_bytes); 
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMalloc instrs_index_info_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy all input data from host to device
    err = cudaMemcpy(instrs_d, instrs_h, instrs_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMemcpy ffi_instrs_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMemcpy(instrs_index_info_d, instrs_index_info_h, instrs_index_info_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMemcpy instrs_index_info_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(all_base_p_at_x_d, all_base_p_at_x_h, base_p_at_x_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMemcpy all_base_p_at_x_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMemcpy(all_ext_p_at_z_d, all_ext_p_at_z_h, ext_p_at_z_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMemcpy all_ext_p_at_z_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMemcpy(all_ext_alpha_pow_d, all_ext_alpha_pow_h, ext_alpha_pow_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMemcpy all_ext_alpha_pow_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    // 3. Launch kernel
    //int num_total_output_rows = num_original_instrs * rows_per_single_instr;
    num_blocks = (total_instrs + 256 - 1) / 256;

    kernel_process_batch_fri_instructions_gpu<<<num_blocks, 256>>>(
        instrs_d,
        instrs_index_info_d,
        all_base_p_at_x_d,
        all_ext_p_at_z_d,
        all_ext_alpha_pow_d,
        output_d,
        total_instrs,
        num_cols_per_row
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs:  Kernel launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 4. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    //err = cudaMemcpy(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess) { fprintf(stderr, "_batch_fri_instrs: cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    if (instrs_d) cudaFree(instrs_d);
    if (instrs_index_info_d) cudaFree(instrs_index_info_d);
    if (all_base_p_at_x_d) cudaFree(all_base_p_at_x_d);
    if (all_ext_p_at_z_d) cudaFree(all_ext_p_at_z_d);
    if (all_ext_alpha_pow_d) cudaFree(all_ext_alpha_pow_d);
    //if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_batch_fri_instrs: CUDA error : %s\n", cudaGetErrorString(err));
    }
} 
