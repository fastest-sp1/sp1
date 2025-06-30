#include <cstdint>
#include <cstdio>

#include "bb31_t.hpp" // For BabyBear type
#include "sp1-recursion-core-sys-cbindgen.hpp" 
#include "fri_fold.hpp"

// Assuming CUDA runtime headers are available
#include <cuda_runtime.h>


using namespace sp1_recursion_core_sys;
typedef bb31_t BabyBear;

// Kernel for processing main trace events
__global__ void kernel_process_fri_fold_events_gpu(
                    const FriFoldEvent<BabyBear>* events_d,   
                    BabyBear* output_d,         
                    uint32_t num_events,               
                    uint32_t num_value_cols          
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_events) {
        FriFoldEvent<BabyBear> event = events_d[idx];        
        BabyBear* target_output_ptr = output_d + idx * num_value_cols ;

        FriFoldCols<BabyBear>& cols = *reinterpret_cast<FriFoldCols<BabyBear>*>(target_output_ptr);

        sp1_recursion_core_sys::fri_fold::event_to_row<BabyBear>(event, cols);
    }
}

__global__ void kernel_process_fri_fold_instructions_gpu(
                    const FriFoldInstrFlat<BabyBear>* instrs_d,
                    const InstrsFlatIdex* instrs_index_info_d,
                    const Address<BabyBear>* all_ext_mat_opening_d,
                    const Address<BabyBear>* all_ext_ps_at_z_d,
                    const Address<BabyBear>* all_alpha_pow_input_d,
                    const Address<BabyBear>* all_ro_input_d,
                    const Address<BabyBear>* all_alpha_pow_output_d,
                    const Address<BabyBear>* all_ro_output_d,
                    const BabyBear* all_alpha_pow_mults_d,
                    const BabyBear* all_ro_mults_d,
                    BabyBear* output_d,
                    uint32_t num_total_output_rows,
                    uint32_t num_cols_per_row
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_total_output_rows) {

        const InstrsFlatIdex& index_info = instrs_index_info_d[idx];

        uint32_t instr_idx = index_info.instr_idx;
        uint32_t child_idx = index_info.arr_idx;

        const FriFoldInstrFlat<BabyBear>& instr = instrs_d[instr_idx]; 

        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row;
        FriFoldPreprocessedCols<BabyBear>& cols = 
            *reinterpret_cast<FriFoldPreprocessedCols<BabyBear>*>(target_output_ptr);

        cols.is_real = BabyBear::one();
        cols.is_first = BabyBear::from_bool(child_idx == 0);

        cols.z_mem.addr = instr.ext_single_addrs.z;
        cols.z_mem.mult = BabyBear::zero() - BabyBear::from_bool(child_idx == 0);

        cols.x_mem.addr = instr.base_single_addrs.x;
        cols.x_mem.mult = BabyBear::zero() - BabyBear::from_bool(child_idx == 0);

        cols.alpha_mem.addr = instr.ext_single_addrs.alpha;
        cols.alpha_mem.mult = BabyBear::zero() - BabyBear::from_bool(child_idx == 0);

        cols.alpha_pow_input_mem.addr = all_alpha_pow_input_d[instr.ext_vec_addrs_alpha_pow_input_offset + child_idx];//instr.ext_vec_addrs_alpha_pow_input_ptr[i];
        cols.alpha_pow_input_mem.mult = BabyBear::zero() - BabyBear::one();

        cols.ro_input_mem.addr = all_ro_input_d[instr.ext_vec_addrs_ro_input_offset + child_idx];//instr.ext_vec_addrs_ro_input_ptr[i];
        cols.ro_input_mem.mult = BabyBear::zero() - BabyBear::one();

        cols.p_at_z_mem.addr = all_ext_ps_at_z_d[instr.ext_vec_addrs_ps_at_z_offset + child_idx];//instr.ext_vec_addrs_ps_at_z_ptr[i];
        cols.p_at_z_mem.mult = BabyBear::zero() - BabyBear::one();

        cols.p_at_x_mem.addr = all_ext_mat_opening_d[instr.ext_vec_addrs_mat_opening_offset + child_idx];//instr.ext_vec_addrs_mat_opening_ptr[i];
        cols.p_at_x_mem.mult = BabyBear::zero() - BabyBear::one();

        cols.alpha_pow_output_mem.addr = all_alpha_pow_output_d[instr.ext_vec_addrs_alpha_pow_output_offset + child_idx];//instr.ext_vec_addrs_alpha_pow_output_ptr[i];
        cols.alpha_pow_output_mem.mult = all_alpha_pow_mults_d[instr.alpha_pow_mults_offset + child_idx];//instr.alpha_pow_mults_ptr[i];

        cols.ro_output_mem.addr = all_ro_output_d[instr.ext_vec_addrs_ro_output_offset + child_idx];//instr.ext_vec_addrs_ro_output_ptr[i];
        cols.ro_output_mem.mult = all_ro_mults_d[instr.ro_mults_offset + child_idx];//instr.ro_mults_ptr[i];
    }
}

extern "C" void process_fri_fold_events_gpu(
                    const FriFoldEvent<BabyBear>* events_h, 
                    uint32_t events_len,  
                    BabyBear* output_h, 
                    uint32_t output_len,              
                    int num_value_cols                        
) {
    if (events_len == 0) return;
    FriFoldEvent<BabyBear> *events_d = nullptr; // Declare at top
    BabyBear *output_d = nullptr; // Declare at top
    cudaError_t err = cudaSuccess; // Initialize err at top

    // Declare all other variables that might be bypassed by goto
    uint32_t input_size_bytes;
    uint32_t output_size_bytes;
    const int threads_per_block = 256;
    int num_blocks;

    // 1. Allocate device memory for input
    input_size_bytes = (uint32_t)events_len * sizeof(FriFoldEvent<BabyBear>); // Assignment here
    err = cudaMalloc(&events_d, input_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_events: cudaMalloc events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy input from host to device
    err = cudaMemcpy(events_d, events_h, input_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_events: cudaMemcpy events_h to events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 3. Allocate device memory for output
    output_size_bytes = (uint32_t)events_len * num_value_cols * sizeof(BabyBear); // Assignment here
    err = cudaMalloc(&output_d, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    // Optional: Initialize device output memory to zero
    err = cudaMemset(output_d, 0, output_size_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_events: cudaMemset output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // Define launch configuration
    num_blocks = (events_len + threads_per_block - 1) / threads_per_block; // Assignment here

    // 4. Launch the kernel
    kernel_process_fri_fold_events_gpu<<<num_blocks, threads_per_block>>>(
        events_d, output_d, events_len, num_value_cols);
    err = cudaGetLastError(); // Check for kernel launch errors
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_fri_fold_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 5. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_events: cudaMemcpy output_d to output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    // 6. Free device memory
    if (events_d) cudaFree(events_d);
    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_fri_fold_events: CUDA error: %s\n", cudaGetErrorString(err));
    }
}

//v1 ok
extern "C" void process_fri_fold_instructions_gpu(
    const FriFoldInstrFlat<BabyBear>* instrs_h,
    uint32_t instrs_len,
    const InstrsFlatIdex* instrs_index_info_h, 
    uint32_t instrs_index_info_len,                      
    const Address<BabyBear>* all_ext_mat_opening_h,
    uint32_t all_ext_mat_opening_len,
    const Address<BabyBear>* all_ext_ps_at_z_h,
    uint32_t all_ext_ps_at_z_len,
    const Address<BabyBear>* all_alpha_pow_input_h,
    uint32_t all_alpha_pow_input_len,
    const Address<BabyBear>* all_ro_input_h,
    uint32_t all_ro_input_len,
    const Address<BabyBear>* all_alpha_pow_output_h,
    uint32_t all_alpha_pow_output_len,
    const Address<BabyBear>* all_ro_output_h,
    uint32_t all_ro_output_len,
    const BabyBear* all_alpha_pow_mults_h,
    uint32_t all_alpha_pow_mults_len,
    const BabyBear* all_ro_mults_h,
    uint32_t all_ro_mults_len,
    BabyBear* output_h,
    uint32_t output_len,
    uint32_t num_total_output_rows,
    uint32_t num_cols_per_row
) {
    if (num_total_output_rows == 0) return;

    FriFoldInstrFlat<BabyBear> *instrs_d = nullptr;
    InstrsFlatIdex *instrs_index_info_d = nullptr; 
    Address<BabyBear> *all_ext_mat_opening_d = nullptr;
    Address<BabyBear> *all_ext_ps_at_z_d = nullptr;
    Address<BabyBear> *all_alpha_pow_input_d = nullptr;
    Address<BabyBear> *all_ro_input_d = nullptr;
    Address<BabyBear> *all_alpha_pow_output_d = nullptr;
    Address<BabyBear> *all_ro_output_d = nullptr;
    BabyBear *all_alpha_pow_mults_d = nullptr;
    BabyBear *all_ro_mults_d = nullptr;

    BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;
    int num_blocks = 0;

    uint32_t instrs_bytes = instrs_len * sizeof(FriFoldInstrFlat<BabyBear>);
    uint32_t instrs_index_info_bytes = instrs_index_info_len * sizeof(InstrsFlatIdex); 
    uint32_t ext_mat_opening_bytes = all_ext_mat_opening_len * sizeof(Address<BabyBear>);
    uint32_t ext_ps_at_z_bytes = all_ext_ps_at_z_len * sizeof(Address<BabyBear>);
    uint32_t alpha_pow_input_bytes = all_alpha_pow_input_len * sizeof(Address<BabyBear>);
    uint32_t ro_input_bytes = all_ro_input_len * sizeof(Address<BabyBear>);
    uint32_t alpha_pow_output_bytes = all_alpha_pow_output_len * sizeof(Address<BabyBear>);
    uint32_t ro_output_bytes = all_ro_output_len * sizeof(Address<BabyBear>);
    uint32_t output_bytes = output_len * sizeof(BabyBear);
    uint32_t alpha_pow_mults_bytes = all_alpha_pow_mults_len * sizeof(BabyBear);
    uint32_t ro_mults_bytes = all_ro_mults_len * sizeof(BabyBear);

    err = cudaMalloc(&instrs_d, instrs_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc instrs_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&instrs_index_info_d, instrs_index_info_bytes); // <--- 新增
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc instrs_index_info_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_ext_mat_opening_d, ext_mat_opening_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc all_ext_mat_opening_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_ext_ps_at_z_d, ext_ps_at_z_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc all_ext_ps_at_z_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_alpha_pow_input_d, alpha_pow_input_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc all_alpha_pow_input_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_ro_input_d, ro_input_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc all_ro_input_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_alpha_pow_output_d, alpha_pow_output_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc all_alpha_pow_output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_ro_output_d, ro_output_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc all_ro_output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMalloc(&all_alpha_pow_mults_d, alpha_pow_mults_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc all_alpha_pow_mults_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_ro_mults_d, ro_mults_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc all_ro_mults_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMalloc(&output_d, output_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(instrs_d, instrs_h, instrs_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy instrs_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMemcpy(instrs_index_info_d, instrs_index_info_h, instrs_index_info_bytes, cudaMemcpyHostToDevice); // <--- 新增
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy instrs_index_info_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMemcpy(all_ext_mat_opening_d, all_ext_mat_opening_h, ext_mat_opening_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy all_ext_mat_opening_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMemcpy(all_ext_ps_at_z_d, all_ext_ps_at_z_h, ext_ps_at_z_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy all_ext_ps_at_z_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMemcpy(all_alpha_pow_input_d, all_alpha_pow_input_h, alpha_pow_input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy all_alpha_pow_input_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMemcpy(all_ro_input_d, all_ro_input_h, ro_input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy all_ro_input_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMemcpy(all_alpha_pow_output_d, all_alpha_pow_output_h, alpha_pow_output_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy all_alpha_pow_output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMemcpy(all_ro_output_d, all_ro_output_h, ro_output_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy all_ro_output_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
     
    err = cudaMemcpy(all_alpha_pow_mults_d, all_alpha_pow_mults_h, alpha_pow_mults_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy all_alpha_pow_mults_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMemcpy(all_ro_mults_d, all_ro_mults_h, ro_mults_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy all_ro_mults_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    num_blocks = (num_total_output_rows + 256 - 1) / 256;

    kernel_process_fri_fold_instructions_gpu<<<num_blocks, 256>>>(
        instrs_d,
        instrs_index_info_d,
        all_ext_mat_opening_d,
        all_ext_ps_at_z_d,
        all_alpha_pow_input_d,
        all_ro_input_d,
        all_alpha_pow_output_d,
        all_ro_output_d,
        all_alpha_pow_mults_d,
        all_ro_mults_d,
        output_d,
        num_total_output_rows,
        num_cols_per_row
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: Kernel launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 4. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_fri_fold_instrs: cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    if (instrs_d) cudaFree(instrs_d);
    if (instrs_index_info_d) cudaFree(instrs_index_info_d);
    if (all_ext_ps_at_z_d) cudaFree(all_ext_ps_at_z_d);
    if (all_ext_mat_opening_d) cudaFree(all_ext_mat_opening_d);
    if (all_alpha_pow_input_d) cudaFree(all_alpha_pow_input_d);
    if (all_ro_input_d) cudaFree(all_ro_input_d);
    if (all_alpha_pow_output_d) cudaFree(all_alpha_pow_output_d);
    if (all_ro_output_d) cudaFree(all_ro_output_d);
    if (all_alpha_pow_mults_d) cudaFree(all_alpha_pow_mults_d);
    if (all_ro_mults_d) cudaFree(all_ro_mults_d);
    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_fri_fold_instrs: CUDA error : %s\n", cudaGetErrorString(err));
    }
} 
