#include <cstdint>
#include <cstdio>

#include "bb31_t.hpp" // For BabyBear type
#include "sp1-recursion-core-sys-cbindgen.hpp" 

// Assuming CUDA runtime headers are available
#include <cuda_runtime.h>


using namespace sp1_recursion_core_sys;
typedef bb31_t BabyBear;

// Kernel for processing main trace events
__global__ void kernel_process_exp_reverse_bits_events_gpu(
                    const ExpReverseBitsEventFlatFFI<BabyBear>* events_d,
                    const ExpReverseBitsFlatIdex<BabyBear>* exp_index_info_d,
                    const BabyBear* all_events_exp_d,
                    BabyBear* output_d,         
                    uint32_t total_events,               
                    uint32_t num_cols_per_row          
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_events) {
        const ExpReverseBitsFlatIdex<BabyBear>& index_info = exp_index_info_d[idx];

        uint32_t event_idx = index_info.event_idx;
        uint32_t exp_idx = index_info.exp_idx;
        BabyBear multiplier = index_info.multiplier;
        BabyBear prev_accum = index_info.prev_accum;

        const ExpReverseBitsEventFlatFFI<BabyBear>& event = events_d[event_idx]; 

        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row;
        ExpReverseBitsLenCols<BabyBear>& cols = 
            *reinterpret_cast<ExpReverseBitsLenCols<BabyBear>*>(target_output_ptr);

        BabyBear accum = prev_accum * prev_accum * multiplier;

        cols.x = event.base_val;
        cols.current_bit = all_events_exp_d[event.exp_bits_offset + exp_idx];
        cols.multiplier = multiplier;
        cols.accum = accum;
        cols.accum_squared = accum * accum;
        cols.prev_accum_squared = prev_accum * prev_accum;
        cols.prev_accum_squared_times_multiplier = cols.prev_accum_squared * cols.multiplier;
    }
}



__global__ void kernel_process_exp_reverse_bits_instructions_gpu(
                    const ExpReverseBitsInstrFlatFFI<BabyBear>* instrs_d,     
                    const InstrsFlatIdex* exp_index_info_d,  
                    const Address<BabyBear>* all_exp_bits_d,   
                    BabyBear* output_d,
                    uint32_t num_total_output_rows, 
                    uint32_t num_cols_per_row 
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_total_output_rows) {

        const InstrsFlatIdex& index_info = exp_index_info_d[idx];

        uint32_t instr_idx = index_info.instr_idx;
        uint32_t arr_idx = index_info.arr_idx;


        const ExpReverseBitsInstrFlatFFI<BabyBear>& instr = instrs_d[instr_idx]; 

        BabyBear* target_output_ptr = output_d + idx * num_cols_per_row;
        ExpReverseBitsLenPreprocessedCols<BabyBear>& cols = 
            *reinterpret_cast<ExpReverseBitsLenPreprocessedCols<BabyBear>*>(target_output_ptr);

        cols.iteration_num = BabyBear::from_canonical_u32(arr_idx);
        cols.is_first = BabyBear::from_bool(arr_idx == 0);
        cols.is_last = BabyBear::from_bool(arr_idx == instr.exp_len - 1);
        
        cols.is_real = BabyBear::one();
        //cols.x_mem = MemoryAccessCols { addr: addrs.base, mult: -BabyBear::from_bool(i == 0) };
        cols.x_mem.addr = instr.base;
        cols.x_mem.mult = BabyBear::zero() - BabyBear::from_bool(arr_idx == 0);

        cols.exponent_mem.addr = all_exp_bits_d[instr.exp_offset + arr_idx];
        cols.exponent_mem.mult = BabyBear::zero() - BabyBear::one();
                
        cols.result_mem.addr = instr.result;
        cols.result_mem.mult = instr.mult * BabyBear::from_bool(arr_idx == instr.exp_len - 1);

    }
}

//v1 ok
extern "C" void process_exp_reverse_bits_events_gpu(
                        const ExpReverseBitsEventFlatFFI<BabyBear>* events_h,
                        size_t events_len,
                        const ExpReverseBitsFlatIdex<BabyBear>* exp_index_info_h,
                        size_t exp_index_info_len,
                        const BabyBear* all_events_exp_h,
                        size_t all_events_exp_len,
                        BabyBear* output_h,
                        size_t output_len,
                        int total_events,
                        int num_cols_per_row
) {
    if (events_len == 0) return;

    // Declare device pointers
    ExpReverseBitsEventFlatFFI<BabyBear> *events_d = nullptr;
    BabyBear *all_events_exp_d = nullptr;
    ExpReverseBitsFlatIdex<BabyBear> *exp_index_info_d = nullptr; 
    BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;
    int num_blocks = 0;

    // 1. Allocate device memory for all input arrays
    size_t events_bytes = events_len * sizeof(ExpReverseBitsEventFlatFFI<BabyBear>);
    size_t all_events_exp_bytes = all_events_exp_len * sizeof(BabyBear);

    size_t output_bytes = output_len * sizeof(BabyBear);
    size_t exp_index_info_bytes = exp_index_info_len * sizeof(ExpReverseBitsFlatIdex<BabyBear>);

    // Malloc all inputs
    err = cudaMalloc(&events_d, events_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMalloc events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_events_exp_d, all_events_exp_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMalloc all_events_exp_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
   
    err = cudaMalloc(&output_d, output_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&exp_index_info_d, exp_index_info_bytes); 
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMalloc exp_index_info_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy all input data from host to device
    err = cudaMemcpy(events_d, events_h, events_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemcpy ffi_instrs_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMemcpy(exp_index_info_d, exp_index_info_h, exp_index_info_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemcpy exp_index_info_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMemcpy(all_events_exp_d, all_events_exp_h, all_events_exp_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemcpy all_base_p_at_x_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
       
    // 3. Launch kernel
    //int num_total_output_rows = num_original_instrs * rows_per_single_instr;
    num_blocks = (total_events + 256 - 1) / 256;

    kernel_process_exp_reverse_bits_events_gpu<<<num_blocks, 256>>>(
        events_d,
        exp_index_info_d,
        all_events_exp_d,
        output_d,
        total_events,
        num_cols_per_row
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_exp_reverse_bits_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 4. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    if (events_d) cudaFree(events_d);
    if (exp_index_info_d) cudaFree(exp_index_info_d);
    if (all_events_exp_d) cudaFree(all_events_exp_d);
    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_exp_reverse_bits_events: CUDA error : %s\n", cudaGetErrorString(err));
    }
}


extern "C" void process_exp_reverse_bits_instructions_gpu(
                    const ExpReverseBitsInstrFlatFFI<BabyBear>* instrs_h,
                    size_t instrs_len,
                     const InstrsFlatIdex* instrs_index_info_h,
                    size_t instrs_index_info_len,
                    const Address<BabyBear>* all_exp_bits_h,
                    size_t all_exp_bits_len,
                    BabyBear* output_h,
                    size_t output_len,
                    int total_instrs,
                    int num_cols_per_row
) {
    if (instrs_len == 0) return;

    // Declare device pointers
    ExpReverseBitsInstrFlatFFI<BabyBear> *instrs_d = nullptr;
    Address<BabyBear> *all_exp_bits_d = nullptr;
    InstrsFlatIdex *instrs_index_info_d = nullptr; 
    BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;
    int num_blocks = 0;

    // 1. Allocate device memory for all input arrays
    size_t instrs_bytes = instrs_len * sizeof(ExpReverseBitsInstrFlatFFI<BabyBear>);
    size_t exp_bits_bytes = all_exp_bits_len * sizeof(Address<BabyBear>);
    
    size_t output_bytes = output_len * sizeof(BabyBear);
    size_t instrs_index_info_bytes = instrs_index_info_len * sizeof(InstrsFlatIdex);

    // Malloc all inputs
    err = cudaMalloc(&instrs_d, instrs_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs: cudaMalloc instrs_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_exp_bits_d, exp_bits_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs: cudaMalloc all_exp_bits failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMalloc(&output_d, output_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
     err = cudaMalloc(&instrs_index_info_d, instrs_index_info_bytes); 
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs: cudaMalloc instrs_index_info_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 2. Copy all input data from host to device
    err = cudaMemcpy(instrs_d, instrs_h, instrs_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs: cudaMemcpy ffi_instrs_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMemcpy(instrs_index_info_d, instrs_index_info_h, instrs_index_info_bytes, cudaMemcpyHostToDevice);

    err = cudaMemcpy(all_exp_bits_d, all_exp_bits_h, exp_bits_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs: cudaMemcpy all_exp_bits_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
        
    // 3. Launch kernel
    //int num_total_output_rows = num_original_instrs * rows_per_single_instr;
    num_blocks = (total_instrs + 256 - 1) / 256;

    kernel_process_exp_reverse_bits_instructions_gpu<<<num_blocks, 256>>>(
        instrs_d,
        instrs_index_info_d,
        all_exp_bits_d,
        output_d,
        total_instrs,
        num_cols_per_row
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs:  Kernel launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 4. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_instrs: cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    if (instrs_d) cudaFree(instrs_d);
    if (instrs_index_info_d) cudaFree(instrs_index_info_d);
    if (all_exp_bits_d) cudaFree(all_exp_bits_d);

    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_exp_reverse_bits_instrs: CUDA error : %s\n", cudaGetErrorString(err));
    }
}
/*//using child kernel
typedef struct {  
    BabyBear multiplier;
    BabyBear prev_accum;
} ExpReverseBitsInfo;

__global__ void kernel_process_exp_reverse_bits_events_child_gpu(
                    const ExpReverseBitsEventFlatFFI<BabyBear>* events_d,
                    ExpReverseBitsInfo* exp_index_info_d,
                    const BabyBear* all_events_exp_d,
                    BabyBear* output_d,  
                    uint32_t start_output_idx,           
                    uint32_t sub_events,               
                    uint32_t num_cols_per_row          
) {
    uint32_t child_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (child_idx < sub_events) {
        const ExpReverseBitsInfo& index_info = exp_index_info_d[child_idx];

        uint32_t exp_idx = child_idx;
        BabyBear multiplier = index_info.multiplier;
        BabyBear prev_accum = index_info.prev_accum;


        uint32_t global_output_row_idx = start_output_idx + child_idx;

        BabyBear* target_output_ptr = output_d + global_output_row_idx * num_cols_per_row;
        ExpReverseBitsLenCols<BabyBear>& cols = 
            *reinterpret_cast<ExpReverseBitsLenCols<BabyBear>*>(target_output_ptr);

        BabyBear accum = prev_accum * prev_accum * multiplier;

        cols.x = events_d->base_val;
        cols.current_bit = all_events_exp_d[events_d->exp_bits_offset + exp_idx];
        //printf("child_idx:%u, exp_idx:%u, sub_events:%u, offset:%u, current_bit:%u \n", child_idx, exp_idx, sub_events, events_d->exp_bits_offset, cols.current_bit);
        cols.multiplier = multiplier;
        cols.accum = accum;
        cols.accum_squared = accum * accum;
        cols.prev_accum_squared = prev_accum * prev_accum;
        cols.prev_accum_squared_times_multiplier = cols.prev_accum_squared * cols.multiplier;
    }
}

//v2
//GPU 100%
__global__ void kernel_process_exp_reverse_bits_events_parent_gpu(
                    const ExpReverseBitsEventFlatFFI<BabyBear>* events_d,
                    ExpReverseBitsInfo* exp_index_info_d,
                    const BabyBear* all_events_exp_d,
                    BabyBear* output_d,  
                    uint32_t total_events,               
                    uint32_t num_cols_per_row          
) {
    uint32_t parent_idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (parent_idx < total_events) {

        const ExpReverseBitsEventFlatFFI<BabyBear>& current_event = events_d[parent_idx];
        uint32_t event_exp_len = current_event.exp_len; 
       
        uint32_t start_output_idx = current_event.exp_bits_offset;

        // Launch child kernel
        uint32_t child_threads_per_block = 256;
        uint32_t child_num_blocks = (event_exp_len + child_threads_per_block - 1) / child_threads_per_block;

        //
        

    
        BabyBear accum = 1;
        for (uint32_t j = 0; j < event_exp_len; j++) {
            BabyBear current_exp_bit = all_events_exp_d[current_event.exp_bits_offset + j];//current_event.exp[j];
            BabyBear multiplier = (current_exp_bit == 1) ? current_event.base_val : 1;
            BabyBear prev_accum = accum;
            //printf("parent_idx:%u , j=%u , current_exp_bit=%u , event_exp_len=%u, exp_bits_offset:%u\n", parent_idx, j, current_exp_bit, event_exp_len, current_event.exp_bits_offset);
            ExpReverseBitsInfo& device_events_idx = exp_index_info_d[current_event.exp_bits_offset + j];

            device_events_idx.multiplier = multiplier;
            device_events_idx.prev_accum = prev_accum;
           
            accum = prev_accum * prev_accum * multiplier;
        }
       //
        kernel_process_exp_reverse_bits_events_child_gpu<<<child_num_blocks, child_threads_per_block>>>(
            &current_event, 
            //device_events_idx,
            &exp_index_info_d[current_event.exp_bits_offset],
            all_events_exp_d,
            output_d,
            start_output_idx,
            event_exp_len,
            num_cols_per_row
        );  
    }
}

//v2 GPU-->100%
extern "C" void process_exp_reverse_bits_events_gpu_v2(
                        const ExpReverseBitsEventFlatFFI<BabyBear>* events_h,
                        size_t events_len,
                        // const ExpReverseBitsFlatIdex<BabyBear>* exp_index_info_h,
                        //size_t exp_index_info_len,
                        const BabyBear* all_events_exp_h,
                        size_t all_events_exp_len,
                        BabyBear* output_h,
                        size_t output_len,
                        int total_events,
                        int num_cols_per_row
) {
    if (events_len == 0) return;

    // Declare device pointers
    ExpReverseBitsEventFlatFFI<BabyBear> *events_d = nullptr;
    BabyBear *all_events_exp_d = nullptr;
    ExpReverseBitsInfo *exp_info_d = nullptr; 
    BabyBear *output_d = nullptr;
    cudaError_t err = cudaSuccess;
    int num_blocks = 0;

    // 1. Allocate device memory for all input arrays
    size_t events_bytes = events_len * sizeof(ExpReverseBitsEventFlatFFI<BabyBear>);
    size_t all_events_exp_bytes = all_events_exp_len * sizeof(BabyBear);

    size_t output_bytes = output_len * sizeof(BabyBear);
    size_t exp_index_info_bytes = total_events * sizeof(ExpReverseBitsInfo);

    // Malloc all inputs
    err = cudaMalloc(&events_d, events_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMalloc events_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&all_events_exp_d, all_events_exp_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMalloc all_events_exp_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
   
    err = cudaMalloc(&output_d, output_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMalloc output_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    err = cudaMalloc(&exp_info_d, exp_index_info_bytes); 
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMalloc exp_info_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemset(exp_info_d, 0, exp_index_info_bytes); 
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemset exp_info_d failed: %s\n", cudaGetErrorString(err)); goto cleanup; }


    // 2. Copy all input data from host to device
    err = cudaMemcpy(events_d, events_h, events_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemcpy ffi_instrs_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    //err = cudaMemcpy(exp_index_info_d, exp_index_info_h, exp_index_info_bytes, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemcpy exp_index_info_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
    
    err = cudaMemcpy(all_events_exp_d, all_events_exp_h, all_events_exp_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemcpy all_base_p_at_x_h failed: %s\n", cudaGetErrorString(err)); goto cleanup; }
       
    // 3. Launch kernel
    //int num_total_output_rows = num_original_instrs * rows_per_single_instr;
    num_blocks = (events_len + 256 - 1) / 256;

    kernel_process_exp_reverse_bits_events_parent_gpu<<<num_blocks, 256>>>(
        events_d,
        exp_info_d,
        all_events_exp_d,
        output_d,
        events_len,
        num_cols_per_row
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "kernel_process_exp_reverse_bits_events_gpu launch failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    // 4. Synchronize and copy output from device to host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

    err = cudaMemcpy(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "_exp_reverse_bits_events: cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err)); goto cleanup; }

cleanup:
    if (events_d) cudaFree(events_d);
    if (exp_info_d) cudaFree(exp_info_d);
    if (all_events_exp_d) cudaFree(all_events_exp_d);
    if (output_d) cudaFree(output_d);
    if (err != cudaSuccess) {
        fprintf(stderr, "_exp_reverse_bits_events: CUDA error : %s\n", cudaGetErrorString(err));
    }
}

*/

