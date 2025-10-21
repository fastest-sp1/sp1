#pragma once 
#include "bb31_t.hpp" 
#include "gpu_types.hpp"

//template <typename F>
//__global__ void bit_reverse_rows_kernel(F* data, int h, int w, int log_h);
extern "C" int bit_reverse_rows_gpu(const GpuMatrix<bb31_t>* d_input_ptr, GpuMatrix<bb31_t>* d_out_ptr);