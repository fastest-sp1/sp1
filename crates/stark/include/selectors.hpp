#pragma once
#include "gpu_types.hpp"

int generate_selectors_on_device(
    int trace_log_size, 
    Val trace_subgroup_generator,
    int coset_log_size, 
    Val coset_shift, 
    Val coset_subgroup_generator,
    Val* d_is_first_row, 
    Val* d_is_last_row,
    Val* d_is_transition, 
    Val* d_inv_vanishing
);