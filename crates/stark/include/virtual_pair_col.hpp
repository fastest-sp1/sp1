#pragma once
#include "gpu_types.hpp" // For Val, PairCol, etc.
//#include "sp1-recursion-core-sys-cbindgen.hpp" // For cbindgen structs if needed

// Define a maximum number of terms (column*weight) a single VirtualPairCol can represent.
// `is_real` for ALU has 4 terms. Let's set a safe upper bound like 8.
constexpr int MAX_VPC_TERMS = 8;
class VirtualPairCol {
  public:
    // USE OUR OWN PAIR STRUCT
    Pair<PairCol, Val> column_weights[MAX_VPC_TERMS];
    int num_weights;
    Val constant;

    // --- Constructors ---
    __host__ __device__ VirtualPairCol() : num_weights(0), constant(Val::zero()) {}
    
    // --- Static Methods ---
    __host__ __device__ static VirtualPairCol from_constant(Val val) {
        VirtualPairCol vpc;
        vpc.constant = val;
        vpc.num_weights = 0;
        return vpc;
    }
    
    __host__ __device__ static VirtualPairCol single(PairCol col) {
        VirtualPairCol vpc;
        // Construct the Pair directly instead of assigning
        vpc.column_weights[0] = Pair<PairCol, Val>({col.type, col.index}, Val::one());
        vpc.num_weights = 1;
        vpc.constant = Val::zero();
        return vpc;
    }

    __host__ __device__ static VirtualPairCol single_main(int index) {
        return single({PairColType::Main, index});
    }

    __host__ __device__ static VirtualPairCol single_preprocessed(int index) {
        return single({PairColType::Preprocessed, index});
    }

    // --- `apply` method (no change needed) ---
    __device__ Val apply(const Val* main_row, const Val* prep_row) const {
        Val result = this->constant;
        for (int i = 0; i < this->num_weights; ++i) {
            const auto& pair = this->column_weights[i];
            Val col_val = pair.first.get(prep_row, main_row);
            result += col_val * pair.second;
        }
        return result;
    }
    
    // --- `operator+` ---
    __host__ __device__ VirtualPairCol operator+(const VirtualPairCol& other) const {
        VirtualPairCol result;
        result.constant = this->constant + other.constant;
        
        int current_idx = 0;
        for (int i = 0; i < this->num_weights; ++i) {
            result.column_weights[current_idx++] = this->column_weights[i];
        }
        for (int i = 0; i < other.num_weights; ++i) {
            result.column_weights[current_idx++] = other.column_weights[i];
        }
        result.num_weights = current_idx;
        
        return result;
    }
};


// Maximum number of values any single interaction can have.
// For `receive_single`, this is `[addr, val, 0, 0, 0]`, so 5. Let's use a safe upper bound.
constexpr int MAX_INTERACTION_VALUES = 8;

//This struct is intended to be created and live entirely on the GPU stack inside the kernel.
// We cannot use std::vector here. We must use a fixed-size array. 
struct Interaction {
    // Use a fixed-size array instead of std::vector
    VirtualPairCol values[MAX_INTERACTION_VALUES];
    int num_values; // Keep track of how many are actually used

    VirtualPairCol multiplicity;
    InteractionKind kind;
    InteractionScope scope;
    bool is_send;
};
