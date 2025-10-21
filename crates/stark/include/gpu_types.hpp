// file: gpu_types.hpp
#pragma once
#include "bb31_t.hpp"
#include "bb31_quartic_extension_t.hpp"
#include "bb31_septic_extension_t.hpp" 
#include <vector>
//#include "sp1-recursion-core-sys-cbindgen.hpp" 

using VarEF = bb31_quartic_extension_t;
// Base field types
using Val = bb31_t;
using Expr = bb31_t; // In eval context, expressions are values
using SepticDigest_bb31 = bb31_septic_digest_t;

// Challenge/Extension field types
using Challenge = bb31_quartic_extension_t;
using ExprEF = bb31_quartic_extension_t;

// Vectorized/Packed types for GPU
// We will use arrays to represent packed values. Let's define a constant.
// This should match p3_field::PackedValue::WIDTH
constexpr int PACKED_WIDTH = 16; 

using PackedVal = Val[PACKED_WIDTH];
using PackedChallenge = Challenge[PACKED_WIDTH];

template <int Width> struct Packed { Val data[Width]; };
template <int Width> struct PackedC { Challenge data[Width]; };

using PackedVal1 = Packed<1>;
using PackedVal8 = Packed<8>;
using PackedVal16 = Packed<16>;

using PackedChallenge1 = PackedC<1>;
using PackedChallenge8 = PackedC<8>;
using PackedChallenge16 = PackedC<16>;

template <typename T, int Width>
struct PackedType; // General template is left undefined.

// --- Specializations for Val ---
template <>
struct PackedType<Val, 1> { using type = PackedVal1; };
template <>
struct PackedType<Val, 8> { using type = PackedVal8; };
template <>
struct PackedType<Val, 16> { using type = PackedVal16; };

// --- Specializations for Challenge ---
template <>
struct PackedType<Challenge, 1> { using type = PackedChallenge1; };
template <>
struct PackedType<Challenge, 8> { using type = PackedChallenge8; };
template <>
struct PackedType<Challenge, 16> { using type = PackedChallenge16; };



constexpr const int NUM_BASE_ALU_COLS =  12;
constexpr const int NUM_BASE_ALU_PREPROCESSED_COLS =  32;
constexpr const int BASE_ALU_INTERACTIONS_PER_ROW = 12;
constexpr const int NUM_BASE_ALU_ENTRIES_PER_ROW = 4;
constexpr const int NUM_EXT_ALU_ENTRIES_PER_ROW = 4;
constexpr const int NUM_BATCH_FRI_COLS = 13;
constexpr const int NUM_BATCH_FRI_PREPROCESSED_COLS = 6;
constexpr const int NUM_EXP_REVERSE_BITS_LEN_COLS = 7;
constexpr const int NUM_EXP_REVERSE_BITS_LEN_PREPROCESSED_COLS = 10;
constexpr const int NUM_FRI_FOLD_COLS = 33;
constexpr const int NUM_FRI_FOLD_PREPROCESSED_COLS = 20;
constexpr const int DIGEST_SIZE  = 8;
constexpr const int NUM_PUBLIC_VALUES_COLS = 1;
constexpr const int NUM_PUBLIC_VALUES_PREPROCESSED_COLS = 10;
constexpr const int SELECT_COLS = 5;
constexpr const int SELECT_PREPROCESSED_COLS = 8;

constexpr const int NUM_EXTERNAL_ROUNDS = 8;
constexpr const int NUM_INTERNAL_ROUNDS = 13;
constexpr const int POSEIDON2_STATE_WIDTH = 16;
constexpr const int P2_GHOST = NUM_INTERNAL_ROUNDS - 1;
//constexpr const int PREPROCESSED_POSEIDON2_WIDTH = 49;//?
//constexpr const int NUM_POSEIDON2_DEGREE3_COLS =  size_of(Poseidon2Degree3Cols<u8>);
//the structure should be defined in p1-recursion-core-sys-cbindgen.hpp. It is for test here.
//base alu chip
template<typename V>
struct BaseAluIo {
  V out;
  V in1;
  V in2;
};

template<typename F>
struct BaseAluValueCols {
  BaseAluIo<F> vals;
};

template<typename F>
using Address = F;

template<typename F>
struct BaseAluAccessCols {
  BaseAluIo<Address<F>> addrs;
  F is_add;
  F is_sub;
  F is_mul;
  F is_div;
  F mult;
};

template<typename F>
struct BaseAluCols {
    BaseAluValueCols<F> values[NUM_BASE_ALU_ENTRIES_PER_ROW];//=4
};

template<typename F>
struct BaseAluPreprocessedCols {
    BaseAluAccessCols<F> accesses[NUM_BASE_ALU_ENTRIES_PER_ROW];
};

//ext aul chip
template<typename V>
struct ExtAluIo {
  V out;
  V in1;
  V in2;
};

template<typename F>
struct ExtAluValueCols {
  ExtAluIo<Block<F>> vals;
};

template<typename F>
struct ExtAluAccessCols {
  ExtAluIo<Address<F>> addrs;
  F is_add;
  F is_sub;
  F is_mul;
  F is_div;
  F mult;
};

template<typename F>
struct ExtAluCols {
    ExtAluValueCols<F> values[NUM_EXT_ALU_ENTRIES_PER_ROW];
};

template<typename F>
struct ExtAluPreprocessedCols {
    ExtAluAccessCols<F> accesses[ NUM_EXT_ALU_ENTRIES_PER_ROW];
};
///end ext_alu chip

//batch_fri chip
template<typename F>
struct BatchFRIPreprocessedCols {
  F is_real;
  F is_end;
  Address<F> acc_addr;
  Address<F> alpha_pow_addr;
  Address<F> p_at_z_addr;
  Address<F> p_at_x_addr;
};

template<typename T>
struct BatchFRICols {
  Block<T> acc;
  Block<T> alpha_pow;
  Block<T> p_at_z;
  T p_at_x;
};

//end batch_fri

//exp_reverse_bits chip
template<typename T>
struct ExpReverseBitsLenCols {
  /// The base of the exponentiation.
  T x;
  /// The current bit of the exponent. This is read from memory.
  T current_bit;
  /// The previous accumulator squared.
  T prev_accum_squared;
  /// Is set to the value local.prev_accum_squared * local.multiplier.
  T prev_accum_squared_times_multiplier;
  /// The accumulator of the current iteration.
  T accum;
  /// The accumulator squared.
  T accum_squared;
  /// A column which equals x if `current_bit` is on, and 1 otherwise.
  T multiplier;
};

template<typename F>
struct MemoryAccessCols {
  /// The address to access.
  Address<F> addr;
  /// The multiplicity which to read/write.
  /// "Positive" values indicate a write, and "negative" values indicate a read.
  F mult;
};

template<typename T>
struct ExpReverseBitsLenPreprocessedCols {
  MemoryAccessCols<T> x_mem;
  MemoryAccessCols<T> exponent_mem;
  MemoryAccessCols<T> result_mem;
  T iteration_num;
  T is_first;
  T is_last;
  T is_real;
};
//end exp_reverse chip

//memory const
#define NUM_CONST_MEM_ENTRIES_PER_ROW 2

// A simple pair struct to hold the (value, access) tuple
template <typename T>
struct ValueAccessPair {
    Block<T> value;
    MemoryAccessCols<T> access;
};

// Corresponds to Rust's MemoryPreprocessedCols<T>
template <typename T>
struct MemoryPreprocessedCols {
    ValueAccessPair<T> values_and_accesses[NUM_CONST_MEM_ENTRIES_PER_ROW];
};

// Corresponds to Rust's MemoryCols<T>
template <typename T>
struct MemoryCols {
    T _nothing;
};

//end memory const

//memory var
#define NUM_VAR_MEM_ENTRIES_PER_ROW 2
template <typename T>
struct MemoryVarPreprocessedCols {
    MemoryAccessCols<T> accesses[NUM_VAR_MEM_ENTRIES_PER_ROW];
};

// Corresponds to Rust's MemoryCols<T> for MemoryVar
template <typename T>
struct MemoryVarCols {
    Block<T> values[NUM_VAR_MEM_ENTRIES_PER_ROW];
};

//end memory var

//fri_fold chip
template <typename T>
struct FriFoldPreprocessedCols {
    T is_first;

    MemoryAccessCols<T> z_mem;
    MemoryAccessCols<T> alpha_mem;
    MemoryAccessCols<T> x_mem;

    MemoryAccessCols<T> alpha_pow_input_mem;
    MemoryAccessCols<T> ro_input_mem;
    MemoryAccessCols<T> p_at_x_mem; // p_at_x is mat_opening
    MemoryAccessCols<T> p_at_z_mem; // p_at_z is ps_at_z

    MemoryAccessCols<T> ro_output_mem;
    MemoryAccessCols<T> alpha_pow_output_mem;

    T is_real;
};

template <typename T>
struct FriFoldCols {
    Block<T> z;
    Block<T> alpha;
    T x;

    Block<T> p_at_x;
    Block<T> p_at_z;
    Block<T> alpha_pow_input;
    Block<T> ro_input;

    Block<T> alpha_pow_output;
    Block<T> ro_output;
};
//end

//public values
template <typename T>
struct PublicValuesPreprocessedCols {
    T pv_idx[DIGEST_SIZE];
    MemoryAccessCols<T> pv_mem;
};

template <typename T>
struct PublicValuesCols {
    T pv_element;
};
//end

//select chip
template <typename T>
struct SelectIo {
    T bit;
    T out1;
    T out2;
    T in1;
    T in2;
    
};

template <typename T>
struct SelectPreprocessedCols {
    T is_real;
    SelectIo<Address<T>> addrs;
    T mult1;
    T mult2;
};

template <typename T>
struct SelectCols {
    SelectIo<T> vals;
};
//end

//poseidon2 wide chip
template <typename T>
struct Poseidon2SBoxCols {
    T external_rounds_sbox_state[NUM_EXTERNAL_ROUNDS][POSEIDON2_STATE_WIDTH];
    T internal_rounds_sbox_state[NUM_INTERNAL_ROUNDS];
};

template <typename T>
struct Poseidon2StateCols {
    T external_rounds_state[NUM_EXTERNAL_ROUNDS][POSEIDON2_STATE_WIDTH];
    T internal_rounds_state[POSEIDON2_STATE_WIDTH];
    T internal_rounds_s0[P2_GHOST];
    T output_state[POSEIDON2_STATE_WIDTH];
};


template <typename T>
struct Poseidon2Degree3Cols {
    Poseidon2StateCols<T> state;
    Poseidon2SBoxCols<T> sbox_state;
};

template<typename T>
struct Poseidon2Io {
  T input[POSEIDON2_STATE_WIDTH];
  T output[POSEIDON2_STATE_WIDTH];
};

template<typename F>
struct Poseidon2SkinnyInstr {
  Poseidon2Io<Address<F>> addrs;
  F mults[POSEIDON2_STATE_WIDTH];
};

template<typename T>
struct Poseidon2PreprocessedColsWide {
  Address<T> input[POSEIDON2_STATE_WIDTH];
  MemoryAccessCols<T> output[POSEIDON2_STATE_WIDTH];
  T is_real_neg;
};

constexpr const int NUM_POSEIDON2_DEGREE3_COLS =  sizeof(Poseidon2Degree3Cols<char>);

constexpr const int SBOX_BASE_OFFSET =  sizeof(Poseidon2StateCols<char>);
constexpr const int PREPROCESSED_POSEIDON2_WIDTH = sizeof(Poseidon2PreprocessedColsWide<char>);

//end

//poseidon2 skinny chip
constexpr const int NUM_INTERNAL_ROUNDS_S0 = NUM_INTERNAL_ROUNDS - 1;
template<typename T>
struct RoundCountersPreprocessedCols {
  T is_input_round;
  T is_external_round;
  T is_internal_round;
  T round_constants[POSEIDON2_STATE_WIDTH];
};

template<typename T>
struct Poseidon2PreprocessedColsSkinny {
  MemoryAccessCols<T> memory_preprocessed[POSEIDON2_STATE_WIDTH];
  RoundCountersPreprocessedCols<T> round_counters_preprocessed;
};

template <typename T>
struct Poseidon2SkinnyCols {
    T state_var[POSEIDON2_STATE_WIDTH];
    T internal_rounds_s0[NUM_INTERNAL_ROUNDS_S0];
};
//end

enum class InteractionKind {
   /// Interaction with the memory table, such as read and write.
    Memory = 1,

    /// Interaction with the program table, loading an instruction at a given pc address.
    Program = 2,

    /// Interaction with instruction oracle.
    Instruction = 3,

    /// Interaction with the ALU operations.
    Alu = 4,

    /// Interaction with the byte lookup table for byte operations.
    Byte = 5,

    /// Requesting a range check for a given value and range.
    Range = 6,

    /// Interaction with the field op table for field operations.
    Field = 7,

    /// Interaction with a syscall.
    Syscall = 8,

    /// Interaction with the global table.
    Global = 9,
    // Add other kinds as needed
};

typedef enum InteractionScope {
  Global = 0,
  Local,
} InteractionScope;

// A simplified C++ representation of sp1_core::air::AirInteraction
// In the GPU context, we just need the values and the multiplicity.
template <typename T>
struct AirInteraction {
    T values[32]; // Use a fixed-size array, large enough for any interaction
    int num_values;
    T multiplicity;
    InteractionKind kind;
    // InteractionScope scope; // Scope might influence which interaction columns are used.
};


enum class PairColType {
        Preprocessed,
        Main
};


struct PairCol {
    PairColType type;
    int index;

    template <typename T>
    __host__ __device__ T get(const T* preprocessed, const T* main) const {
        switch (type) {
            case PairColType::Preprocessed:
                return preprocessed[index];
            case PairColType::Main:
                return main[index];
        }
            // Should be unreachable, but compilers might complain without a default return.
            // On device, we can use an assertion that will halt execution.
            #ifdef __CUDA_ARCH__
            asm("trap;");
            #endif
            return main[0]; // Return something to satisfy host compilers
    }
};

// NEW: A GPU-compatible replacement for std::pair
template <typename T1, typename T2>
struct Pair {
    T1 first;
    T2 second;

    // Default constructor
     __host__ __device__ Pair() = default;

    // Member-wise constructor
     __host__ __device__ Pair(const T1& f, const T2& s) : first(f), second(s) {}
};

enum ChipId {
    BASE_ALU = 0,
    EXT_ALU = 1,
    BATCH_FRI = 2,
    EXP_REVERSE_BITS = 3,
    FRI_FOLD = 4,
    PUBLIC_VALUES = 5,
    SELECT = 6,
    P2_WIDE = 7,
    P2_SKINNY = 8,
    MEM_CONST = 9,
    MEM_VAR = 10
};

template <typename T>
struct GpuMatrix {
    T*  d_data;
    std::size_t width;  // Use std::size_t to match Rust's usize
    std::size_t height;
};

// Flat representation of a single term in a VirtualPairCol
// Corresponds to (PairCol, Weight)
struct FfiVpcTerm {
    int col_type; // 0 for Preprocessed, 1 for Main
    int col_index;
    Val weight;
};

// Flat representation of a VirtualPairCol
struct FfiVirtualPairCol {
    FfiVpcTerm* terms_ptr;
    int num_terms;
    Val constant;
};

// Flat representation of an Interaction
struct FfiInteraction {
    FfiVirtualPairCol values[8]; // MAX_INTERACTION_VALUES
    int num_values;
    FfiVirtualPairCol multiplicity;
    //int argument_index;
    int kind;
    /// The scope of the interaction.
    int scope;
    bool is_send;
};

