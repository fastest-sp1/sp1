#include <vector>
#include <cmath>
#include <cassert>
#include <map>
#include "bb31_t.hpp" 
#include "utils.hpp"
#include "bb31_quartic_extension_t.hpp"
#include "poseidon2.hpp"
#include "poseidon2_wide.hpp"
#include "sp1-recursion-core-sys-cbindgen.hpp"


using namespace sp1_recursion_core_sys;

// This struct acts as a "handle" that manages all device memory
// associated with a single Merkle tree commitment.
struct GpuMerkleTree {
    bb31_t* d_flat_data;
    int* d_matrix_info;
    std::vector<bb31_t*> digest_layers_device;
    int log_max_height;
};

struct gpu_matrix_t {
    bb31_t*  d_data;
    std::size_t width;  // Use std::size_t to match Rust's usize
    std::size_t height;
};


template <typename F, int WIDTH>
__device__ void poseidon2_permute_mut(F* state) {
    // Use the exact constants from sp1 recursion core implementation's namespace
    using namespace sp1_recursion_core_sys::poseidon2;
    using namespace sp1_recursion_core_sys::constants;
    using namespace sp1_recursion_core_sys::poseidon2_wide;

    // --- Allocate temporary arrays on the stack, just like `event_to_row` does ---
    F external_rounds_state[WIDTH * NUM_EXTERNAL_ROUNDS];
    F internal_rounds_state[WIDTH];
    F internal_rounds_s0[NUM_INTERNAL_ROUNDS - 1];
    F output_state[WIDTH];
    
    F external_sbox[WIDTH * NUM_EXTERNAL_ROUNDS];
    F internal_sbox[NUM_INTERNAL_ROUNDS];

    populate_perm<F>(
        state, // input
        external_rounds_state,
        internal_rounds_state,
        internal_rounds_s0,
        external_sbox,
        internal_sbox,
        output_state // The final result will be here
    );

    // --- Copy the final result back to the original `state` array ---
    for (int i = 0; i < WIDTH; ++i) {
        state[i] = output_state[i];
    }
}


// Hashes all leaf rows of the tallest matrices in parallel ---
// This kernel simulates plonky3:merkle_tree `first_digest_layer`.
template <typename F, int WIDTH, int RATE, int OUT_ELEMS>
__global__ void hash_leaves_kernel(
    const F* flat_data,
    const int* matrix_info,
    int num_tallest_matrices,
    int h_actual, 
  //  int h_padded, // <-- This parameter is now the padded height
    F* digest_layer_out
) {
    int row_idx = threadIdx.x + blockIdx.x * blockDim.x;
    //if (row_idx >= h_padded) return; // Grid is launched for padded height

    // If we are in the padding region, do nothing. The output is already zeroed.
    if (row_idx >= h_actual) {
        return;
    }

    F sponge_state[WIDTH] = {F(0)};
    int input_pos = 0;

    // Absorb the row `row_idx` from all tallest matrices.
    for (int i = 0; i < num_tallest_matrices; ++i) {
        int offset = matrix_info[i * 3 + 0];
        int w = matrix_info[i * 3 + 2];
        const F* row_ptr = flat_data + offset + (size_t)row_idx * w;

        for (int c = 0; c < w; ++c) {
            sponge_state[input_pos++] = row_ptr[c];
            if (input_pos == RATE) {
                poseidon2_permute_mut<F, WIDTH>(sponge_state);
                input_pos = 0;
            }
        }
    }

    if (input_pos > 0) {
        poseidon2_permute_mut<F, WIDTH>(sponge_state);
    }

    // Squeeze and write the output digest.
    F* digest_out = digest_layer_out + (size_t)row_idx * OUT_ELEMS;
    for (int i = 0; i < OUT_ELEMS; ++i) {
        digest_out[i] = sponge_state[i];
    }
}



//prove commit_phase
// This kernel simulates plonky3::merkle_tree `compress_and_inject`.
template <typename F, int WIDTH, int RATE, int OUT_ELEMS>
__global__ void inject_and_compress_kernel(
    const F* prev_layer_digests,
    int num_prev_layer_nodes, //for fri_commit
    F* next_layer_digests,
    int num_compressions, 
    
    // Injection data (can be NULL if no injection)
    const F* flat_data,
    const int* injected_matrix_info,
    int num_injected_matrices,
    int h_injected
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_compressions) return;

    // --- Step 1: Prepare left and right inputs for compression ---
    const F* left = prev_layer_digests + (size_t)tid * 2 * OUT_ELEMS;
    F right_buffer[OUT_ELEMS]; // A temporary buffer on the stack for the right sibling

    // Check if we are processing the last node of an odd-length layer.
    if ((tid == num_compressions - 1) && (num_prev_layer_nodes % 2 != 0)) {
        // If the previous layer had an odd number of nodes, the last node's sibling
        // is a default "zero" digest, mimicking Plonky3's padding behavior.
        for (int k = 0; k < OUT_ELEMS; ++k) {
            right_buffer[k] = F(0);
        }
    } else {
        // Normal case: the right sibling is the next digest in the array.
        const F* right_ptr = left + OUT_ELEMS;
        for (int k = 0; k < OUT_ELEMS; ++k) {
            right_buffer[k] = right_ptr[k];
        }
    }

    // --- Step 2: Perform the first compression (L, R) ---
    F temp_digest_state[WIDTH];
    for(int k = 0; k < OUT_ELEMS; ++k) temp_digest_state[k] = left[k];  
    for(int k = 0; k < OUT_ELEMS; ++k) temp_digest_state[OUT_ELEMS + k] = right_buffer[k];
    for(int k = 2 * OUT_ELEMS; k < WIDTH; ++k) temp_digest_state[k] = F(0);
    poseidon2_permute_mut<F, WIDTH>(temp_digest_state);

    F* dest = next_layer_digests + (size_t)tid * OUT_ELEMS;
    
    // --- Step 3: Handle injection (if any) ---
    if (h_injected > 0) {
        // This part handles hashing the injected rows for the current thread (tid).
        F injected_digest_state[WIDTH] = {F(0)};
        int input_pos = 0;
        for (int mat_idx = 0; mat_idx < num_injected_matrices; ++mat_idx) {
            int offset = injected_matrix_info[mat_idx * 3 + 0];
            int w = injected_matrix_info[mat_idx * 3 + 2];
            const F* row_ptr = flat_data + offset + (size_t)tid * w;
            for (int c = 0; c < w; ++c) {
                injected_digest_state[input_pos++] = row_ptr[c];
                if (input_pos == RATE) {
                    poseidon2_permute_mut<F, WIDTH>(injected_digest_state);
                    input_pos = 0;
                }
            }
        }
        if (input_pos > 0) {
            poseidon2_permute_mut<F, WIDTH>(injected_digest_state);
        }
            
        // Final compression of the two digests (temp_digest_state and injected_digest_state)
        F final_state[WIDTH];
        for(int k=0; k<OUT_ELEMS; ++k) final_state[k] = temp_digest_state[k];
        for(int k=0; k<OUT_ELEMS; ++k) final_state[OUT_ELEMS + k] = injected_digest_state[k];
        for(int k=2*OUT_ELEMS; k<WIDTH; ++k) final_state[k] = F(0);
        poseidon2_permute_mut<F, WIDTH>(final_state);
        
        // Write the final result to the destination.
        for(int k=0; k<OUT_ELEMS; ++k) dest[k] = final_state[k];
    } else {
        // No injection: just write the result of the first compression.
        for(int k=0; k<OUT_ELEMS; ++k) dest[k] = temp_digest_state[k];
    }
}

__global__ void generate_merkle_proofs_kernel(
    bb31_t** digest_layers, // Array of pointers to each layer
    int tree_height,
    const unsigned int* query_indices,
    int num_queries,
    bb31_t* proofs_out,
    const int DIGEST_SIZE
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    unsigned int leaf_index = query_indices[query_idx];
    bb31_t* proof_ptr = proofs_out + (size_t)query_idx * tree_height * DIGEST_SIZE;

    // No need for complex offset calculations anymore!
    for (int depth = 0; depth < tree_height; ++depth) {
        // Get the specific layer's data pointer
        const bb31_t* current_layer_data = digest_layers[depth];

        unsigned int sibling_index = (leaf_index % 2 == 0) ? leaf_index + 1 : leaf_index - 1;
        
        // The position is now just a simple index into the current layer's array
        const bb31_t* sibling_ptr = current_layer_data + (size_t)sibling_index * DIGEST_SIZE;

        // Copy the sibling's digest
        for (int k = 0; k < DIGEST_SIZE; ++k) {
            proof_ptr[depth * DIGEST_SIZE + k] = sibling_ptr[k];
        }

        // Move to the parent for the next layer
        leaf_index /= 2;
    }
}

//For GpuMerkleTreeMmcs::commit
extern "C" int stark_merkle_commit_gpu(
    const bb31_t* flat_data, 
    const int* matrix_info, 
    int num_matrices,
    bb31_t* root_out, 
    void** device_tree_handle) 
{
    if (num_matrices == 0) return -1;

    // === 1. Host-side Setup: Replicate plonky3's grouping logic ===
    // Group matrix indices by their next_power_of_two height.
    // The key is the log2 of the padded height.
    std::map<int, std::vector<int>> matrices_by_log_height;
    int max_log_height = 0;
    for (int i = 0; i < num_matrices; ++i) {
        int h = matrix_info[i * 3 + 1];
        int log_h_pow2 = (h > 0) ? integer_log2(h) : 0;
        matrices_by_log_height[log_h_pow2].push_back(i);
        if (log_h_pow2 > max_log_height) {
            max_log_height = log_h_pow2;
        }
    }

    // === 2. GPU Memory Allocation for inputs & handle ===
    size_t total_elements = 0;
    if (num_matrices > 0) {
        int last_mat_idx = num_matrices - 1;
        total_elements = matrix_info[last_mat_idx*3+0] + (size_t)matrix_info[last_mat_idx*3+1] * matrix_info[last_mat_idx*3+2];
    }

    bb31_t *d_flat_data;
    int *d_matrix_info;
    CUDA_CHECK(cudaMalloc(&d_flat_data, total_elements * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_matrix_info, (size_t)num_matrices * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_flat_data, flat_data, total_elements * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matrix_info, matrix_info, (size_t)num_matrices * 3 * sizeof(int), cudaMemcpyHostToDevice));
    
    GpuMerkleTree* tree = new GpuMerkleTree();
    tree->d_flat_data = d_flat_data;
    tree->d_matrix_info = d_matrix_info;
    tree->log_max_height = max_log_height;

    // === 3. GPU Kernel Launch Configuration ===
    int num_threads = 256;
    dim3 block_dim(num_threads);

    // === 4. Stage 1: Leaf Hashing for TALLEST matrices ===
    const auto& tallest_indices = matrices_by_log_height[max_log_height];
   
    // Create and transfer info for only the tallest matrices for the hash_leaves_kernel
    std::vector<int> tallest_matrix_info_host;
    for (int idx : tallest_indices) {
        tallest_matrix_info_host.push_back(matrix_info[idx*3+0]);
        tallest_matrix_info_host.push_back(matrix_info[idx*3+1]);
        tallest_matrix_info_host.push_back(matrix_info[idx*3+2]);
    }
    int* d_tallest_matrix_info;
    CUDA_CHECK(cudaMalloc(&d_tallest_matrix_info, tallest_matrix_info_host.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tallest_matrix_info, tallest_matrix_info_host.data(), tallest_matrix_info_host.size() * sizeof(int), cudaMemcpyHostToDevice));

    int h_tallest = 1 << max_log_height;
    int current_layer_active_digests = h_tallest;
    int current_layer_padded_size = current_layer_active_digests;
    if (current_layer_padded_size > 1 && current_layer_padded_size % 2 != 0) {
        current_layer_padded_size++;
    }
    
    bb31_t* d_layer_0;
    CUDA_CHECK(cudaMalloc(&d_layer_0, current_layer_padded_size * DIGEST_SIZE * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemset(d_layer_0, 0, current_layer_padded_size * DIGEST_SIZE * sizeof(bb31_t)));
    tree->digest_layers_device.push_back(d_layer_0);
    
    //int h_padded = next_power_of_two(h_tallest);
    dim3 grid_dim_leaves((h_tallest + num_threads - 1) / num_threads);
    hash_leaves_kernel<bb31_t, 16, 8, DIGEST_SIZE><<<grid_dim_leaves, block_dim>>>(
        d_flat_data, d_tallest_matrix_info, tallest_indices.size(), h_tallest, d_layer_0);
    CUDA_CHECK(cudaFree(d_tallest_matrix_info));

    // === 5. Stage 2: Compressing Layers with INJECTION ===
    bb31_t* d_prev_layer = d_layer_0;
    for (int current_log_size = max_log_height; current_log_size > 0; --current_log_size) {
        int num_compressions = current_layer_active_digests / 2;
        if (num_compressions == 0) break;

        int next_log_size = current_log_size - 1;
        int next_layer_height = 1 << next_log_size;
        
        // Find matrices to inject at this level
        const auto& injection_indices_map_entry = matrices_by_log_height.find(next_log_size);
        bool inject = injection_indices_map_entry != matrices_by_log_height.end();

        // Prepare info for injected matrices if they exist
        int* d_injected_info = nullptr;
        int num_injected_matrices = 0;
        int h_injected = 0;
        if (inject) {
            const auto& injection_indices = injection_indices_map_entry->second;
            num_injected_matrices = injection_indices.size();
            h_injected = 1 << next_log_size;
            
            std::vector<int> injected_info_host;
            for (int idx : injection_indices) {
                injected_info_host.push_back(matrix_info[idx*3+0]);
                injected_info_host.push_back(matrix_info[idx*3+1]);
                injected_info_host.push_back(matrix_info[idx*3+2]);
            }
            CUDA_CHECK(cudaMalloc(&d_injected_info, injected_info_host.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_injected_info, injected_info_host.data(), injected_info_host.size() * sizeof(int), cudaMemcpyHostToDevice));
        }

        int next_layer_active_size = num_compressions;
        int next_layer_padded_size = next_layer_active_size;
        if (next_layer_padded_size > 1 && next_layer_padded_size % 2 != 0) {
            next_layer_padded_size++;
        }

        bb31_t* d_next_layer;
        CUDA_CHECK(cudaMalloc(&d_next_layer, (size_t)next_layer_padded_size * DIGEST_SIZE * sizeof(bb31_t)));
        CUDA_CHECK(cudaMemset(d_next_layer, 0, (size_t)next_layer_padded_size * DIGEST_SIZE * sizeof(bb31_t)));
        tree->digest_layers_device.push_back(d_next_layer);

        dim3 grid_dim((num_compressions + num_threads - 1) / num_threads);

        inject_and_compress_kernel<bb31_t, 16, 8, DIGEST_SIZE><<<grid_dim, block_dim>>>(
                d_prev_layer, 
                0, //require  the input maxtrix height is 2^n
                d_next_layer, 
                num_compressions,
                d_flat_data, 
                d_injected_info, 
                num_injected_matrices, 
                h_injected);

        if (d_injected_info) CUDA_CHECK(cudaFree(d_injected_info));
        
        d_prev_layer = d_next_layer;
        current_layer_active_digests = next_layer_active_size;
        
    }
    
    // --- 6. Finalize ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(root_out, d_prev_layer, DIGEST_SIZE * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    *device_tree_handle = tree; // Return the handle containing all GPU resources
    
    return 0;
}


extern "C" void stark_merkle_free_gpu(void* device_tree_handle) {
    if (!device_tree_handle) return;
    
    GpuMerkleTree* tree = static_cast<GpuMerkleTree*>(device_tree_handle);
    
    // Free each individual digest layer allocated on the GPU.
    for (bb31_t* layer_ptr : tree->digest_layers_device) {
        if (layer_ptr) {
            cudaFree(layer_ptr);
        }
    }
    
    // Free the other top-level GPU buffers.
    if (tree->d_flat_data) {
        cudaFree(tree->d_flat_data);
    }
    if (tree->d_matrix_info) {
        cudaFree(tree->d_matrix_info);
    }
    
    // Finally, free the host-side struct itself.
    delete tree;
}

////For GpuMerkleTreeMmcs::open_batch
extern "C" int stark_merkle_open_batch_gpu(
    void* device_tree_handle, 
    int index, 
    const int* matrix_info, 
    int num_matrices,
    bb31_t* flat_opened_values_out, 
    bb31_t* proof_out)
{
    if (!device_tree_handle) return -1;
    GpuMerkleTree* tree = static_cast<GpuMerkleTree*>(device_tree_handle);

    // --- 1. Generate the authentication path (proof) ---
    int current_index = index;
    for (int i = 0; i < tree->log_max_height; ++i) {
        int sibling_idx = current_index ^ 1;
        bb31_t* d_current_layer = tree->digest_layers_device[i];
        
        // Copy one sibling digest from GPU to the host proof buffer
        CUDA_CHECK(cudaMemcpy(
            proof_out + (size_t)i * DIGEST_SIZE,      // Destination on host
            d_current_layer + (size_t)sibling_idx * DIGEST_SIZE, // Source on device
            DIGEST_SIZE * sizeof(bb31_t),
            cudaMemcpyDeviceToHost
        ));
        
        current_index >>= 1;
    }
    
    // --- 2. Get the opened values (leaves) ---
    // This is complex because different matrices have different heights and widths.
    // The opened row for each matrix needs to be copied.
    bb31_t* current_out_ptr = flat_opened_values_out;
    for (int i = 0; i < num_matrices; ++i) {
        int h = matrix_info[i*3+1];
        int w = matrix_info[i*3+2];
        
        int log_h = integer_log2(h);
        int bits_reduced = tree->log_max_height - log_h;
        int reduced_index = index >> bits_reduced;

        // Pointer to the start of the i-th matrix's data on GPU
        const bb31_t* d_matrix_start = tree->d_flat_data + matrix_info[i*3+0];
        // Pointer to the specific row to be opened
        const bb31_t* d_row_ptr = d_matrix_start + (size_t)reduced_index * w;

        // Copy the row from GPU to the correct position in the host output buffer
        CUDA_CHECK(cudaMemcpy(
            current_out_ptr,
            d_row_ptr,
            w * sizeof(bb31_t),
            cudaMemcpyDeviceToHost
        ));
        
        current_out_ptr += w;
    }

    return 0;
}

////For GpuMerkleTreeMmcs::open_batches_batched
extern "C" int stark_merkle_generate_proofs_gpu(
    // Inputs: A batch of queries for multiple trees
    const void* const* h_prover_data_handles,   // Host array of GpuMerkleTree* handles
    const unsigned int* h_query_indices_flat,   // Host array of all indices
    const unsigned int* h_query_offsets,        // Host array of offsets
    int num_trees,
    int total_queries,
    int digest_size,

    // Output: A pointer to a host buffer where the flattened proofs will be written.
    bb31_t* h_proofs_out_flat
) {
    if (total_queries == 0) return 0;

    // --- 1. Calculate total size for the GPU output buffer for proofs ---
    size_t total_proof_elements = 0;
    for (int i = 0; i < num_trees; ++i) {
        const GpuMerkleTree* tree = static_cast<const GpuMerkleTree*>(h_prover_data_handles[i]);
        unsigned int num_queries_for_tree = h_query_offsets[i+1] - h_query_offsets[i];
        total_proof_elements += (size_t)num_queries_for_tree * tree->log_max_height;
        //printf("&&& stark_merkle_generate_proofs_gpu--000,num=%u, total_queries=%u,tree.addr=%p, log_max_height=%u\n", 
         //   i, total_queries, h_prover_data_handles[i], tree->log_max_height);
    }
    total_proof_elements *= digest_size;
    
    bb31_t* d_proofs_out_flat;
    CUDA_CHECK(cudaMalloc(&d_proofs_out_flat, total_proof_elements * sizeof(bb31_t)));

    // --- 2. Copy all query indices to the GPU ---
    unsigned int* d_query_indices_flat;
    CUDA_CHECK(cudaMalloc(&d_query_indices_flat, (size_t)total_queries * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_query_indices_flat, h_query_indices_flat, (size_t)total_queries * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // --- 3. Launch one kernel for each tree's proof generation ---
    size_t current_proof_output_offset = 0;
    for (int i = 0; i < num_trees; ++i) {
        const GpuMerkleTree* tree = static_cast<const GpuMerkleTree*>(h_prover_data_handles[i]);
        unsigned int query_start_offset = h_query_offsets[i];
        unsigned int num_queries_for_tree = h_query_offsets[i+1] - query_start_offset;
        if (num_queries_for_tree > 0) {
            bb31_t** d_digest_layers;
            CUDA_CHECK(cudaMalloc(&d_digest_layers, tree->digest_layers_device.size() * sizeof(bb31_t*)));
            CUDA_CHECK(cudaMemcpy(d_digest_layers, tree->digest_layers_device.data(), tree->digest_layers_device.size() * sizeof(bb31_t*), cudaMemcpyHostToDevice));

            int num_threads = 256;
            dim3 block_dim(num_threads);
            dim3 grid_dim((num_queries_for_tree + num_threads - 1) / num_threads);

            // We need the first layer of digests (the leaves' parents)
            // The GpuMerkleTree struct stores all layers. The one we need for proofs
            // is the one containing the siblings of the leaves.
            // Assuming  tree layers are stored leaf-parents-upwards.
            //const bb31_t* d_tree_leaf_parent_layer = tree->digest_layers_device[0]; 

            generate_merkle_proofs_kernel<<<grid_dim, block_dim>>>(
                d_digest_layers,
                tree->log_max_height,
                d_query_indices_flat + query_start_offset,
                num_queries_for_tree,
                d_proofs_out_flat + current_proof_output_offset,
                digest_size
            );

            CUDA_CHECK(cudaFree(d_digest_layers));
        }
        current_proof_output_offset += (size_t)num_queries_for_tree * tree->log_max_height * digest_size;
    }

    // --- 4. Synchronize and copy all generated proofs back to the host ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_proofs_out_flat, d_proofs_out_flat, total_proof_elements * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    // --- 5. Cleanup ---
    CUDA_CHECK(cudaFree(d_proofs_out_flat));
    CUDA_CHECK(cudaFree(d_query_indices_flat));
    
    return 0;
}

//For Prover::commit_phase_gpu
extern "C" int fri_commit_on_gpu(
    // Input: Evaluations vector already on the GPU
    const bb31_quartic_extension_t* d_evals,
    int num_evals,

    // Output:
    bb31_t* h_root_out,                      // Pointer to a HOST buffer for the root
    void** device_tree_handle_out        // Pointer to a HOST variable for the handle
) {
    if (num_evals == 0) return -1; // Cannot commit to empty data
    if (num_evals % 2 != 0) {
        // This should not happen in FRI, as each layer is a pairing.
        return -2; // Invalid input length
    }

    // --- 1. Interpret the EF evals as a base field matrix ---
    // An FRI layer is a column matrix of EF elements.
    // ExtensionMmcs treats this as a matrix of base field elements with `EF::DIMENSION` columns.
     const int h = num_evals / 2; // The height of the logical matrix
    const int w = 8;             // The width in base field elements (2 * 4)
    const size_t total_base_elements = (size_t)h * w;
    const bb31_t* d_flat_data = (const bb31_t*)d_evals;

    // We only have one matrix in this case.
    const int num_matrices = 1;
    int matrix_info[3] = {0, h, w}; // offset=0, height=h, width=w

    // --- 2. GPU Memory Allocation for the new tree's metadata ---
    // The leaf data `d_flat_data` is already on the GPU, so we don't copy it.
    // We just need to copy the matrix_info.
    int* d_matrix_info;
    CUDA_CHECK(cudaMalloc(&d_matrix_info, sizeof(matrix_info)));
    CUDA_CHECK(cudaMemcpy(d_matrix_info, matrix_info, sizeof(matrix_info), cudaMemcpyHostToDevice));

    // Create the handle struct. We will store pointers to GPU memory in it.
    GpuMerkleTree* tree = new GpuMerkleTree();
    // We don't own d_flat_data, it was passed in. So we don't store it for later freeing.
    // Or, if this function takes ownership, we should store it.
    // Let's assume the CALLER (commit_phase_gpu) manages the lifecycle of d_evals.
    tree->d_flat_data = nullptr; // Not owned by this handle.
    tree->d_matrix_info = d_matrix_info; // Owned by this handle.
    tree->log_max_height = (h > 0) ? integer_log2(next_power_of_two(h)) : 0;

    // --- 3. GPU Kernel Launch Configuration ---
    int num_threads = 256;
    dim3 block_dim(num_threads);

    // --- 4. Stage 1: Leaf Hashing ---
    // Since there's only one "matrix" (the layer), it's the tallest.
    int h_padded = next_power_of_two(h);
    //printf("=======tree-log_max_h:%d, h_padded=%d\n", tree->log_max_height, h_padded);
    // Create and pad the first digest layer buffer
    bb31_t* d_layer_0;
    CUDA_CHECK(cudaMalloc(&d_layer_0, (size_t)h_padded * DIGEST_SIZE * sizeof(bb31_t)));
    if (h_padded > h) { // Zero out the padding area if necessary
        CUDA_CHECK(cudaMemset(d_layer_0 + (size_t)h * DIGEST_SIZE, 0, (size_t)(h_padded - h) * DIGEST_SIZE * sizeof(bb31_t)));
    }
    tree->digest_layers_device.push_back(d_layer_0);
  //printf(" ===fri_commit_on_gpu 222\n");
    dim3 grid_dim_leaves((h + num_threads - 1) / num_threads);
    hash_leaves_kernel<bb31_t, 16, 8, DIGEST_SIZE><<<grid_dim_leaves, block_dim>>>(
        d_flat_data, 
        d_matrix_info, 
        num_matrices, 
        h,          // Pass original height as h_actual
        //h_padded,   // Pass padded height as h_padded
        d_layer_0
    );

    //debug
    /*CUDA_CHECK(cudaDeviceSynchronize());
    bb31_t output_layer[h_padded * DIGEST_SIZE];
    CUDA_CHECK(cudaMemcpy(output_layer, d_layer_0, h_padded * DIGEST_SIZE * sizeof(bb31_t), cudaMemcpyDeviceToHost));  
    printf("GPU----layer 0, size=%d, values\n", h_padded * DIGEST_SIZE);
    for (int i=0; i < h_padded * DIGEST_SIZE ; ++i) {
        printf(" %u ",  output_layer[i].as_canonical_u32());
        //printf(" %u ",  output_layer[i]);
        if ((i+1)%8 == 0 ) printf(" \n");
    }
    printf(" \n");
    //end debug
     */

    // --- 5. Stage 2: Compressing Layers (no injection in base FRI) ---
    bb31_t* d_prev_layer = d_layer_0;
    int current_layer_nodes = h;

    // Loop until we have a single root node.
    //debug
    int layer_num=0;
    while (current_layer_nodes > 1) {
        // Calculate the number of compressions needed for the next layer.
        // This is equivalent to `ceil(current_layer_nodes / 2)`.
        int num_compressions = (current_layer_nodes + 1) / 2;
        
        // Allocate the buffer for the next layer.
        bb31_t* d_next_layer;
        CUDA_CHECK(cudaMalloc(&d_next_layer, (size_t)num_compressions * DIGEST_SIZE * sizeof(bb31_t)));
        tree->digest_layers_device.push_back(d_next_layer);

        dim3 grid_dim((num_compressions + num_threads - 1) / num_threads);
        
        // Call the modified compression kernel.
        inject_and_compress_kernel<bb31_t, 16, 8, DIGEST_SIZE><<<grid_dim, block_dim>>>(
                d_prev_layer,
                current_layer_nodes, // Pass the actual number of nodes from the previous layer
                d_next_layer,
                num_compressions,
                nullptr, nullptr, 0, 0); // No injection for FRI commit phase
        //debug
    /*CUDA_CHECK(cudaDeviceSynchronize());
    //bb31_t output_layer[h_padded * DIGEST_SIZE];
    CUDA_CHECK(cudaMemcpy(output_layer, d_next_layer, num_compressions * DIGEST_SIZE * sizeof(bb31_t), cudaMemcpyDeviceToHost));  
    printf("GPU----layer :%d, size=%d, values\n", ++layer_num,  num_compressions * DIGEST_SIZE);
    for (int i=0; i < num_compressions * DIGEST_SIZE ; ++i) {
        printf(" %u ",  output_layer[i].as_canonical_u32());
        //printf(" %u ",  output_layer[i]);
        if ((i+1)%8 == 0 ) printf(" \n");
    }
    printf(" \n"); */
    //end debug

        
        d_prev_layer = d_next_layer;
        current_layer_nodes = num_compressions;
    }
   //printf(" ===fri_commit_on_gpu 333333\n");
    // --- 6. Finalize ---
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_root_out, d_prev_layer, DIGEST_SIZE * sizeof(bb31_t), cudaMemcpyDeviceToHost));  
    *device_tree_handle_out = tree; // Return the handle
    
    return 0;
}

// In merkle.cpp

//For Prover::answer_queries_gpu
extern "C" int stark_fri_generate_proofs_gpu(
    const void* const* h_prover_data_handles,
    const unsigned int* h_query_indices_flat,
    const unsigned int* h_query_offsets,
    int num_trees,
    int total_queries,
    int digest_size,
    bb31_t* h_proofs_out_flat
) {
    if (total_queries == 0) return 0;

    // --- 1. Calculate total size for GPU output buffer ---
    size_t total_proof_elements = 0;
    for (int i = 0; i < num_trees; ++i) {
        const GpuMerkleTree* tree = static_cast<const GpuMerkleTree*>(h_prover_data_handles[i]);
        unsigned int num_queries_for_tree = h_query_offsets[i+1] - h_query_offsets[i];
        total_proof_elements += (size_t)num_queries_for_tree * tree->log_max_height;
    }
    total_proof_elements *= digest_size;  
    bb31_t* d_proofs_out_flat;
    CUDA_CHECK(cudaMalloc(&d_proofs_out_flat, total_proof_elements * sizeof(bb31_t)));

    // --- 2. Copy all query indices to the GPU ---
    unsigned int* d_query_indices_flat;
    CUDA_CHECK(cudaMalloc(&d_query_indices_flat, (size_t)total_queries * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_query_indices_flat, h_query_indices_flat, (size_t)total_queries * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // --- 3. Launch one kernel for each tree's proof generation ---
    size_t current_proof_output_offset = 0;
    for (int i = 0; i < num_trees; ++i) {
        const GpuMerkleTree* tree = static_cast<const GpuMerkleTree*>(h_prover_data_handles[i]);
        unsigned int query_start_offset = h_query_offsets[i];
        unsigned int num_queries_for_tree = h_query_offsets[i+1] - query_start_offset;
        
        if (num_queries_for_tree > 0) {
            // ================== THE FIX IS HERE ==================
            // The `tree->digest_layers_device` is a `std::vector<bb31_t*>` on the HOST.
            // The kernel needs an array of `bb31_t*` pointers on the DEVICE.
            // We need to allocate memory for this pointer array on the GPU and copy the pointers.

            bb31_t** d_digest_layers;
            size_t num_layers = tree->digest_layers_device.size();
            CUDA_CHECK(cudaMalloc(&d_digest_layers, num_layers * sizeof(bb31_t*)));
            CUDA_CHECK(cudaMemcpy(d_digest_layers, tree->digest_layers_device.data(), num_layers * sizeof(bb31_t*), cudaMemcpyHostToDevice));
            
            // ======================================================

            int num_threads = 256;
            dim3 block_dim(num_threads);
            dim3 grid_dim((num_queries_for_tree + num_threads - 1) / num_threads);

            generate_merkle_proofs_kernel<<<grid_dim, block_dim>>>(
                d_digest_layers, // <-- Now passing the correct type: bb31_t**
                tree->log_max_height,
                d_query_indices_flat + query_start_offset,
                num_queries_for_tree,
                d_proofs_out_flat + current_proof_output_offset,
                digest_size
            );
            
            // Don't forget to free the device-side pointer array
            CUDA_CHECK(cudaFree(d_digest_layers));
        }
        current_proof_output_offset += (size_t)num_queries_for_tree * tree->log_max_height * digest_size;
    }

    // --- 4. Synchronize and copy all generated proofs back to the host ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_proofs_out_flat, d_proofs_out_flat, total_proof_elements * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    // --- 5. Cleanup ---
    CUDA_CHECK(cudaFree(d_proofs_out_flat));
    CUDA_CHECK(cudaFree(d_query_indices_flat));
    
    return 0;
}

extern "C" int stark_test_hash_leaves_gpu(
    const bb31_t* flat_data, const int* matrix_info, int num_tallest_matrices,
    int h_tallest, bb31_t* digest_out)
{
    if (h_tallest == 0 || num_tallest_matrices == 0) return 0;
    
    // --- GPU Memory Allocation & Transfer ---
    bb31_t *d_flat_data,  *d_digest_out;
    int *d_matrix_info;
    
    // Calculate total size from the passed info
    size_t total_elements = 0;
    for (int i=0; i<num_tallest_matrices; ++i) {
        total_elements += (size_t)matrix_info[i*3+1] * matrix_info[i*3+2];
    }

    CUDA_CHECK(cudaMalloc(&d_flat_data, total_elements * sizeof(bb31_t)));
    CUDA_CHECK(cudaMalloc(&d_matrix_info, (size_t)num_tallest_matrices * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_digest_out, (size_t)h_tallest * DIGEST_SIZE * sizeof(bb31_t)));

    CUDA_CHECK(cudaMemcpy(d_flat_data, flat_data, total_elements * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matrix_info, matrix_info, (size_t)num_tallest_matrices * 3 * sizeof(int), cudaMemcpyHostToDevice));
    
    // --- Launch Kernel ---
    int num_threads = 256;
    dim3 block_dim(num_threads);
    dim3 grid_dim_leaves((h_tallest + num_threads - 1) / num_threads);

    //int h_padded = next_power_of_two(h_tallest);
    hash_leaves_kernel<bb31_t, 16, 8, DIGEST_SIZE><<<grid_dim_leaves, block_dim>>>(
        d_flat_data, d_matrix_info, num_tallest_matrices, h_tallest, d_digest_out);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    
    // --- Copy Result Back ---
    CUDA_CHECK(cudaMemcpy(digest_out, d_digest_out, (size_t)h_tallest * DIGEST_SIZE * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_flat_data));
    CUDA_CHECK(cudaFree(d_matrix_info));
    CUDA_CHECK(cudaFree(d_digest_out));
    
    return 0;
}


template <typename F, int WIDTH>
__global__ void test_permute_kernel(F* state) {
    // We only need one thread to perform one permutation.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        poseidon2_permute_mut<F, WIDTH>(state);
    }
}


// --- The FFI wrapper for the test ---
extern "C" int stark_test_poseidon2_permute_gpu(bb31_t* state) {
    const int WIDTH = 16; // Hardcoded for Poseidon2BabyBear<16>
    
    bb31_t* d_state;
    CUDA_CHECK(cudaMalloc(&d_state, WIDTH * sizeof(bb31_t)));
    CUDA_CHECK(cudaMemcpy(d_state, state, WIDTH * sizeof(bb31_t), cudaMemcpyHostToDevice));

    // Launch a single thread in a single block to perform the permutation.
    dim3 grid_dim(1);
    dim3 block_dim(1);
    
    test_permute_kernel<bb31_t, WIDTH><<<grid_dim, block_dim>>>(d_state);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(state, d_state, WIDTH * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_state));
    
    return 0;
}

/////////////////new sep 10 /////////////////////////


template <typename F, int WIDTH, int RATE, int OUT_ELEMS>
__global__ void hash_leaves_from_pointers_kernel(
    const gpu_matrix_t* d_matrices, // Array of matrix descriptors (ON DEVICE)
    int num_matrices,
    int leaf_height,                // The height of all matrices in this batch
    F* d_leaf_digests               // Output buffer for leaf digests
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= leaf_height) return;

    F sponge_state[WIDTH] = {F(0)};
    int input_pos = 0;

    // This thread is responsible for hashing all elements for `row_idx`.
    // It iterates through all matrices passed in `d_matrices`.
    for (int i = 0; i < num_matrices; ++i) {
        // Since we group by height, we can assume mat.height == leaf_height.
        const gpu_matrix_t& mat = d_matrices[i];
        
        // Cast the void* to the correct type.
        const F* mat_ptr = static_cast<const F*>(mat.d_data);
        const F* row_ptr = mat_ptr + (size_t)row_idx * mat.width;

        for (int c = 0; c < mat.width; ++c) {
            sponge_state[input_pos++] = row_ptr[c];
            if (input_pos == RATE) {
                poseidon2_permute_mut<F, WIDTH>(sponge_state);
                input_pos = 0;
            }
        }
    }

    // Final permutation if there are remaining elements in the sponge.
    if (input_pos > 0) {
        poseidon2_permute_mut<F, WIDTH>(sponge_state);
    }

    // Squeeze and write the output digest.
    F* digest_out = d_leaf_digests + (size_t)row_idx * OUT_ELEMS;
    for (int i = 0; i < OUT_ELEMS; ++i) {
        digest_out[i] = sponge_state[i];
    }
}

template <typename F, int WIDTH, int OUT_ELEMS>
__global__ void compress_two_digests_kernel(
    const F* d_input_A,
    const F* d_input_B,
    F* d_output,
    int num_pairs
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_pairs) return;

    const F* left = d_input_A + (size_t)tid * OUT_ELEMS;
    const F* right = d_input_B + (size_t)tid * OUT_ELEMS;

    F compression_state[WIDTH];
    for(int k=0; k<OUT_ELEMS; ++k) 
        compression_state[k] = left[k];
    
    for(int k=0; k<OUT_ELEMS; ++k) 
        compression_state[OUT_ELEMS + k] = right[k];

    for(int k=2*OUT_ELEMS; k<WIDTH; ++k) 
        compression_state[k] = F(0);
    poseidon2_permute_mut<F, WIDTH>(compression_state);

    F* dest = d_output + (size_t)tid * OUT_ELEMS;
    for(int k=0; k<OUT_ELEMS; ++k) 
        dest[k] = compression_state[k];
}

template <typename F, int WIDTH, int OUT_ELEMS>
__global__ void compress_layer_kernel(
    const F* d_prev_layer,
    int num_prev_layer_nodes,
    F* d_next_layer,
    int num_compressions // This is num_prev_layer_nodes / 2
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_compressions) return;

    // Get left sibling
    const F* left = d_prev_layer + (size_t)tid * 2 * OUT_ELEMS;

    // Get right sibling, handling padding for odd-length layers
    F right_buffer[OUT_ELEMS];
    if ((tid == num_compressions - 1) && (num_prev_layer_nodes % 2 != 0)) {
        for (int k = 0; k < OUT_ELEMS; ++k) right_buffer[k] = F(0);
    } else {
        const F* right_ptr = left + OUT_ELEMS;
        for (int k = 0; k < OUT_ELEMS; ++k) right_buffer[k] = right_ptr[k];
    }
    
    // Perform the compression
    F compression_state[WIDTH];
    for(int k = 0; k < OUT_ELEMS; ++k) compression_state[k] = left[k];  
    for(int k = 0; k < OUT_ELEMS; ++k) compression_state[OUT_ELEMS + k] = right_buffer[k];
    for(int k = 2 * OUT_ELEMS; k < WIDTH; ++k) compression_state[k] = F(0);
    poseidon2_permute_mut<F, WIDTH>(compression_state);

    // Write the result
    F* dest = d_next_layer + (size_t)tid * OUT_ELEMS;
    for(int k = 0; k < OUT_ELEMS; ++k) dest[k] = compression_state[k];
}

// Helper function to launch the leaf hasher for a set of matrices
bb31_t* hash_leaves_for_height(const std::vector<gpu_matrix_t>& h_matrices_for_height) {
    if (h_matrices_for_height.empty()) return nullptr; // Should not happen

    //tranfer matrix to row-major
    int num_threads = 256;
    dim3 block_dim(num_threads);
    /*std::vector<gpu_matrix_t> new_matrices;//the rust need the the old data
   
    for (gpu_matrix_t mat : h_matrices_for_height) {
        size_t lde_h  = mat.height;
        int log_lde_h = integer_log2(mat.height);
        int w =mat.width;
        dim3 grid_dim_flat_ldeh((lde_h * w + num_threads - 1) / num_threads);

        gpu_matrix_t local_mat;
        local_mat.height = lde_h;
        local_mat.width = w;
        CUDA_CHECK(cudaMalloc(&local_mat.d_data, w * lde_h * sizeof(bb31_t)));
        CUDA_CHECK(cudaMemcpy(local_mat.d_data, mat.d_data,  w * lde_h * sizeof(bb31_t), cudaMemcpyDeviceToDevice));

        bit_reverse_rows_kernel<bb31_t><<<grid_dim_flat_ldeh, block_dim>>>(local_mat.d_data, lde_h, w, log_lde_h);
        new_matrices.push_back(local_mat);
    }
    */

    int leaf_height = h_matrices_for_height[0].height;

    // Transfer matrix descriptors to the device
    gpu_matrix_t* d_matrices;
    CUDA_CHECK(cudaMalloc(&d_matrices, h_matrices_for_height.size() * sizeof(gpu_matrix_t)));
    CUDA_CHECK(cudaMemcpy(d_matrices, h_matrices_for_height.data(), h_matrices_for_height.size() * sizeof(gpu_matrix_t), cudaMemcpyHostToDevice));

    // Allocate output buffer for digests
    bb31_t* d_leaf_digests;
    CUDA_CHECK(cudaMalloc(&d_leaf_digests, (size_t)leaf_height * DIGEST_SIZE * sizeof(bb31_t)));

    // Launch kernel
    //dim3 block_dim(256);
    dim3 grid_dim((leaf_height + 256 - 1) / 256);
    hash_leaves_from_pointers_kernel<bb31_t, 16, 8, DIGEST_SIZE><<<grid_dim, block_dim>>>(
        d_matrices,
        h_matrices_for_height.size(),
        leaf_height,
        d_leaf_digests
    );

    CUDA_CHECK(cudaFree(d_matrices));
    // The caller is responsible for freeing d_leaf_digests
    return d_leaf_digests;
}


// In merkle.cu

extern "C" int stark_merkle_commit_data_in_gpu(
    const gpu_matrix_t* h_matrices,
    int num_matrices,
    bb31_t* root_out,
    void** device_tree_handle) 
{
    if (num_matrices == 0) return -1;

    // === 1. Host-side Setup  ===
    std::map<int, std::vector<int>> matrices_by_log_height;
    int max_log_height = 0;
    for (int i = 0; i < num_matrices; ++i) {
        int h = h_matrices[i].height;
        int log_h_pow2 = (h > 0) ? integer_log2(h) : 0;
        matrices_by_log_height[log_h_pow2].push_back(i);
        if (log_h_pow2 > max_log_height) 
            max_log_height = log_h_pow2;
    }

    // === 2. GPU Handle Setup  ===
    GpuMerkleTree* tree = new GpuMerkleTree();
    tree->log_max_height = max_log_height;

    // === 3. Stage 1: Leaf Hashing for TALLEST matrices ===
    const auto& tallest_indices = matrices_by_log_height.at(max_log_height);
    std::vector<gpu_matrix_t> h_tallest_matrices;
    for (int idx : tallest_indices) {
        h_tallest_matrices.push_back(h_matrices[idx]);
    }
    
    bb31_t* d_current_layer = hash_leaves_for_height(h_tallest_matrices);
    tree->digest_layers_device.push_back(d_current_layer);
    int current_layer_height = 1 << max_log_height;
    // === 4. REFACTORED Stage 2: Compressing Layers with Correct Injection ===
    for (int current_log_size = max_log_height; current_log_size > 0; --current_log_size) {
        int num_nodes_prev_layer = current_layer_height;
        int num_compressions = num_nodes_prev_layer / 2;
        if (num_compressions == 0) break;

        bb31_t* d_compressed_layer;
        CUDA_CHECK(cudaMalloc(&d_compressed_layer, (size_t)num_compressions * DIGEST_SIZE * sizeof(bb31_t)));
        
        // A. Compress the current layer
        dim3 grid_dim((num_compressions + 255) / 256);
        dim3 block_dim(256);
        compress_layer_kernel<bb31_t, 16, DIGEST_SIZE><<<grid_dim, block_dim>>>(
            d_current_layer, num_nodes_prev_layer, d_compressed_layer, num_compressions);
        
        // Check for matrices to inject at the new height
        int next_log_size = current_log_size - 1;
        auto injection_entry = matrices_by_log_height.find(next_log_size);

        bb31_t* d_next_layer;
        if (injection_entry != matrices_by_log_height.end()) {
            // B. We have an injection! Hash the leaves of the injected matrices.
            const auto& injection_indices = injection_entry->second;
            std::vector<gpu_matrix_t> h_injected_matrices;
            for (int idx : injection_indices) h_injected_matrices.push_back(h_matrices[idx]);
            
            bb31_t* d_injected_leaves = hash_leaves_for_height(h_injected_matrices);
            
            d_next_layer = d_compressed_layer; // Reuse the buffer for the final output
            
            // This is a new kernel that takes two input digest arrays.
            compress_two_digests_kernel<bb31_t, 16, DIGEST_SIZE><<<grid_dim, block_dim>>>(
                d_compressed_layer, // Input A
                d_injected_leaves,  // Input B
                d_next_layer,       // Output
                num_compressions
            );
            
            cudaFree(d_injected_leaves);
        } else {
            // D. No injection, the next layer is just the compressed layer.
            d_next_layer = d_compressed_layer;
        }

        d_current_layer = d_next_layer;
        current_layer_height = num_compressions;
        tree->digest_layers_device.push_back(d_current_layer);
    }
    
    // --- 5. Finalize (Unchanged) ---
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(root_out, d_current_layer, DIGEST_SIZE * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    *device_tree_handle = tree;
    // Don't free the final d_current_layer, it's the root layer and is stored in the tree handle.
    
    return 0;
}
//test
extern "C" int stark_test_hash_leaves_data_in_gpu(
    const gpu_matrix_t* h_matrices, // Array of structs, from HOST
    int num_matrices,
    bb31_t* hash_out) 
{
    if (num_matrices == 0) return -1;

    // === 1. Host-side Setup: Replicate plonky3's grouping logic ===
    // Group matrix indices by their next_power_of_two height.
    // The key is the log2 of the padded height.
    std::map<int, std::vector<int>> matrices_by_log_height;
    int max_log_height = 0;
    for (int i = 0; i < num_matrices; ++i) {
        int h = h_matrices[i].height;
        int log_h_pow2 = (h > 0) ? integer_log2(h) : 0;
        matrices_by_log_height[log_h_pow2].push_back(i);
        if (log_h_pow2 > max_log_height) {
            max_log_height = log_h_pow2;
        }
    }

    
    // === 3. Stage 1: Leaf Hashing for TALLEST matrices ===
    const auto& tallest_indices = matrices_by_log_height[max_log_height];
    std::vector<gpu_matrix_t> h_tallest_matrices;
    for (int idx : tallest_indices) {
        h_tallest_matrices.push_back(h_matrices[idx]);
    }
    
    bb31_t* d_current_layer = hash_leaves_for_height(h_tallest_matrices);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hash_out, d_current_layer, h_tallest_matrices[0].height*DIGEST_SIZE * sizeof(bb31_t), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_current_layer));
    return 0;
}