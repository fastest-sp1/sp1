//! A module for managing GPU memory with an arena allocator.

use crate::gpu::ffi::*;
use crate::gpu::matrix::{CudaResultCheck, GpuMatrix, CudaError};

use crossbeam_channel::{bounded, Receiver, Sender};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use parking_lot::Mutex;

#[derive(Debug, Default)]
struct GpuMemBlkState {
    pos: usize,
}

/// Owns a single, large GPU memory allocation and provides a bump allocator interface.
/// The `Drop` implementation ensures the entire arena is freed when it goes out of scope.
#[derive(Debug)]
pub struct GpuMemBlk {
    base_ptr: *mut c_void,
    capacity: usize,
    //pos: usize,
    state: Mutex<GpuMemBlkState>,
}


impl GpuMemBlk {
    /// Creates a new gpuMemBlk by allocating a large block of GPU memory.
    pub fn new(size_bytes: usize) -> Result<Self, CudaError> {
        let mut ptr = std::ptr::null_mut();
        if size_bytes > 0 {
            // println!("[GpuMemBlk] Allocating block of {:.2} GB", size_bytes as f64 / 1e9);
            unsafe {
                cuda_malloc(&mut ptr, size_bytes)
                    .check(&format!("GpuMemBlk::new failed to allocate {} bytes", size_bytes))?;
            }
        }
        Ok(Self {
            base_ptr: ptr,
            capacity: size_bytes,
            //state: Mutex::new(GpuMemBlkState { pos: 0 }),
            state: Mutex::new(GpuMemBlkState::default()),
        })
    }

    /// Allocates a `GpuMatrix` handle from the gpu mem block.
    ///
    /// This is a fast, host-side-only operation that bumps a pointer.
    /// It returns a non-owning `GpuMatrix` handle.
    ///
    /// # Panics
    /// Panics if the block is out of memory.
    pub fn alloc_matrix<T>(&self, height: usize, width: usize) -> GpuMatrix<T> {
        let mut state = self.state.lock();

        let align = std::mem::align_of::<T>().max(1); // Alignment must be at least 1
        let size = height.saturating_mul(width).saturating_mul(std::mem::size_of::<T>());
        
        // Calculate the next memory address that respects the required alignment.
        let aligned_pos = (state.pos + align - 1) & !(align - 1);
        
        if aligned_pos.saturating_add(size) > self.capacity {
            panic!(
                "GPU Block out of memory. Requested {} bytes (aligned to {}), but only {} bytes remaining of {}.H:{},W:{}",
                size, aligned_pos, self.capacity - state.pos, self.capacity, height, width
            );
        }

        let ptr = unsafe { self.base_ptr.add(aligned_pos) };
        //self.pos = aligned_pos + size;
         state.pos = aligned_pos + size;

        unsafe { GpuMatrix::from_raw_parts(ptr, width, height) }
    }

    pub fn alloc_matrix_from_vec<T>(&self, host_vec: &Vec<T>, height: usize, width: usize) -> GpuMatrix<T> {
        let gpu_mat = self.alloc_matrix::<T>(height, width);
        gpu_mat.copy_from_host(host_vec);
        gpu_mat
    }
    
    /// Resets the allocator's position to the beginning of the arena.
    /// This makes all memory in the arena available again for new allocations.
    /// It does not free or reallocate the underlying GPU buffer.
    pub fn reset(&mut self) {
        //self.pos = 0;
        let mut state = self.state.lock();
        state.pos = 0;
    }
}

impl Drop for GpuMemBlk {
    fn drop(&mut self) {
        if !self.base_ptr.is_null() {
             println!("[GpuMemBlk] Freeing block pointer {:p}", self.base_ptr);
            unsafe {
                let result = cuda_free(self.base_ptr);
                if result != 0 {
                    eprintln!(
                        "CRITICAL: cuda_free failed in GpuMemBlk::drop with code {}. GPU MEMORY LEAK.",
                        result
                    );
                }
            }
        }
    }
}

unsafe impl Send for GpuMemBlk {}
unsafe impl Sync for GpuMemBlk {}

/// A thread-safe pool of `GpuMemBlk` objects for managing concurrent GPU jobs.
#[derive(Debug)]
pub struct GpuMemBlkPool {
    sender: Sender<GpuMemBlk>,
    receiver: Receiver<GpuMemBlk>,
}

impl GpuMemBlkPool {
   /// Creates a new pool with a specified number of memory blocks, each of a given size.
    ///
    /// # Arguments
    /// * `num_blks` - The number of concurrent jobs this pool can support (e.g., 2 for 2 threads).
    /// * `size_per_blk` - The size in bytes for each individual memory block.
    ///
    /// # Panics
    /// Panics if `GpuMemBlk::new` fails, which would indicate a CUDA memory allocation error,
    /// or if `num_blks` is zero.
    pub fn new(num_blks: usize, size_per_blk: usize) -> Self {
        if num_blks == 0 {
            panic!("Cannot create a GpuMemBlkPool with zero memory blocks.");
        }
        
        // A bounded channel of size `num_blks` acts as our pool.
        let (sender, receiver) = bounded(num_blks);

        for i in 0..num_blks {
            // Provide informative logging during initialization.
            let size_mb = size_per_blk as f64 / (1024.0 * 1024.0);
            println!(
                "[GpuMemBlkPool] Initializing memory block {} with size {:.2} MB",
                i, size_mb
            );
            let blk = GpuMemBlk::new(size_per_blk)
                .expect("Failed to create GPU memory block for the pool");
            sender.send(blk).unwrap(); // Should not fail on a new channel
        }

        Self { sender, receiver }
    }

    /// Acquires a memory block from the pool, returning a lease.
    ///
    /// This function will block the current thread until a memory block becomes available.
    /// The returned `GpuMemBlkLease` is a smart pointer that automatically returns the block
    /// to the pool when it is dropped (goes out of scope).
    pub fn lease(&self) -> GpuMemBlkLease {
        // `recv()` will block if the channel (pool) is empty.
        let blk = self.receiver.recv().expect("GpuMemBlkPool channel was disconnected. This indicates a catastrophic failure.");
        
        GpuMemBlkLease {
            blk: Some(blk),
            sender: self.sender.clone(),
           // _phantom: PhantomData,
        }
    }
}


/// A smart pointer representing a temporary lease on a `GpuMemBlk` from a `GpuMemBlkPool`.
///
/// This struct implements `Deref` and `DerefMut`, allowing it to be used just like a
/// mutable reference to a `GpuMemBlk` (e.g., `lease.reset()`, `lease.alloc_matrix(...)`).
///
/// When an `GpuMemBlkLease` is dropped, its `Drop` implementation automatically and safely
/// returns the `GpuMemBlk` to the pool, making it available for other threads.
#[derive(Debug)]
pub struct GpuMemBlkLease {
    // We use Option to allow us to `take()` the block in the Drop impl.
    pub(super) blk: Option<GpuMemBlk>,
    // A cloned Sender to return the block to the pool.
    pub(super) sender: Sender<GpuMemBlk>,
    // A phantom marker to tie the lease's lifetime to the pool's lifetime.
    //pub(super) _phantom: PhantomData<&'a GpuMemBlkPool>,
}

impl Drop for GpuMemBlkLease {
    fn drop(&mut self) {
        // `take()` removes the GpuMemBlk from the Option, leaving `None`.
        if let Some(blk) = self.blk.take() {
            // Send the block back to the pool's channel.
            // `send` should not fail unless the receiver (the pool) has been dropped,
            // which would be a logic error in the program.
            self.sender.send(blk).expect("Failed to return GpuMemBlk to the pool. The pool may have been dropped while a lease was active.");
        }
    }
}

/// Allows calling `GpuMemBlk` methods directly on an `GpuMemBlkLease`.
/// Example: `let lease = pool.lease(); lease.reset();`
impl Deref for GpuMemBlkLease {
    type Target = GpuMemBlk;

    fn deref(&self) -> &Self::Target {
        // `unwrap` is safe because `blk` is only `None` after `drop` is called.
        self.blk.as_ref().unwrap()
    }
}

/// Allows calling mutable `GpuMemBlk` methods directly on an `GpuMemBlkLease`.
/// Example: `let mut lease = pool.lease(); lease.alloc_matrix(...);`
impl DerefMut for GpuMemBlkLease {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // `unwrap` is safe because `blk` is only `None` after `drop` is called.
        self.blk.as_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*; 
    use crate::gpu::matrix::GpuMatrix;
    use crate::baby_bear_poseidon2::{Val, Challenge};
    use std::thread;
    use std::sync::Arc;
    use std::time::Duration;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_gpuMemBlk_and_pool_lifecycle() {
        // Initialize the CUDA context for this test thread.
        //crate::gpu::init_gpu_context();
        // Pool with only ONE block to make the handoff explicit and test blocking.
        //let pool = Arc::new(GpuMemBlkPool::new(1, 16 * 1024 * 1024));
        let pool = Arc::new(GpuMemBlkPool::new(1, 1600 * 1024 * 1024));
        // Use a channel to pass the generated data from the producer to the consumer.
        // The message will contain the GpuMatrix AND the lease that keeps it alive.
        struct GpuTraceBundle {
            matrix: GpuMatrix<Val>,
            _lease: GpuMemBlkLease, // The lease is part of the bundle
        }
 

        let (sender, receiver) = std::sync::mpsc::channel::<GpuTraceBundle>();

        // --- PRODUCER THREAD (Trace Generation) ---
        let pool_clone = Arc::clone(&pool);
        let producer_handle = thread::spawn(move || {
            println!("[Producer] Leasing an arena...");
            let mut lease = pool_clone.lease();
            println!("[Producer] Leased arena. Pool size: {}", pool_clone.receiver.len());
            assert_eq!(pool_clone.receiver.len(), 0);
            lease.reset();

            // Generate data within the leased arena.
            let matrix = lease.alloc_matrix::<Val>(128, 128);
            let host_data: Vec<Val> = (0..128 * 128).map(|i| Val::from_usize(i)).collect();
            matrix.copy_from_host(&host_data);
            println!("[Producer] Generated matrix at ptr {:p}", matrix.ptr);
            
            // Bundle the matrix handle AND the lease together and send them.
            let bundle = GpuTraceBundle {
                matrix,
                _lease: lease, // Ownership of the lease is MOVED into the bundle
            };
            
            println!("[Producer] Sending bundle to consumer...");
            sender.send(bundle).unwrap();
            println!("[Producer] Bundle sent. Thread finishing.");
            // The lease is NOT dropped here. It now "lives" in the channel.
        });

        // --- CONSUMER THREAD (Proving) ---
        let pool_clone = Arc::clone(&pool);
        let consumer_handle = thread::spawn(move || {
            println!("[Consumer] Waiting to receive bundle...");
            // This will block until the producer sends the bundle.
            let received_bundle = receiver.recv().unwrap();
            println!("[Consumer] Received bundle with matrix at ptr {:p}", received_bundle.matrix.ptr);
            
            // At this point, the producer thread might have already finished, but that's okay.
            // The `received_bundle` now owns the `GpuMemBlkLease`, so the GPU memory is still valid.

            // Verify the data is still accessible and correct.
            let downloaded_data = received_bundle.matrix.to_host();
            assert_eq!(downloaded_data.len(), 128 * 128);
            assert_eq!(downloaded_data[100], Val::from_usize(100));
            println!("[Consumer] Data in received matrix is valid.");

            // The consumer could even try to lease another block. It will hang,
            // because the producer's lease is still held by `received_bundle`.
            // let another_lease_attempt = pool_clone.try_lease(); // Would return None
            // assert!(another_lease_attempt.is_none());

            println!("[Consumer] Finished work. Dropping bundle...");
            // `received_bundle` goes out of scope here. Its `_lease` field is dropped.
            // The `Drop` impl for `GpuMemBlkLease` runs, and the GpuMem is returned to the pool.
        });

        producer_handle.join().unwrap();
        consumer_handle.join().unwrap();
        
        // After both threads are done, the block should be back in the pool.
        thread::sleep(Duration::from_millis(20)); // Give channel time
        assert_eq!(pool.receiver.len(), 1);
        println!("\nSUCCESS: GpuMemBlk lease was correctly handed off and memory remained valid.");
    }
}