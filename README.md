# SP1

![SP1](https://github.com/succinctlabs/sp1/blob/dev/README.md)

This modified version of SP1(based on SP1 v4.2.0) is the fastest SP1, which uses the latest Plonky3(commit 5c04950709d20c8d88c3e64d37c935d7d759a01c).

> [!NOTE]
> The CUDA acceleration in SP1 is only partially open, so performance comparison is currently not possible. 

# Usage

## Clone the customized Plonky3 repository

```sh
# mkdir fastest-sp1
# cd fastest-sp1
# git clone https://github.com/fastest-sp1/Plonky3.git
```

## Clone the customized SP1 repository

```sh
# cd fastest-sp1
# git clone https://github.com/fastest-sp1/sp1.git
```

## Build the Groth16 circuit

1. Generate the build_groth16_bn254
```sh
# cd fastest-sp1/sp1/crates/prover
# RUSTFLAGS="-C target-feature=+avx2" cargo build -r
```
> [!NOTE]
> Modified the "target-feature" according your CPU feature.  
> Plonky3 only supports avx2 or avx512.  
> Use this command for cross-compilation: RUSTFLAGS="-C target-feature=+avx512f" cargo build --release --target x86_64-unknown-linux-gnu  


2. Run the build_groth16_bn254
> [!NOTE]
> The machine need free memory 30G.

```sh
# mkdir -p ~/.sp1/circuits/groth16/v4.2.0-new-p3/
# cd fastest-sp1/sp1/target/release
# export SP1_ALLOW_DEPRECATED_HOOKS=true
# RUST_LOG=debug nohup ./build_groth16_bn254 --build-dir ~/.sp1/circuits/groth16/v4.2.0-new-p3  > ./gen-groth16-key.log 2>&1 &
```
The generation will last about 30 minutes according your cpu speed. If ok, the log's tail is similar to :

```sh
16:44:38 INF compiling circuit
16:44:38 INF parsed circuit inputs nbPublic=2 nbSecret=24233
16:45:54 INF building constraint builder nbConstraints=8279429
16:48:29 DBG constraint system solver done nbConstraints=8279429 took=1683.562738
16:48:35 DBG prover done acceleration=none backend=groth16 curve=bn254 nbConstraints=8279429 took=5720.093624
16:48:35 DBG verifier done backend=groth16 curve=bn254 took=0.892739
16:48:35 DBG hash to field function not set, using keccak256 as default
```

and the build-dir has the files:

```sh
$ ls -l ~/.sp1/circuits/groth16/v4.2.0-new-p3/
total 4109340
-rw-r--r-- 1 gavin gavin      24977 Jun  1 17:41 Groth16Verifier.sol
-rw-r--r-- 1 gavin gavin       2481 Jun  1 17:41 SP1VerifierGroth16.sol
-rw-r--r-- 1 gavin gavin  172615564 Jun  1 17:41 constraints.json
-rw-r--r-- 1 gavin gavin 1169531596 Jun  1 17:41 groth16_circuit.bin
-rw-r--r-- 1 gavin gavin 2864539679 Jun  1 17:41 groth16_pk.bin
-rw-r--r-- 1 gavin gavin        396 Jun  1 17:41 groth16_vk.bin
-rw-r--r-- 1 gavin gavin     823105 Jun  1 17:41 groth16_witness.json
-rw-r--r-- 1 gavin gavin        648 Jun  1 17:41 wrap_vk.bin
-rw-r--r-- 1 gavin gavin     393926 Jun  1 17:41 wrapped_proof.bin
```

If failure, please check the machine's free memory which is at lease 30G.

## Build the Plonk circuit

1. Generate build_plonk_bn254  

   If generating build_groth16_bn254 ok, it will also generate build_plonk_bn254.
   
2.  Run the build_plonk_bn254

> [!NOTE]
> The machine need free memory 60G.

```sh
# mkdir -p ~/.sp1/circuits/plonk/v4.2.0-new-p3/
# cd fastest-sp1/sp1/target/release
# export SP1_ALLOW_DEPRECATED_HOOKS=true
# RUST_LOG=debug nohup ./build_plonk_bn254 --build-dir ~/.sp1/circuits/plonk/v4.2.0-new-p3  > ./gen-plonk-key.log 2>&1 &
```
The generation will last about 60 minutes according your cpu speed. If ok, the build-dir has the files :

```sh
$ ls -l ~/.sp1/circuits/plonk/v4.2.0-new-p3/
total 1610284
-rw-r--r-- 1 gavin gavin      58858 Jun  2 22:58 PlonkVerifier.sol
-rw-r--r-- 1 gavin gavin       2510 Jun  2 22:59 SP1VerifierPlonk.sol
-rw-r--r-- 1 gavin gavin  172615564 Jun  2 22:38 constraints.json
-rw-r--r-- 1 gavin gavin  401593333 Jun  2 22:58 plonk_circuit.bin
-rw-r--r-- 1 gavin gavin 1073776296 Jun  2 22:59 plonk_pk.bin
-rw-r--r-- 1 gavin gavin      34368 Jun  2 22:58 plonk_vk.bin
-rw-r--r-- 1 gavin gavin     823370 Jun  2 22:38 plonk_witness.json
```
If failure, please check the machine's free memory which is at lease 60G.

# Performance

## Hardware

CPU: Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz  
Memory: 20G Ram + 18G swap  
OS: ubuntu20  

> [!NOTE]
> Only compare CPU avx2 between the orginal SP1 v4.2 and this customed version.  
> Only run the three host programs in sp1/examples/fibonacci: fibonacci-script, groth16_bn254, plonk_bn254.

## fibonacci-script

|   Item     | Customed version   | SP1 v4.2.0 | Improvement (%) |
|------------|-------------------|------------|-----------------|
| execute    | 8.08ms            | 8.17ms     |
| prove_core | 32.0s             | 68.7s      | 53.4% faster    |
| compress   | 113s              | 208s       | 45.7% faster    |

## groth16_bn254

|   Item     | Customed version   | SP1 v4.2.0 | Improvement (%) |
|------------|-------------------|-------------|-----------------|
| prove_core | 34.0s             | 69.2s      |   53.4% faster    |
| compress   | 107s              | 215s       |   50.2% faster    |
| shrink     | 9.79s             | 18.3s      |   46.5% faster    |
| wrap_bn254 |  238s             | 362s       |   34.3% faster    |
| wrap_groth16_bn254 |  53.0s             | 53.6s       |        |


## plonk_bn254

|   Item     | Customed version   | SP1 v4.2.0 | Improvement (%) |
|------------|-------------------|-------------|-----------------|
| prove_core | 29.1s             | 63.6s      |   54.2% faster    |
| compress   | 114s              | 199s       |   42.7% faster    |
| shrink     | 8.44s             | 16.7s      |   49.5% faster    |
| wrap_bn254 |  242s             | 352s       |   31.2% faster    |
| wrap_plonk_bn254 |  361.0s             | 369s       |        |
