# SP1

![SP1](https://github.com/succinctlabs/sp1/blob/dev/README.md)

This modified version of SP1(based on SP1 v4.2.0) is the fastest SP1, which uses the latest Plonky3(commit 5c04950709d20c8d88c3e64d37c935d7d759a01c).

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

### Build the Groth16 circuit

1. generate the build_groth16_bn254
```sh
# cd fastest-sp1/sp1/crates/prover
# RUSTFLAGS="-C target-feature=+avx2" cargo build -r
```
> [!NOTE]
> Modified the "target-feature" according your CPU feature.
> Plonky3 only supports avx2 or avx512.
> Use this command for cross-compilation: RUSTFLAGS="-C target-feature=+avx512f" cargo build --release --target x86_64-unknown-linux-gnu

2. it will report the following similar error:
```sh
error: extern blocks must be unsafe
    --> /home/gavin/zkvm/fastest-sp1/sp1/target/release/build/sp1-recursion-gnark-ffi-e09b39f64bd06e95/out/bindings.rs:1485:1
     |
1485 | / extern "C" {
1486 | |     pub fn FreeString(s: *mut ::std::os::raw::c_char);
1487 | | }
     | |_^
```
please copy sp1/examples/fibonacci/bindings.rs to your target directory, such as sp1/target/release/build/sp1-recursion-gnark-ffi-e09b39f64bd06e95/out/.
then, rebuild it.

3. run the build_groth16_bn254
> [!NOTE]
> The machine need free memory 30G.

```sh
# mkdir -p ~/.sp1/circuits/groth16/v4.2.0-new-p3/
# cd fastest-sp1/sp1/target/release
# export SP1_ALLOW_DEPRECATED_HOOKS=true
# RUST_LOG=debug nohup ./build_groth16_bn254 --build-dir ./groth16  > ./gen-groth16-key.log 2>&1 &
```
The generation will last about 30 minutes according your cpu speed. If ok, the log is similar to :

```sh
16:44:38 INF compiling circuit
16:44:38 INF parsed circuit inputs nbPublic=2 nbSecret=24233
16:45:54 INF building constraint builder nbConstraints=8279429
16:48:29 DBG constraint system solver done nbConstraints=8279429 took=1683.562738
16:48:35 DBG prover done acceleration=none backend=groth16 curve=bn254 nbConstraints=8279429 took=5720.093624
16:48:35 DBG verifier done backend=groth16 curve=bn254 took=0.892739
16:48:35 DBG hash to field function not set, using keccak256 as default
```
if failure, please check the machine free memory which is at lease 30G.

### Build the Plonk circuit

