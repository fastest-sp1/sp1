#RUST_LOG=info  RUST_LOGGER=forest  nohup ../target/release/groth16_bn254  >./groth-test.log 2>&1 &
RUST_LOG=info  RUST_LOGGER=forest   nohup ../target/release/fibonacci-script  >./test-f.log 2>&1 &
