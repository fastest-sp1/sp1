#![no_main]
sp1_zkvm::entrypoint!(main);

use rand::Rng;

pub fn main() {
    let mut rng = rand::thread_rng();
    for _ in 0..16 {
        let num = rng.r#gen::<u64>();
        println!("{num}");
    }
}
