use pathdiff::diff_paths;
use std::env;
use std::fs;
use std::os;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // This is a common pattern to avoid issues when docs.rs tries to build the crate.
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // --- 1. Configuration and Path Setup ---

    // The name for our new static library for stark CUDA kernels.
    const LIB_NAME: &str = "sp1_stark_cuda";

    // Directory containing C++/CUDA source files for this crate.
    const SOURCE_DIRNAME: &str = "gpu_ffi";

    // Directory containing header files for this crate.
    const INCLUDE_DIRNAME: &str = "include";

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Find the `target` directory for IDE header symlinks.
    let target_dir = {
        let mut dir = out_dir.clone();
        loop {
            if dir.ends_with("target") {
                break dir;
            }
            if !dir.pop() {
                // Fallback for non-standard project structures.
                // You might see this warning if running tests from within IntelliJ/VSCode in a specific way.
                // It's generally safe to ignore if compilation proceeds.
                println!(
                    "cargo:warning=Could not find 'target' directory in OUT_DIR path. IDE header symlinks might not be created."
                );
                break out_dir.clone();
            }
        }
    };

    let source_include_dir = crate_dir.join(INCLUDE_DIRNAME);
    let target_include_dir = out_dir.join(INCLUDE_DIRNAME);
    let target_include_dir_fixed = target_dir.join(INCLUDE_DIRNAME);

    // --- 2. Header File Management ---

    // Find all header files.
    let headers = glob::glob(source_include_dir.join("**/*.h*").to_str().unwrap())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Copy headers to the output directory and create symlinks for the IDE.
    for header in &headers {
        let relpath = diff_paths(header, &source_include_dir).unwrap();
        let dst = target_include_dir.join(&relpath);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::copy(header, &dst).unwrap();
        rel_symlink_file(dst, target_include_dir_fixed.join(relpath));
    }

    let core_machine_include_path = crate_dir
        .parent() // Move from 'stark' up to 'crates'
        .unwrap()
        .join("core")
        .join("machine")
        .join("include");

    if !core_machine_include_path.exists() {
        panic!(
            "Could not find sp1-core-machine include directory at hardcoded path: {:?}",
            core_machine_include_path
        );
    }

    println!(
        "cargo:warning=Using hardcoded relative path for sp1-core-machine include: {}",
        core_machine_include_path.to_string_lossy()
    );

    let recursion_core_include_path = crate_dir
        .parent() // Move from 'stark' up to 'crates'
        .unwrap()
        .join("recursion")
        .join("core")
        .join("include");

    if !recursion_core_include_path.exists() {
        panic!(
            "Could not find sp1-core-machine include directory at hardcoded path: {:?}",
            recursion_core_include_path
        );
    }

    println!(
        "cargo:warning=Using hardcoded relative path for sp1-core-machine include: {}",
        recursion_core_include_path.to_string_lossy()
    );

    // --- 3. Source File Discovery ---

    // Find all CUDA source files in the `cpp` directory.
    let cuda_sources =
        glob::glob(crate_dir.join(SOURCE_DIRNAME).join("**/*.cpp").to_str().unwrap())
            .unwrap()
            .map(|p| p.unwrap())
            .collect::<Vec<_>>();

    // --- 4. Rerun Instructions for Cargo ---

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", INCLUDE_DIRNAME);
    println!("cargo:rerun-if-changed={}", SOURCE_DIRNAME);

    // Cargo build script metadata, used by dependents' build scripts.
    // The root directory containing the library archive.
    println!("cargo::metadata=root={}", out_dir.to_str().unwrap());

    if cuda_sources.is_empty() {
        println!(
            "cargo:warning=CUDA feature is enabled, but no .cu files were found in the '{}' directory.",
            SOURCE_DIRNAME
        );
        return;
    }

    // Determine GPU architecture. Default to sm_75 if not set.
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".into());

    // Common arguments for nvcc compilation.
    let cuda_common_args = &[
        "-c", // Compile only, do not link into an executable.
        &format!("-arch={}", arch),
        "--expt-relaxed-constexpr",
        // Include paths for our own headers and any dependency headers.
        &format!("-I{}", target_include_dir.to_str().unwrap()),
        &format!("-I{}", core_machine_include_path.to_str().unwrap()),
        &format!("-I{}", recursion_core_include_path.to_str().unwrap()),
        "-I /usr/local/cuda/include", //CUB lib
    ];

    let mut object_files = Vec::new();

    // Compile each CUDA source file into an object file.
    for cu_file in &cuda_sources {
        let obj_name = out_dir.join(cu_file.file_stem().unwrap()).with_extension("o");

        let output = Command::new("nvcc")
            .args(cuda_common_args)
            //.include("/usr/local/cuda/include") //CUB lib
            .arg("--x")
            .arg("cu") // Force nvcc to treat the file as .cu
            .arg("-o")
            .arg(obj_name.to_str().unwrap())
            .arg(cu_file.to_str().unwrap())
            .output()
            .expect("Failed to execute nvcc command");

        if !output.status.success() {
            eprintln!(
                "nvcc compilation stdout for {}:\n{}",
                cu_file.display(),
                String::from_utf8_lossy(&output.stdout)
            );
            eprintln!(
                "nvcc compilation stderr for {}:\n{}",
                cu_file.display(),
                String::from_utf8_lossy(&output.stderr)
            );
            panic!("nvcc compilation failed for {}", cu_file.display());
        }
        object_files.push(obj_name);
    }

    // --- 6. Archiving Object Files into a Static Library ---

    let lib_path = out_dir.join(format!("lib{}.a", LIB_NAME));

    // Use `ar` to create the static library. This is the standard Unix tool for this.
    // `nvcc -lib` is just a wrapper around `ar`. Using `ar` directly is more explicit.
    let mut ar_command = Command::new("ar");
    ar_command
        .arg("rcs") // r: insert with replacement, c: create if not exists, s: write index
        .arg(lib_path.to_str().unwrap());
    ar_command.args(object_files.iter().map(|p| p.to_str().unwrap()));

    let output = ar_command.output().expect("Failed to execute ar command");
    if !output.status.success() {
        eprintln!("ar stdout:\n{}", String::from_utf8_lossy(&output.stdout));
        eprintln!("ar stderr:\n{}", String::from_utf8_lossy(&output.stderr));
        panic!("Failed to create static library with ar.");
    }

    // --- Linker Instructions ---
    println!("cargo:rustc-link-search=native={}", out_dir.to_str().unwrap());
    println!("cargo:rustc-link-lib=static={}", LIB_NAME);

    // Find and link the CUDA runtime library.
    println!("cargo::rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo::rustc-link-search=native=/usr/local/cuda/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Link against the C++ standard library, which is often needed by nvcc-compiled code.
    //println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// Creates a relative symlink pointing to `original` at `link`.
/// This function is from your original script and is preserved.
fn rel_symlink_file<P, Q>(original: P, link: Q)
where
    P: AsRef<Path>,
    Q: AsRef<Path>,
{
    let target_dir = link.as_ref().parent().unwrap();
    if !target_dir.exists() {
        fs::create_dir_all(target_dir).unwrap();
    }

    // Remove existing link/file to avoid errors
    let _ = fs::remove_file(&link);

    // Use a try-catch for different OS symlink functions if needed
    if let Err(e) = os::unix::fs::symlink(diff_paths(&original, target_dir).unwrap(), &link) {
        // This might happen on Windows, or if permissions are wrong.
        // You could add a fallback to copy the file on non-unix systems.
        println!(
            "cargo:warning=Failed to create symlink from {:?} to {:?}: {}",
            original.as_ref(),
            link.as_ref(),
            e
        );
    }
}
