// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.
#[macro_use]
extern crate failure;

use std::env;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::borrow::{Cow, Borrow};
use std::process::Command;
use std::os::unix::fs::PermissionsExt;

use cmake::Config;
use curl::easy::Easy;
use failure::Fallible;
use zip;

const TORCH_VERSION: &'static str = "1.3.1";

fn find_executable<P: AsRef<Path>>(name: P) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .filter_map(|dir| {
                let full_path = dir.join(&name);
                if full_path.is_file()
                    && full_path
                        .metadata()
                        .map(|m| m.permissions().mode() & 0o111 != 0)
                        .is_ok()
                {
                    Some(full_path)
                } else {
                    None
                }
            })
            .next()
    })
}

fn find_cuda_root() -> Option<PathBuf> {
    find_executable("nvcc")
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .or_else(|| {
            let p = Path::new("/usr").join("local").join("cuda");
            if p.is_dir() {
                return Some(p);
            }
            let p = Path::new("/opt").join("cuda");
            if p.is_dir() {
                return Some(p);
            }
            None
        })
        .map(|x| x.to_path_buf())
}

/// Given the full filename for a library (e.g., "libc.a" or "libc.so"), this
/// asks the compiler to find the library and, if found, will add it to the
/// cargo search path and then return `true`. If no such library is found by
/// the compiler, `false` is returned.
fn add_search_directory_for_library(name: &str) -> bool {
    if let Some(path) = find_library(name) {
        if let Some(path) = path.parent() {
            println!("cargo:rustc-link-search={}", path.display());
            return true;
        }
    }
    false
}

fn find_library(name: &str) -> Option<PathBuf> {
    let output = Command::new(
            cc::Build::new().get_compiler().path()
        )
        .arg(&format!("-print-file-name={}", name))
        .output();

    match output {
        Ok(output) => {
            if !output.status.success() {
                None
            } else if let Ok(path) = ::std::str::from_utf8(&output.stdout) {
                let path = path.trim();
                if path == name {
                    None
                } else {
                    Some(Path::new(path).to_path_buf())
                }
            } else {
                None
            }
        }
        Err(_) => None
    }
}

fn download<P: AsRef<Path>>(source_url: &str, target_file: P) -> Fallible<()> {
    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    let mut easy = Easy::new();
    easy.url(&source_url)?;
    easy.write_function(move |data| Ok(writer.write(data).unwrap()))?;
    easy.perform()?;
    let response_code = easy.response_code()?;
    if response_code == 200 {
        Ok(())
    } else {
        Err(format_err!(
            "Unexpected response code {} for {}",
            response_code,
            source_url
        ))
    }
}

fn extract<P: AsRef<Path>>(filename: P, outpath: P) -> Fallible<()> {
    let file = fs::File::open(&filename)?;
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = outpath.as_ref().join(file.sanitized_name());
        if !(&*file.name()).ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(&p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

fn prepare_libtorch_dir() -> PathBuf {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    let cuda_version = match env::var("TORCH_CUDA_VERSION") {
        Ok(cuda_env) => Some(Cow::Owned(cuda_env)),
        Err(_) => if cfg!(feature = "cuda101") {
                Some(Cow::Borrowed("cu101"))
            } else if cfg!(feature = "cuda92") {
                Some(Cow::Borrowed("cu92"))
            } else {
                None
            },
    };

    let device = match cuda_version.borrow() {
        Some(cuda_env) => match os.as_str() {
            "linux" => cuda_env
                .trim()
                .to_lowercase()
                .trim_start_matches("cu")
                .split(".")
                .take(2)
                .fold("cu".to_string(), |mut acc, curr| {
                    acc += curr;
                    acc
                }),
            os_str => panic!(
                "CUDA was specified with `TORCH_CUDA_VERSION`, but pre-built \
                 binaries with CUDA are only available for Linux, not: {}.",
                os_str
            ),
        },
        None => "cpu".to_string(),
    };

    if let Ok(libtorch) = env::var("LIBTORCH") {
        PathBuf::from(libtorch)
    } else {
        let libtorch_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("libtorch");
        if !libtorch_dir.exists() {
            fs::create_dir(&libtorch_dir).unwrap_or_default();
            let libtorch_url = match os.as_str() {
                "linux" => format!(
                    "https://download.pytorch.org/libtorch/{}/libtorch-cxx11-abi-shared-with-deps-{}{}.zip",
                    device, TORCH_VERSION, match device.as_ref() { "cpu" => "%2Bcpu", "cu92" => "%2Bcu92", _ => "" }
                ),
                "macos" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
                    TORCH_VERSION
                ),
                "windows" => format!(
                    "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-{}.zip",
                    TORCH_VERSION
                ),
                _ => panic!("Unsupported OS"),
            };

            let filename = libtorch_dir.join(format!("v{}.zip", TORCH_VERSION));
            download(&libtorch_url, &filename).unwrap();
            extract(&filename, &libtorch_dir).unwrap();
        }

        libtorch_dir.join("libtorch")
    }
}

fn make<P: AsRef<Path>>(libtorch: P) {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    match os.as_str() {
        "linux" | "macos" => {
            let libtorch_cxx11_abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or("1".to_string());
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch.as_ref().join("include"))
                .include(libtorch.as_ref().join("include/torch/csrc/api/include"))
                .flag(&format!(
                    "-Wl,-rpath={}",
                    libtorch.as_ref().join("lib").display()
                ))
                .flag("-std=c++11")
                .flag(&format!("-D_GLIBCXX_USE_CXX11_ABI={}", libtorch_cxx11_abi))
                .file("libtch/torch_api.cpp")
                .compile("libtorch");
        }
        "windows" => {
            // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
            //       Not yet supported by cc=rs.
            //       https://github.com/alexcrichton/cc-rs/issues/323
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch.as_ref().join("include"))
                .include(libtorch.as_ref().join("include/torch/csrc/api/include"))
                .file("libtch/torch_api.cpp")
                .compile("libtorch");
        }
        _ => panic!("Unsupported OS"),
    };
}

fn cmake<P: AsRef<Path>>(libtorch: P) {
    let dst = Config::new("libtch")
        .define("CMAKE_PREFIX_PATH", libtorch.as_ref())
        .build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=tch");
    println!("cargo:rustc-link-lib=stdc++");
}

fn main() {
    println!("cargo:rerun-if-env-changed=TORCH_CUDA_VERSION");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_CMAKE");

    let libtorch = prepare_libtorch_dir();
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch.join("lib").display()
    );

    if env::var("LIBTORCH_USE_CMAKE").is_ok() {
        cmake(&libtorch)
    } else {
        make(&libtorch)
    }

    let paths = fs::read_dir(libtorch.join("lib")).expect("Failed to enumerate libraries.");
    for path in paths {
        let entry = path.expect("Failed to parse path.");
        let path = entry.path();
        if path.is_file() {
            if path.extension().map(|ext| ext == "so" || ext == "a").unwrap_or(false) {
                if path.file_name().map(|n| n.to_str().unwrap().starts_with("lib")).unwrap_or(false) {
                    let p = path.file_stem().unwrap().to_str().unwrap();
                    println!("cargo:rustc-link-lib={}", p.split_at(3).1);
                }
            }
        }
    }

    //println!("cargo:rustc-link-lib=torch");
    //println!("cargo:rustc-link-lib=c10");

    let target = env::var("TARGET").unwrap();

    if !target.contains("msvc") && !target.contains("apple") {
        println!("cargo:rustc-link-lib=gomp");
    }

    if cfg!(feature = "cuda92") || cfg!(feature = "cuda101") {
        let cuda_root = find_cuda_root().expect("Failed to find CUDA root.");
        eprintln!("Detected CUDA root as: {}", cuda_root.display());

        println!("cargo:rustc-link-lib=cudart");
	println!("cargo:rustc-link-lib=cudart_static");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=culibos"); // NO _static
        println!("cargo:rustc-link-lib=nvToolsExt");

        println!("cargo:rustc-link-search={}", cuda_root.join("lib64").display());

        // CUDA 10.1 moves cuBLAS out of the CUDA toolkit directory and into
        // the system libraries instead. So we try to find the library and add
        // it to the Cargo search path. In the case of CUDA 10.1 and beyond,
        // this will let the linker find cuBLAS; for CUDA 10 and below, this
        // call will fail to find the library, but that's fine because we
        // already added the CUDA `lib64` directory just above.
        add_search_directory_for_library("libcublas.a");

        // CUDA 10.1 also splits cuBLAS into another library: cuBLASLt.
        if find_library("libcublasLt.a").is_some() {
            println!("cargo:rustc-link-lib=cublasLt");
        }
    } else {
        eprintln!("CUDA support disabled.");
    }
}
