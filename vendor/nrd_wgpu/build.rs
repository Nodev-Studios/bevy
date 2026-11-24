//! Builds NRD from the SDK that `tools/fetch-sdks.sh nrd` fetched, the C++ shim over it, and this crate's two HLSL
//! passes. Everything it produces stays in `OUT_DIR`, except `Shaders/NRDConfig.hlsli`, which NRD's own CMake
//! project writes into the SDK folder.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-env-changed=NRD_SDK");
    println!("cargo:rerun-if-changed=shim");
    println!("cargo:rerun-if-changed=shaders");

    let sdk = PathBuf::from(env::var("NRD_SDK").expect(
        "NRD_SDK is not set. Run `eval \"$(tools/fetch-sdks.sh nrd)\"` from the root of the Bevy fork, then build again",
    ));
    assert!(
        sdk.join("Include/NRD.h").is_file(),
        "NRD_SDK ({}) does not hold the NRD SDK",
        sdk.display()
    );
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib_dir = out_dir.join("lib");
    let shader_dir = out_dir.join("nrd-shaders");

    // NRD's CMake project builds its shader compiler driver (ShaderMake) for the host, downloads a DXC binary,
    // compiles NRD's shaders to SPIR-V and embeds them in the library.
    let mut config = cmake::Config::new(&sdk);
    #[cfg(unix)]
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        // Cross-building for Windows. Two things in NRD's project assume a Windows host:
        // - ShaderMake is still built for this machine, as `ShaderMake`, but is then called by the target's name
        //   for an executable, `ShaderMake.exe`. Give it that name.
        let tool_dir = out_dir.join("build/_deps/shadermake-build/build");
        fs::create_dir_all(&tool_dir).unwrap();
        let alias = tool_dir.join("ShaderMake.exe");
        if alias.symlink_metadata().is_err() {
            std::os::unix::fs::symlink("ShaderMake", alias).unwrap();
        }
        // - its resource script includes `version.h`, and the file is `Version.h`: a different file on a
        //   case-sensitive file system. The folder the shaders are generated into is on the include path.
        fs::create_dir_all(&shader_dir).unwrap();
        let alias = shader_dir.join("version.h");
        if alias.symlink_metadata().is_err() {
            std::os::unix::fs::symlink(sdk.join("Resources/Version.h"), alias).unwrap();
        }
    }
    config
        // Always the release build: a debug build links the debug C runtime on Windows, which Rust never does.
        .profile("Release")
        .define("NRD_STATIC_LIBRARY", "ON")
        .define("NRD_NRI", "OFF")
        // The quad intrinsics need VK_KHR_compute_shader_derivatives, which wgpu does not enable.
        .define("NRD_SUPPORTS_QUAD_INTRINSICS", "OFF")
        // wgpu takes passthrough SPIR-V; the DirectX containers would also need FXC, which only exists on Windows.
        .define("NRD_EMBEDS_SPIRV_SHADERS", "ON")
        .define("NRD_EMBEDS_DXIL_SHADERS", "OFF")
        .define("NRD_EMBEDS_DXBC_SHADERS", "OFF")
        .define("SHADERMAKE_FIND_FXC", "OFF")
        .define("NRD_SHADERS_PATH", &shader_dir)
        .define("CMAKE_ARCHIVE_OUTPUT_DIRECTORY", &lib_dir)
        .define("CMAKE_LIBRARY_OUTPUT_DIRECTORY", &lib_dir)
        .define("CMAKE_RUNTIME_OUTPUT_DIRECTORY", &lib_dir)
        .build_target("NRD");
    if cc::Build::new().cpp(true).get_compiler().is_like_clang_cl() {
        // NRD's math library uses SSSE3 intrinsics. NRD's project enables them for GCC and Clang and relies on
        // MSVC not asking; clang-cl does ask.
        config.cxxflag("-mssse3");
    }
    let cmake_dir = config.build().join("build");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // Where a multi-configuration generator (Visual Studio) puts the libraries.
    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.join("Release").display()
    );

    // The shim goes first: a static library must come before the ones it needs.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .cpp_link_stdlib(None)
        .file(manifest_dir.join("shim/nrd_shim.cpp"))
        .include(sdk.join("Include"))
        .compile("nrd_shim");
    println!("cargo:rustc-link-lib=static=NRD");
    println!("cargo:rustc-link-lib=static=ShaderMakeBlob");
    if env::var("CARGO_CFG_TARGET_ENV").unwrap() != "msvc" {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // The pack and unpack passes are compiled with the DXC that NRD's build downloaded, against the SDK's own
    // `NRD.hlsli` (and the `NRDConfig.hlsli` the configure step wrote beside it).
    let dxc = dxc_path(&cmake_dir.join("CMakeCache.txt"));
    for name in ["pack", "unpack"] {
        let status = Command::new(&dxc)
            .args(["-T", "cs_6_0", "-E", "main", "-spirv"])
            .args(["-fspv-target-env=vulkan1.2", "-O3", "-WX"])
            .arg("-I")
            .arg(sdk.join("Shaders"))
            .arg("-Fo")
            .arg(out_dir.join(format!("{name}.spv")))
            .arg(manifest_dir.join(format!("shaders/{name}.cs.hlsl")))
            .status()
            .unwrap_or_else(|error| panic!("could not run {}: {error}", dxc.display()));
        assert!(status.success(), "DXC failed on shaders/{name}.cs.hlsl");
    }
}

/// The DXC binary NRD's build uses for SPIR-V, read from the cache of its CMake project.
fn dxc_path(cmake_cache: &Path) -> PathBuf {
    let cache = fs::read_to_string(cmake_cache).unwrap();
    for variable in ["SHADERMAKE_DXC_VK_PATH", "SHADERMAKE_DXC_PATH"] {
        let value = cache.lines().find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.split(':').next() == Some(variable)).then_some(value)
        });
        if let Some(value) = value
            && Path::new(value).is_file()
        {
            return PathBuf::from(value);
        }
    }
    panic!("NRD's CMake project did not find a DXC binary");
}
