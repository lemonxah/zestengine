// The vendored astcenc (built below) references the C++ runtime (e.g.
// __gxx_personality_v0), so pull the runtime in on unix. Android's system
// libstdc++.so lacks the NDK libc++ symbols it needs (std::__ndk1,
// __cxa_throw), so the shared library would fail to load on device: link the
// NDK's static libc++ instead (nothing extra to ship).
// `-bundle` defers the lookup to the final link, where the NDK clang finds
// the libraries in its sysroot.
fn main() {
    build_astcenc();
    let os = std::env::var("CARGO_CFG_TARGET_OS");
    if os.as_deref() == Ok("android") {
        println!("cargo:rustc-link-lib=static:-bundle=c++_static");
        println!("cargo:rustc-link-lib=static:-bundle=c++abi");
    } else if os.as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// ARM astcenc (third_party/astcenc, Apache-2.0) for every ASTC encode, with
/// SSE4.1 on x86_64, NEON on aarch64 and scalar code elsewhere. The C++
/// runtime is linked by `main` above, not by cc.
fn build_astcenc() {
    let dir = std::path::Path::new("third_party");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").warnings(false).include(dir).cpp_link_stdlib(None);
    let (sse, popcnt, neon) = match arch.as_str() {
        "x86_64" => ("41", "1", "0"),
        "aarch64" => ("0", "0", "1"),
        _ => ("0", "0", "0"),
    };
    for (define, value) in [("ASTCENC_SSE", sse), ("ASTCENC_POPCNT", popcnt), ("ASTCENC_NEON", neon), ("ASTCENC_AVX", "0"), ("ASTCENC_F16C", "0")] {
        build.define(define, value);
    }
    if arch == "x86_64" {
        build.flag_if_supported("-msse4.1").flag_if_supported("-mpopcnt");
    }
    for entry in std::fs::read_dir(dir.join("astcenc")).expect("third_party/astcenc") {
        let path = entry.expect("astcenc entry").path();
        if path.extension().is_some_and(|e| e == "cpp") {
            build.file(&path);
        }
    }
    build.file(dir.join("zestengine_astc.cpp"));
    build.compile("zestengine_astcenc");
    println!("cargo:rerun-if-changed=third_party");
}
