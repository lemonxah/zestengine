//! Records the compiler version for the plugin ABI stamp (`plugin_abi.rs`): a plugin cdylib shares
//! Rust-ABI types with its host, which is only sound when both were built by the same rustc.
fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    println!("cargo:rustc-env=ZESTENGINE_RUSTC_VERSION={}", version.trim());
    println!("cargo:rerun-if-changed=build.rs");
}
