//! The ABI stamp of a component plugin (NET-3).
//!
//! A plugin is a `cdylib` whose `zestengine_register(&mut ComponentRegistry)` takes Rust-ABI types.
//! That is sound only when the plugin and its host were built by the same rustc, from the same
//! engine version, with the same `debug_assertions` and pointer width. A plugin proves it by
//! exporting `zestengine_plugin_abi` (add `zestengine_core::export_plugin_abi!();` to the crate);
//! the host compares the stamp with its own before calling anything else. The stamp is plain
//! `#[repr(C)]` data returned by an `extern "C"` function, so reading it is safe even when the
//! other side is a different compiler.

/// Bump when the `Component` / `ComponentRegistry` contract changes incompatibly.
pub const PLUGIN_ABI_VERSION: u32 = 1;
const MAGIC: u32 = 0x5A50_4C47; // "ZPLG"

/// What a plugin or a host was built with.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PluginAbi {
    /// Magic number identifying a plugin ABI stamp.
    pub magic: u32,
    /// The ABI version.
    pub abi: u32,
    /// Core major version.
    pub core_major: u16,
    /// Core minor version.
    pub core_minor: u16,
    /// Core patch version.
    pub core_patch: u16,
    /// Built with debug assertions.
    pub debug_assertions: u8,
    /// Pointer width in bits (64).
    pub pointer_width: u8,
    /// FNV-1a of `rustc --version` at build time.
    pub rustc_hash: u64,
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

impl PluginAbi {
    /// The stamp of the code being compiled right now (inside a plugin: the plugin's copy of core).
    pub fn current() -> Self {
        let mut v = env!("CARGO_PKG_VERSION").split('.').map(|p| p.parse::<u16>().unwrap_or(0));
        Self {
            magic: MAGIC,
            abi: PLUGIN_ABI_VERSION,
            core_major: v.next().unwrap_or(0),
            core_minor: v.next().unwrap_or(0),
            core_patch: v.next().unwrap_or(0),
            debug_assertions: cfg!(debug_assertions) as u8,
            pointer_width: usize::BITS as u8,
            rustc_hash: fnv(env!("ZESTENGINE_RUSTC_VERSION")),
        }
    }

    /// `Ok` when a plugin with stamp `self` may be loaded by `host`; otherwise a sentence naming
    /// the first mismatch and what to do about it.
    pub fn check_against(&self, host: &PluginAbi) -> Result<(), String> {
        if self.magic != MAGIC {
            return Err("the plugin's `zestengine_plugin_abi` export is not a ZestEngine ABI stamp".into());
        }
        if self.abi != host.abi {
            return Err(format!("plugin ABI version {} but this build expects {}: rebuild the plugin against this engine", self.abi, host.abi));
        }
        if (self.core_major, self.core_minor, self.core_patch) != (host.core_major, host.core_minor, host.core_patch) {
            return Err(format!(
                "plugin built against zestengine-core {}.{}.{} but this build is {}.{}.{}: rebuild the plugin",
                self.core_major, self.core_minor, self.core_patch, host.core_major, host.core_minor, host.core_patch
            ));
        }
        if self.rustc_hash != host.rustc_hash {
            return Err("plugin built with a different rustc than this build: Rust-ABI types are not compatible, rebuild the plugin with the same toolchain".into());
        }
        if self.debug_assertions != host.debug_assertions {
            let mode = |d: u8| if d != 0 { "debug" } else { "release" };
            return Err(format!("plugin built in {} mode but this build is {}: build the plugin with the same profile", mode(self.debug_assertions), mode(host.debug_assertions)));
        }
        if self.pointer_width != host.pointer_width {
            return Err(format!("plugin is {}-bit but this build is {}-bit", self.pointer_width, host.pointer_width));
        }
        Ok(())
    }
}

/// Export the ABI stamp from a plugin crate: `zestengine_core::export_plugin_abi!();`
#[macro_export]
macro_rules! export_plugin_abi {
    () => {
        #[unsafe(no_mangle)]
        pub extern "C" fn zestengine_plugin_abi() -> $crate::PluginAbi {
            $crate::PluginAbi::current()
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matching_stamp_passes_and_each_mismatch_is_named() {
        let host = PluginAbi::current();
        assert!(host.check_against(&host).is_ok());
        let bad = |f: &dyn Fn(&mut PluginAbi)| {
            let mut p = host;
            f(&mut p);
            p.check_against(&host).unwrap_err()
        };
        assert!(bad(&|p| p.magic = 1).contains("not a ZestEngine ABI stamp"));
        assert!(bad(&|p| p.abi += 1).contains("ABI version"));
        assert!(bad(&|p| p.core_minor += 1).contains("zestengine-core"));
        assert!(bad(&|p| p.rustc_hash ^= 1).contains("rustc"));
        assert!(bad(&|p| p.debug_assertions ^= 1).contains("mode"));
        assert!(bad(&|p| p.pointer_width = 32).contains("bit"));
    }
}
