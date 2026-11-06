//! Player identities: every install has a persistent Ed25519 key pair whose
//! public key is the player's id. A joining client signs the handshake
//! transcript with it, so the host learns a verified id per peer (shown as a
//! short fingerprint, matched by allow / ban lists, readable from gameplay),
//! and the host signs its WELCOME and pre-auth REJECTs the same way.
//!
//! The key lives in `identity.key` in the user's config folder, next to the
//! game's `input.ron` (`set_identity_dir` points it there). Hosts can
//! optionally be pinned on first use (`known_hosts`), so a later impostor
//! for the same address or room code is refused.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

use crate::net_secure::{hash, random_bytes};

/// An install's signing identity.
#[derive(Clone)]
pub struct Identity {
    key: SigningKey,
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Identity({})", fingerprint(&self.public()))
    }
}

static DIR: OnceLock<PathBuf> = OnceLock::new();
static INSTALL: Mutex<Option<Identity>> = Mutex::new(None);

/// Where `identity.key` and `known_hosts` live (call once at startup: the
/// folder of the game's `input.ron`).
pub fn set_identity_dir(dir: PathBuf) {
    let _ = DIR.set(dir);
}

fn identity_dir() -> Option<PathBuf> {
    if cfg!(test) || std::env::var_os("ZESTENGINE_NO_IDENTITY_FILE").is_some() {
        return None;
    }
    DIR.get().cloned()
}

impl Identity {
    /// A fresh random identity (tests, or no writable config folder).
    pub fn generate() -> Self {
        Self::from_seed(random_bytes())
    }

    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self { key: SigningKey::from_bytes(&seed) }
    }

    /// This install's identity: loaded from (or created in) the identity
    /// folder, else generated for this run.
    pub fn install() -> Self {
        let mut slot = INSTALL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(id) = slot.as_ref() {
            return id.clone();
        }
        let id = match identity_dir() {
            Some(dir) => Self::load_or_create(&dir.join("identity.key")).unwrap_or_else(|e| {
                tracing::warn!("net: identity file unavailable ({e}); using a temporary identity");
                Self::generate()
            }),
            None => Self::generate(),
        };
        *slot = Some(id.clone());
        id
    }

    /// Read a hex seed from `path`, or create one there (owner-only on Unix).
    pub fn load_or_create(path: &Path) -> std::io::Result<Self> {
        if let Ok(text) = std::fs::read_to_string(path)
            && let Some(seed) = from_hex::<32>(text.trim())
        {
            return Ok(Self::from_seed(seed));
        }
        let id = Self::generate();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, format!("{}\n", to_hex(&id.key.to_bytes())))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        tracing::info!("net: created player identity {} in {}", fingerprint(&id.public()), path.display());
        Ok(id)
    }

    /// The public key: the player id.
    pub fn public(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    pub fn sign(&self, msg: &[u8]) -> [u8; 64] {
        self.key.sign(msg).to_bytes()
    }

    pub fn fingerprint(&self) -> String {
        fingerprint(&self.public())
    }
}

/// Whether `sig` is `public`'s signature over `msg`.
pub fn verify(public: &[u8; 32], msg: &[u8], sig: &[u8; 64]) -> bool {
    VerifyingKey::from_bytes(public).is_ok_and(|k| k.verify(msg, &Signature::from_bytes(sig)).is_ok())
}

/// A player id's fingerprint: 128 bits of its SHA-256 as 8 groups of 4 hex digits.
pub fn fingerprint(public: &[u8; 32]) -> String {
    let h = hash(&[b"zestengine player id", public]);
    let hex = to_hex(&h[..16]);
    hex.as_bytes().chunks(4).map(|c| std::str::from_utf8(c).unwrap_or("")).collect::<Vec<_>>().join("-")
}

/// The ban-list key of an anti-cheat provider's account id: same shape as a player fingerprint
/// (so it lives in the same ban list), hashed under its own domain so the two can never collide.
pub fn provider_fingerprint(provider: &str, id: &str) -> String {
    let h = hash(&[b"zestengine provider id", provider.as_bytes(), &[0], id.as_bytes()]);
    let hex = to_hex(&h[..16]);
    hex.as_bytes().chunks(4).map(|c| std::str::from_utf8(c).unwrap_or("")).collect::<Vec<_>>().join("-")
}

/// The first two groups, for compact display.
pub fn short_fingerprint(fp: &str) -> &str {
    fp.get(..9).unwrap_or(fp)
}

/// A fingerprint as typed (any case, with or without separators): hex only.
pub fn normalize_fingerprint(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_hexdigit()).map(|c| c.to_ascii_lowercase()).collect()
}

/// Whether a list entry names this fingerprint (exact, separators ignored).
pub fn fingerprint_matches(entry: &str, fp: &str) -> bool {
    let e = normalize_fingerprint(entry);
    !e.is_empty() && e == normalize_fingerprint(fp)
}

/// Trust on first use: the first host identity seen for `target` (an
/// address or room code) is remembered; later a different one is refused.
/// Without an identity folder (tests) pins live in memory.
pub fn check_pin(target: &str, host: &[u8; 32]) -> Result<(), String> {
    let _guard = PIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let fp = fingerprint(host);
    let (path, mut pins) = read_pins();
    if let Some((_, known)) = pins.iter().find(|(t, _)| t == target) {
        return if fingerprint_matches(known, &fp) {
            Ok(())
        } else {
            Err(format!(
                "the server's identity key changed for {target} (pinned {}, now {}): refusing a possible impostor. If the server was reinstalled, forget the saved key for {target} (delete its line from known_hosts, or call NetSession::forget_host_pin) and join again",
                short_fingerprint(known),
                short_fingerprint(&fp)
            ))
        };
    }
    pins.push((target.to_owned(), fp));
    match &path {
        Some(p) => {
            let text: String = pins.iter().map(|(t, f)| format!("{t} {f}\n")).collect();
            if let Err(e) = std::fs::write(p, text) {
                tracing::warn!("net: could not save the host pin: {e}");
            }
        }
        None => *memory_pins().lock().unwrap_or_else(|e| e.into_inner()) = pins,
    }
    Ok(())
}

/// Serialises pin reads and writes (a check is read, compare, write).
static PIN_LOCK: Mutex<()> = Mutex::new(());

fn read_pins() -> (Option<PathBuf>, Vec<(String, String)>) {
    let path = identity_dir().map(|d| d.join("known_hosts"));
    let pins = match &path {
        Some(p) => std::fs::read_to_string(p)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| l.split_once(' ').map(|(t, f)| (t.to_owned(), f.trim().to_owned())))
            .collect(),
        None => memory_pins().lock().unwrap_or_else(|e| e.into_inner()).clone(),
    };
    (path, pins)
}

fn memory_pins() -> &'static Mutex<Vec<(String, String)>> {
    static MEMORY: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
    &MEMORY
}

/// Whether a host identity is pinned for `target` already.
pub fn is_pinned(target: &str) -> bool {
    read_pins().1.iter().any(|(t, _)| t == target)
}

/// Forget the pinned identity for `target` (an address, or `room:CODE@relay`)
/// so the next join pins whatever key the host has now. Returns whether there was one.
pub fn forget_pin(target: &str) -> bool {
    let _guard = PIN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (path, mut pins) = read_pins();
    let before = pins.len();
    pins.retain(|(t, _)| t != target);
    if pins.len() == before {
        return false;
    }
    match &path {
        Some(p) => {
            let text: String = pins.iter().map(|(t, f)| format!("{t} {f}\n")).collect();
            if let Err(e) = std::fs::write(p, text) {
                tracing::warn!("net: could not save the host pins: {e}");
            }
        }
        None => *memory_pins().lock().unwrap_or_else(|e| e.into_inner()) = pins,
    }
    true
}

pub fn to_hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}

pub fn from_hex<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != N * 2 {
        return None;
    }
    let mut out = [0u8; N];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_fingerprints_and_files() {
        let a = Identity::generate();
        let b = Identity::generate();
        let sig = a.sign(b"transcript");
        assert!(verify(&a.public(), b"transcript", &sig));
        assert!(!verify(&b.public(), b"transcript", &sig));
        assert!(!verify(&a.public(), b"transcripT", &sig));
        let fp = a.fingerprint();
        assert_eq!(fp.len(), 39);
        assert_ne!(fp, b.fingerprint());
        assert!(fingerprint_matches(&fp.to_uppercase().replace('-', " "), &fp));
        assert!(!fingerprint_matches(short_fingerprint(&fp), &fp), "only full fingerprints match");
        let dir = std::env::temp_dir().join(format!("zestengine-identity-{}", std::process::id()));
        let path = dir.join("identity.key");
        let first = Identity::load_or_create(&path).unwrap();
        let again = Identity::load_or_create(&path).unwrap();
        assert_eq!(first.public(), again.public(), "persistent");
        let _ = std::fs::remove_dir_all(&dir);
        // Trust on first use.
        assert!(check_pin("room:PINTST", &a.public()).is_ok());
        assert!(check_pin("room:PINTST", &a.public()).is_ok());
        assert!(check_pin("room:PINTST", &b.public()).is_err());
    }
}
