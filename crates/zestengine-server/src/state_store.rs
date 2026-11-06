//! A crash-safe file for server state (NET-6): checksummed, versioned, written
//! atomically, with rotating backups.
//!
//! ```text
//! ZESTSTATE 1 <schema version> <payload bytes> <crc32 hex>\n
//! <payload>
//! ```
//!
//! A save writes `<name>.tmp`, syncs it, shifts `<name>` to `<name>.bak1`
//! (`bak1` to `bak2` ...), moves the temp file over `<name>` and syncs the
//! folder, so the newest *complete* file is always on disk: a crash at any
//! point leaves either the old generation or the new one, never half of one.
//! A load takes the newest generation whose length and checksum match, and
//! quarantines (`.corrupt-<unix>`) the ones that do not, so a damaged file is
//! kept for inspection and never rotated over a good backup.

use std::fs::{self, File};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

const MAGIC: &str = "ZESTSTATE";
const CONTAINER: u32 = 1;

/// CRC-32 (IEEE), table built at compile time.
fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                k += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    };
    !data.iter().fold(!0u32, |c, b| TABLE[((c ^ u32::from(*b)) & 0xFF) as usize] ^ (c >> 8))
}

/// What [`Store::load`] found.
#[derive(Debug, PartialEq, Eq)]
pub enum Loaded {
    /// No state file of any generation: a fresh start.
    Missing,
    /// The newest intact generation. `generation` 0 is the current file, 1 is `.bak1` ...
    /// `damaged` lists the newer files that were unusable (and were set aside).
    Found { version: u32, payload: Vec<u8>, generation: u32, damaged: Vec<String> },
    /// Files exist but none is usable (all set aside).
    Unusable { damaged: Vec<String> },
}

pub struct Store {
    dir: PathBuf,
    name: String,
    keep: u32,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Check the container of `bytes`; returns `(schema version, payload)`.
fn decode(bytes: &[u8]) -> Result<(u32, &[u8]), String> {
    let nl = bytes.iter().position(|b| *b == b'\n').ok_or("no header")?;
    let header = std::str::from_utf8(&bytes[..nl]).map_err(|_| "header is not text")?;
    let mut it = header.split(' ');
    if it.next() != Some(MAGIC) {
        return Err("not a state file".into());
    }
    let mut num = |what: &str| it.next().and_then(|v| v.parse::<u64>().ok()).ok_or(format!("bad {what} in the header"));
    let container = num("container version")?;
    if container != u64::from(CONTAINER) {
        return Err(format!("container version {container} is newer than this server understands"));
    }
    let version = num("schema version")? as u32;
    let len = num("length")? as usize;
    let crc = it.next().and_then(|v| u32::from_str_radix(v, 16).ok()).ok_or("bad checksum in the header")?;
    let payload = &bytes[nl + 1..];
    if payload.len() != len {
        return Err(format!("truncated or extended: header says {len} bytes, file has {}", payload.len()));
    }
    if crc32(payload) != crc {
        return Err("checksum mismatch".into());
    }
    Ok((version, payload))
}

impl Store {
    /// `dir/name` plus `keep` backup generations.
    pub fn new(dir: &Path, name: &str, keep: u32) -> Self {
        Self { dir: dir.to_path_buf(), name: name.to_owned(), keep: keep.min(20) }
    }

    pub fn path(&self, generation: u32) -> PathBuf {
        if generation == 0 { self.dir.join(&self.name) } else { self.dir.join(format!("{}.bak{generation}", self.name)) }
    }

    fn tmp(&self) -> PathBuf {
        self.dir.join(format!("{}.tmp", self.name))
    }

    /// Write `payload` (schema `version`) as the new current generation.
    pub fn save(&self, version: u32, payload: &[u8]) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let tmp = self.tmp();
        {
            let mut f = File::create(&tmp)?;
            f.write_all(format!("{MAGIC} {CONTAINER} {version} {} {:08x}\n", payload.len(), crc32(payload)).as_bytes())?;
            f.write_all(payload)?;
            f.sync_all()?;
        }
        // Only an intact current file is worth keeping as a backup.
        let current = self.path(0);
        if current.exists() {
            if fs::read(&current).is_ok_and(|b| decode(&b).is_ok()) {
                if self.keep > 0 {
                    for g in (1..self.keep).rev() {
                        if self.path(g).exists() {
                            fs::rename(self.path(g), self.path(g + 1))?;
                        }
                    }
                    fs::rename(&current, self.path(1))?;
                }
            } else {
                self.set_aside(&current);
            }
        }
        fs::rename(&tmp, &current)?;
        #[cfg(unix)]
        File::open(&self.dir).and_then(|d| d.sync_all())?;
        Ok(())
    }

    fn set_aside(&self, path: &Path) -> String {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let aside = self.dir.join(format!("{name}.corrupt-{}", now_unix()));
        match fs::rename(path, &aside) {
            Ok(()) => aside.display().to_string(),
            Err(_) => {
                let _ = fs::remove_file(path);
                format!("{} (removed)", path.display())
            }
        }
    }

    /// Newest intact generation. Unusable files are set aside as `.corrupt-<unix>`.
    pub fn load(&self) -> Loaded {
        let _ = fs::remove_file(self.tmp());
        let mut damaged = Vec::new();
        let mut any = false;
        for g in 0..=self.keep {
            let path = self.path(g);
            let Ok(bytes) = fs::read(&path) else { continue };
            any = true;
            match decode(&bytes) {
                Ok((version, payload)) => return Loaded::Found { version, payload: payload.to_vec(), generation: g, damaged },
                Err(why) => {
                    let moved = self.set_aside(&path);
                    damaged.push(format!("{}: {why} (kept as {moved})", path.display()));
                }
            }
        }
        if any { Loaded::Unusable { damaged } } else { Loaded::Missing }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zest-store-{tag}-{}-{}", std::process::id(), now_unix()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn crc_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn round_trip_and_backup_rotation() {
        let d = dir("rot");
        let s = Store::new(&d, "world.state", 2);
        assert_eq!(s.load(), Loaded::Missing);
        for n in 1..=4u8 {
            s.save(3, format!("gen {n}").as_bytes()).unwrap();
        }
        assert!(matches!(s.load(), Loaded::Found { version: 3, ref payload, generation: 0, .. } if payload == b"gen 4"));
        assert!(fs::read(s.path(1)).unwrap().ends_with(b"gen 3"));
        assert!(fs::read(s.path(2)).unwrap().ends_with(b"gen 2"));
        assert!(!s.path(3).exists(), "only `keep` backups");
        assert!(!s.tmp().exists());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn truncated_flipped_and_empty_files_fall_back_to_the_previous_generation() {
        for damage in ["truncate", "flip", "empty", "garbage", "header-only", "extend"] {
            let d = dir(damage);
            let s = Store::new(&d, "p.state", 2);
            s.save(1, b"old state, still good").unwrap();
            s.save(1, b"new state which gets damaged").unwrap();
            let mut bytes = fs::read(s.path(0)).unwrap();
            match damage {
                "truncate" => bytes.truncate(bytes.len() - 6),
                "flip" => *bytes.last_mut().unwrap() ^= 0x20,
                "empty" => bytes.clear(),
                "garbage" => bytes = b"\x00\xff not a state file".to_vec(),
                "header-only" => bytes.truncate(bytes.iter().position(|b| *b == b'\n').unwrap() + 1),
                _ => bytes.extend_from_slice(b"junk"),
            }
            fs::write(s.path(0), bytes).unwrap();
            let Loaded::Found { payload, generation, damaged, .. } = s.load() else { panic!("{damage}: expected the backup") };
            assert_eq!((payload.as_slice(), generation), (&b"old state, still good"[..], 1), "{damage}");
            assert_eq!(damaged.len(), 1, "{damage}");
            assert!(fs::read_dir(&d).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().contains(".corrupt-")), "{damage}: kept for inspection");
            let _ = fs::remove_dir_all(&d);
        }
    }

    #[test]
    fn a_crash_between_the_rotation_and_the_final_rename_loses_nothing() {
        let d = dir("crash");
        let s = Store::new(&d, "w.state", 2);
        s.save(1, b"one").unwrap();
        s.save(1, b"two").unwrap();
        // Simulate: the new file was written, the old one rotated, power lost before the final rename.
        fs::write(s.tmp(), b"ZESTSTATE 1 1 5 00000000\nhalf").unwrap();
        fs::rename(s.path(0), s.path(1)).unwrap();
        assert!(!s.path(0).exists());
        let Loaded::Found { payload, generation, .. } = s.load() else { panic!() };
        assert_eq!((payload.as_slice(), generation), (&b"two"[..], 1));
        assert!(!s.tmp().exists(), "leftover temp file removed");
        // The next save starts from the recovered state.
        s.save(1, b"three").unwrap();
        assert!(matches!(s.load(), Loaded::Found { generation: 0, ref payload, .. } if payload == b"three"));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn every_generation_damaged_is_reported_and_a_damaged_file_is_not_rotated_over_a_backup() {
        let d = dir("all");
        let s = Store::new(&d, "x.state", 1);
        s.save(1, b"a").unwrap();
        s.save(1, b"b").unwrap();
        fs::write(s.path(0), b"junk").unwrap();
        // Saving over a damaged current file keeps the intact backup.
        s.save(1, b"c").unwrap();
        assert!(fs::read(s.path(1)).unwrap().ends_with(b"a"), "bak1 is still the last good file, not the junk");
        fs::write(s.path(0), b"junk").unwrap();
        fs::write(s.path(1), b"junk too").unwrap();
        let Loaded::Unusable { damaged } = s.load() else { panic!() };
        assert_eq!(damaged.len(), 2);
        assert_eq!(s.load(), Loaded::Missing, "both were set aside; the next start is fresh");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_newer_container_is_refused_not_overwritten_silently() {
        let d = dir("future");
        fs::create_dir_all(&d).unwrap();
        let s = Store::new(&d, "f.state", 1);
        fs::write(s.path(0), b"ZESTSTATE 9 1 2 00000000\nhi").unwrap();
        let Loaded::Unusable { damaged } = s.load() else { panic!() };
        assert!(damaged[0].contains("newer"), "{damaged:?}");
        let _ = fs::remove_dir_all(&d);
    }
}
