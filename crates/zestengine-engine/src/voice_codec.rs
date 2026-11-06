//! Voice codec: Opus (loaded from the system's libopus at runtime, so builds
//! need no C toolchain) with in-band forward error correction and packet-loss
//! concealment, falling back to raw 16-bit PCM when libopus isn't installed.
//! The first payload byte names the codec, so mixed peers still understand
//! each other.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

pub const RATE: i32 = 16_000;
/// Samples per frame (20 ms).
pub const FRAME: usize = 320;

const CODEC_PCM: u8 = 0;
const CODEC_OPUS: u8 = 1;
const OPUS_APPLICATION_VOIP: c_int = 2048;
const OPUS_SET_BITRATE: c_int = 4002;
const OPUS_SET_INBAND_FEC: c_int = 4012;
const OPUS_SET_PACKET_LOSS_PERC: c_int = 4014;

type CreateEnc = unsafe extern "C" fn(i32, c_int, c_int, *mut c_int) -> *mut c_void;
type Encode = unsafe extern "C" fn(*mut c_void, *const i16, c_int, *mut u8, i32) -> i32;
type EncCtl = unsafe extern "C" fn(*mut c_void, c_int, ...) -> c_int;
type CreateDec = unsafe extern "C" fn(i32, c_int, *mut c_int) -> *mut c_void;
type Decode = unsafe extern "C" fn(*mut c_void, *const u8, i32, *mut i16, c_int, c_int) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void);

struct Opus {
    _lib: libloading::Library,
    create_enc: CreateEnc,
    encode: Encode,
    enc_ctl: EncCtl,
    destroy_enc: Destroy,
    create_dec: CreateDec,
    decode: Decode,
    destroy_dec: Destroy,
}

fn opus() -> Option<&'static Opus> {
    static OPUS: OnceLock<Option<Opus>> = OnceLock::new();
    OPUS.get_or_init(|| {
        let names: &[&str] = if cfg!(windows) {
            &["opus.dll", "libopus-0.dll"]
        } else if cfg!(target_os = "macos") {
            &["libopus.0.dylib", "libopus.dylib"]
        } else {
            &["libopus.so.0", "libopus.so"]
        };
        let lib = names.iter().find_map(|n| unsafe { libloading::Library::new(n) }.ok());
        let Some(lib) = lib else {
            tracing::info!("voice: libopus not found, sending uncompressed PCM");
            return None;
        };
        // SAFETY: the symbols are libopus's documented C API with these signatures.
        let loaded = unsafe {
            Some(Opus {
                create_enc: *lib.get::<CreateEnc>(b"opus_encoder_create\0").ok()?,
                encode: *lib.get::<Encode>(b"opus_encode\0").ok()?,
                enc_ctl: *lib.get::<EncCtl>(b"opus_encoder_ctl\0").ok()?,
                destroy_enc: *lib.get::<Destroy>(b"opus_encoder_destroy\0").ok()?,
                create_dec: *lib.get::<CreateDec>(b"opus_decoder_create\0").ok()?,
                decode: *lib.get::<Decode>(b"opus_decode\0").ok()?,
                destroy_dec: *lib.get::<Destroy>(b"opus_decoder_destroy\0").ok()?,
                _lib: lib,
            })
        };
        if loaded.is_some() {
            tracing::info!("voice: Opus codec loaded");
        }
        loaded
    })
    .as_ref()
}

/// Whether frames are Opus-compressed on this machine.
#[cfg(test)]
pub fn opus_available() -> bool {
    opus().is_some()
}

pub struct Encoder {
    opus: Option<*mut c_void>,
}

// SAFETY: an encoder is only used by one thread at a time (owned, `&mut`).
unsafe impl Send for Encoder {}

impl Encoder {
    pub fn new(bitrate: u32) -> Self {
        let opus = opus().and_then(|o| {
            let mut err = 0;
            let st = unsafe { (o.create_enc)(RATE, 1, OPUS_APPLICATION_VOIP, &mut err) };
            if st.is_null() || err != 0 {
                return None;
            }
            unsafe {
                (o.enc_ctl)(st, OPUS_SET_BITRATE, bitrate.clamp(6_000, 64_000) as c_int);
                (o.enc_ctl)(st, OPUS_SET_INBAND_FEC, 1 as c_int);
                (o.enc_ctl)(st, OPUS_SET_PACKET_LOSS_PERC, 10 as c_int);
            }
            Some(st)
        });
        Self { opus }
    }

    /// One 20 ms frame to a payload.
    pub fn encode(&mut self, pcm: &[i16]) -> Vec<u8> {
        if let (Some(st), Some(o)) = (self.opus, opus())
            && pcm.len() == FRAME
        {
            let mut out = vec![0u8; 1 + 400];
            out[0] = CODEC_OPUS;
            let n = unsafe { (o.encode)(st, pcm.as_ptr(), FRAME as c_int, out[1..].as_mut_ptr(), 400) };
            if n > 0 {
                out.truncate(1 + n as usize);
                return out;
            }
        }
        let mut out = Vec::with_capacity(1 + pcm.len() * 2);
        out.push(CODEC_PCM);
        for s in pcm {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        if let (Some(st), Some(o)) = (self.opus.take(), opus()) {
            unsafe { (o.destroy_enc)(st) };
        }
    }
}

pub struct Decoder {
    opus: Option<*mut c_void>,
}

// SAFETY: a decoder is only used by one thread at a time (owned, `&mut`).
unsafe impl Send for Decoder {}

impl Default for Decoder {
    fn default() -> Self {
        let opus = opus().and_then(|o| {
            let mut err = 0;
            let st = unsafe { (o.create_dec)(RATE, 1, &mut err) };
            (!st.is_null() && err == 0).then_some(st)
        });
        Self { opus }
    }
}

impl Decoder {
    fn run(&mut self, data: Option<&[u8]>, fec: bool) -> Option<Vec<i16>> {
        let (st, o) = (self.opus?, opus()?);
        let mut pcm = vec![0i16; FRAME];
        let (ptr, len) = match data {
            Some(d) => (d.as_ptr(), d.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe { (o.decode)(st, ptr, len, pcm.as_mut_ptr(), FRAME as c_int, fec as c_int) };
        (n > 0).then(|| {
            pcm.truncate(n as usize);
            pcm.resize(FRAME, 0);
            pcm
        })
    }

    /// Decode a payload (None if it's malformed).
    pub fn decode(&mut self, payload: &[u8]) -> Option<Vec<i16>> {
        match payload.split_first()? {
            (&CODEC_OPUS, body) => self.run(Some(body), false),
            (&CODEC_PCM, body) => {
                let mut pcm: Vec<i16> = body.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
                pcm.resize(FRAME, 0);
                Some(pcm)
            }
            _ => None,
        }
    }

    /// Recover the frame before `next` from the forward-error-correction data
    /// Opus carries in it.
    pub fn recover(&mut self, next: &[u8]) -> Option<Vec<i16>> {
        match next.split_first()? {
            (&CODEC_OPUS, body) => self.run(Some(body), true),
            _ => None,
        }
    }

    /// A stand-in for a lost frame: Opus concealment, or a faded copy of
    /// silence for PCM.
    pub fn conceal(&mut self) -> Vec<i16> {
        self.run(None, false).unwrap_or_else(|| vec![0; FRAME])
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        if let (Some(st), Some(o)) = (self.opus.take(), opus()) {
            unsafe { (o.destroy_dec)(st) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone() -> Vec<i16> {
        (0..FRAME).map(|i| ((i as f32 * 0.2).sin() * 8000.0) as i16).collect()
    }

    #[test]
    fn round_trips_a_frame() {
        let mut enc = Encoder::new(24_000);
        let mut dec = Decoder::default();
        let mut last = Vec::new();
        for _ in 0..5 {
            let payload = enc.encode(&tone());
            if opus_available() {
                assert!(payload.len() < 200, "Opus compresses a 640-byte frame: {}", payload.len());
            }
            last = dec.decode(&payload).unwrap();
        }
        assert_eq!(last.len(), FRAME);
        let energy: f64 = last.iter().map(|s| (*s as f64).powi(2)).sum();
        assert!(energy > 1e6, "decoded audio isn't silent");
        assert_eq!(dec.conceal().len(), FRAME);
    }
}
