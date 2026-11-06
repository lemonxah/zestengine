//! Audio mixer / buses (ENGINE_FEATURE_CHECKLIST T0 #14 completion).
//!
//! Named buses (Master / Music / SFX / Voice / …) each with a linear volume + mute,
//! arranged in a parent chain so a sound's effective gain is its bus volume times
//! every ancestor's (and Master's). The audio backend multiplies each playing
//! source by `effective_gain(bus)`. Pure logic, fully testable; the rodio sink just
//! reads the computed gain.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Bus {
    volume: f32,
    muted: bool,
    parent: Option<String>,
}

/// A small bus graph. "master" always exists and is the root. Saved in
/// `project.zestengine`; `AudioSource.bus` picks the bus a source plays on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioMixer {
    buses: HashMap<String, Bus>,
}

impl Default for AudioMixer {
    fn default() -> Self {
        let mut buses = HashMap::new();
        buses.insert("master".into(), Bus { volume: 1.0, muted: false, parent: None });
        // Common defaults, all routed to master.
        for name in ["music", "sfx", "voice", "ui"] {
            buses.insert(
                name.into(),
                Bus { volume: 1.0, muted: false, parent: Some("master".into()) },
            );
        }
        Self { buses }
    }
}

impl AudioMixer {
    /// The default mixer (master plus the common buses).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add / re-route a bus under `parent` (defaults to master). No-op cycle guard:
    /// a bus can't be its own ancestor.
    pub fn add_bus(&mut self, name: &str, parent: Option<&str>) {
        if name == "master" {
            return;
        }
        let parent = parent.map(str::to_string).unwrap_or_else(|| "master".into());
        if self.would_cycle(name, &parent) {
            return;
        }
        let entry = self
            .buses
            .entry(name.to_string())
            .or_insert_with(|| Bus { volume: 1.0, muted: false, parent: None });
        entry.parent = Some(parent);
    }

    fn would_cycle(&self, name: &str, parent: &str) -> bool {
        let mut anc = parent.to_string();
        loop {
            if anc == name {
                return true;
            }
            match self.buses.get(&anc).and_then(|b| b.parent.clone()) {
                Some(p) => anc = p,
                None => return false,
            }
        }
    }

    /// Set a bus's linear volume (clamped to 0+).
    pub fn set_volume(&mut self, bus: &str, volume: f32) {
        if let Some(b) = self.buses.get_mut(bus) {
            b.volume = volume.max(0.0);
        }
    }

    /// Mute or unmute a bus.
    pub fn set_muted(&mut self, bus: &str, muted: bool) {
        if let Some(b) = self.buses.get_mut(bus) {
            b.muted = muted;
        }
    }

    /// A bus's linear volume (1 for an unknown bus).
    pub fn volume(&self, bus: &str) -> f32 {
        self.buses.get(bus).map(|b| b.volume).unwrap_or(1.0)
    }
    /// Whether a bus is muted (false for an unknown bus).
    pub fn is_muted(&self, bus: &str) -> bool {
        self.buses.get(bus).map(|b| b.muted).unwrap_or(false)
    }

    /// Every bus name, sorted.
    pub fn bus_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.buses.keys().cloned().collect();
        v.sort();
        v
    }

    /// Effective linear gain for a sound playing on `bus`: the product of this bus
    /// and every ancestor's volume up to master. Any muted bus in the chain → 0.
    pub fn effective_gain(&self, bus: &str) -> f32 {
        let mut gain = 1.0;
        let mut cur = Some(bus.to_string());
        let mut guard = 0;
        while let Some(name) = cur {
            let Some(b) = self.buses.get(&name) else {
                break;
            };
            if b.muted {
                return 0.0;
            }
            gain *= b.volume;
            cur = b.parent.clone();
            guard += 1;
            if guard > 64 {
                break; // safety
            }
        }
        gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_gain_multiplies_chain() {
        let mut m = AudioMixer::new();
        m.set_volume("master", 0.5);
        m.set_volume("sfx", 0.8);
        // sfx -> master: 0.8 * 0.5 = 0.4
        assert!((m.effective_gain("sfx") - 0.4).abs() < 1e-6);
        // master alone
        assert!((m.effective_gain("master") - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mute_anywhere_in_chain_silences() {
        let mut m = AudioMixer::new();
        m.set_muted("master", true);
        assert_eq!(m.effective_gain("music"), 0.0);
        m.set_muted("master", false);
        m.set_muted("music", true);
        assert_eq!(m.effective_gain("music"), 0.0);
        assert!(m.effective_gain("sfx") > 0.0);
    }

    #[test]
    fn nested_buses_and_cycle_guard() {
        let mut m = AudioMixer::new();
        m.add_bus("weapons", Some("sfx"));
        m.set_volume("sfx", 0.5);
        m.set_volume("weapons", 0.5);
        m.set_volume("master", 1.0);
        // weapons -> sfx -> master = 0.5 * 0.5 * 1.0
        assert!((m.effective_gain("weapons") - 0.25).abs() < 1e-6);
        // Attempting to make sfx a child of weapons would cycle -> rejected.
        m.add_bus("sfx", Some("weapons"));
        assert!((m.effective_gain("weapons") - 0.25).abs() < 1e-6);
    }
}
