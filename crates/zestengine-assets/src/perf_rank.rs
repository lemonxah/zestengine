//! Performance ranks for avatars and other user content (GEO-2).
//!
//! A rank is the worst of per-metric ranks: triangles, materials, bones,
//! blendshapes and texture memory, each against a budget table for the content
//! kind and the device tier. Everything here is pure (no scene, no GPU) so the
//! editor, the package builder and the runtime safety settings share one
//! answer. Ranks come from *source* counts; the automatic screen-size LOD
//! makes triangles cheap at a distance, but memory, draws, bones and
//! blendshapes cost the same wherever the avatar stands, so they stay in.

use serde::{Deserialize, Serialize};
use zestengine_core::ContentKind;

use crate::package::PackageStats;

/// Best to worst.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PerfRank {
    #[default]
    Excellent,
    Good,
    Medium,
    Poor,
    VeryPoor,
}

impl PerfRank {
    pub const ALL: [PerfRank; 5] = [Self::Excellent, Self::Good, Self::Medium, Self::Poor, Self::VeryPoor];

    pub fn label(self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Medium => "Medium",
            Self::Poor => "Poor",
            Self::VeryPoor => "Very Poor",
        }
    }
}

/// The device class the budgets are for. Mobile = standalone VR / phones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerfTier {
    #[default]
    Desktop,
    Mobile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PerfMetric {
    Triangles,
    Materials,
    Bones,
    Blendshapes,
    TextureMemory,
}

impl PerfMetric {
    pub const ALL: [PerfMetric; 5] = [Self::Triangles, Self::Materials, Self::Bones, Self::Blendshapes, Self::TextureMemory];

    pub fn label(self) -> &'static str {
        match self {
            Self::Triangles => "Triangles",
            Self::Materials => "Materials",
            Self::Bones => "Bones",
            Self::Blendshapes => "Blendshapes",
            Self::TextureMemory => "Texture memory",
        }
    }
}

const MIB: u64 = 1024 * 1024;

/// What an object (or package) costs, measured from its content.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfStats {
    pub triangles: u64,
    pub materials: u32,
    pub bones: u32,
    pub blendshapes: u32,
    pub texture_bytes: u64,
}

impl PerfStats {
    pub fn value(&self, m: PerfMetric) -> u64 {
        match m {
            PerfMetric::Triangles => self.triangles,
            PerfMetric::Materials => self.materials as u64,
            PerfMetric::Bones => self.bones as u64,
            PerfMetric::Blendshapes => self.blendshapes as u64,
            PerfMetric::TextureMemory => self.texture_bytes,
        }
    }
}

impl From<&PackageStats> for PerfStats {
    fn from(s: &PackageStats) -> Self {
        Self { triangles: s.triangles, materials: s.materials, bones: s.bones, blendshapes: s.blendshapes, texture_bytes: s.texture_bytes }
    }
}

/// The largest value that still earns Excellent, Good, Medium and Poor; above
/// the last is Very Poor.
pub type Limits = [u64; 4];

/// Budget table for a content kind on a device tier.
pub fn limits(kind: ContentKind, tier: PerfTier, metric: PerfMetric) -> Limits {
    use PerfMetric::*;
    match (kind, tier) {
        (ContentKind::Avatar, PerfTier::Desktop) => match metric {
            Triangles => [32_000, 70_000, 110_000, 150_000],
            Materials => [4, 8, 16, 32],
            Bones => [150, 256, 400, 600],
            Blendshapes => [32, 64, 128, 256],
            TextureMemory => [40 * MIB, 75 * MIB, 110 * MIB, 150 * MIB],
        },
        (ContentKind::Avatar, PerfTier::Mobile) => match metric {
            Triangles => [7_500, 10_000, 15_000, 20_000],
            Materials => [1, 2, 4, 8],
            Bones => [75, 150, 200, 256],
            Blendshapes => [8, 16, 32, 64],
            TextureMemory => [10 * MIB, 18 * MIB, 25 * MIB, 40 * MIB],
        },
        (ContentKind::Prop, PerfTier::Desktop) => match metric {
            Triangles => [20_000, 50_000, 100_000, 250_000],
            Materials => [2, 4, 8, 16],
            Bones => [64, 128, 256, 512],
            Blendshapes => [8, 32, 64, 128],
            TextureMemory => [32 * MIB, 64 * MIB, 96 * MIB, 128 * MIB],
        },
        (ContentKind::Prop, PerfTier::Mobile) => match metric {
            Triangles => [5_000, 10_000, 20_000, 50_000],
            Materials => [1, 2, 4, 8],
            Bones => [32, 64, 128, 256],
            Blendshapes => [4, 8, 16, 32],
            TextureMemory => [8 * MIB, 16 * MIB, 32 * MIB, 64 * MIB],
        },
        (ContentKind::World, PerfTier::Desktop) => match metric {
            Triangles => [250_000, 1_000_000, 2_500_000, 5_000_000],
            Materials => [64, 256, 1024, 4096],
            Bones => [2_000, 5_000, 10_000, 20_000],
            Blendshapes => [256, 1024, 4096, 16_384],
            TextureMemory => [256 * MIB, 512 * MIB, 768 * MIB, 1024 * MIB],
        },
        (ContentKind::World, PerfTier::Mobile) => match metric {
            Triangles => [50_000, 150_000, 400_000, 1_000_000],
            Materials => [16, 48, 128, 512],
            Bones => [500, 1_500, 3_000, 6_000],
            Blendshapes => [64, 256, 1024, 4096],
            TextureMemory => [64 * MIB, 128 * MIB, 256 * MIB, 384 * MIB],
        },
    }
}

/// Rank of one value against a limits row.
pub fn rank_value(value: u64, limits: &Limits) -> PerfRank {
    PerfRank::ALL[limits.iter().position(|&l| value <= l).unwrap_or(4)]
}

/// One metric of a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetricReport {
    pub metric: PerfMetric,
    pub value: u64,
    pub rank: PerfRank,
    /// The largest value that would rank one step better (None when Excellent).
    pub next_better: Option<u64>,
}

/// An overall rank and the per-metric breakdown behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PerfReport {
    pub overall: PerfRank,
    pub metrics: Vec<MetricReport>,
}

impl PerfReport {
    /// The metric that sets the overall rank (the first on a tie).
    pub fn worst(&self) -> Option<&MetricReport> {
        self.metrics.iter().find(|m| m.rank == self.overall)
    }
}

pub fn rank(stats: &PerfStats, kind: ContentKind, tier: PerfTier) -> PerfReport {
    let metrics: Vec<MetricReport> = PerfMetric::ALL
        .iter()
        .map(|&metric| {
            let row = limits(kind, tier, metric);
            let value = stats.value(metric);
            let rank = rank_value(value, &row);
            let next_better = (rank != PerfRank::Excellent).then(|| row[rank as usize - 1]);
            MetricReport { metric, value, rank, next_better }
        })
        .collect();
    let overall = metrics.iter().map(|m| m.rank).max().unwrap_or_default();
    PerfReport { overall, metrics }
}

/// A player's safety settings for other people's avatars.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AvatarPolicy {
    /// Avatars ranked worse than this are replaced by the fallback.
    pub min_rank: PerfRank,
    /// At most this many avatars are shown at full detail, nearest first
    /// (0 = no limit).
    pub max_shown: u32,
    /// Avatars farther than this are not shown (0 = no limit).
    pub cull_distance: f32,
    /// Avatars farther than this (but inside `cull_distance`) are drawn as
    /// impostors instead of their full model (0 = never). An impostor is the
    /// rest pose turned towards the camera, so it is a stand-in for distance.
    pub impostor_distance: f32,
}

impl Default for AvatarPolicy {
    fn default() -> Self {
        Self { min_rank: PerfRank::VeryPoor, max_shown: 0, cull_distance: 0.0, impostor_distance: 0.0 }
    }
}

impl AvatarPolicy {
    /// Default for a device tier: standalone devices hide Very Poor avatars.
    pub fn for_tier(tier: PerfTier) -> Self {
        match tier {
            PerfTier::Desktop => Self::default(),
            PerfTier::Mobile => Self { min_rank: PerfRank::Poor, max_shown: 24, cull_distance: 0.0, impostor_distance: 0.0 },
        }
    }
}

/// What to draw for an avatar under a policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AvatarDisplay {
    Full,
    /// Replaced by the fallback (hidden until a fallback model is configured).
    Fallback(FallbackReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackReason {
    /// Its rank is worse than the policy's minimum.
    Rank(PerfRank),
    /// Past the avatar culling distance.
    Distance,
    /// More avatars are nearer than the limit allows.
    TooMany,
    /// Beyond the impostor distance: still shown, as an impostor.
    Far,
}

impl FallbackReason {
    pub fn label(self) -> String {
        match self {
            Self::Rank(r) => format!("rank {} is below the minimum", r.label()),
            Self::Distance => "past the avatar culling distance".to_string(),
            Self::TooMany => "over the shown-avatar limit".to_string(),
            Self::Far => "beyond the impostor distance".to_string(),
        }
    }
}

/// Decide each avatar's display. `avatars` are `(rank, distance)`; the result
/// is parallel. Nearest avatars win the `max_shown` slots, and avatars that
/// are already hidden by rank or distance do not use one.
pub fn decide_avatars(avatars: &[(PerfRank, f32)], policy: &AvatarPolicy) -> Vec<AvatarDisplay> {
    let mut out: Vec<AvatarDisplay> = avatars
        .iter()
        .map(|&(rank, distance)| {
            if rank > policy.min_rank {
                AvatarDisplay::Fallback(FallbackReason::Rank(rank))
            } else if policy.cull_distance > 0.0 && distance > policy.cull_distance {
                AvatarDisplay::Fallback(FallbackReason::Distance)
            } else if policy.impostor_distance > 0.0 && distance > policy.impostor_distance {
                AvatarDisplay::Fallback(FallbackReason::Far)
            } else {
                AvatarDisplay::Full
            }
        })
        .collect();
    if policy.max_shown > 0 {
        let mut shown: Vec<usize> = (0..out.len()).filter(|&i| out[i] == AvatarDisplay::Full).collect();
        shown.sort_by(|&a, &b| avatars[a].1.total_cmp(&avatars[b].1));
        for &i in shown.iter().skip(policy.max_shown as usize) {
            out[i] = AvatarDisplay::Fallback(FallbackReason::TooMany);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(triangles: u64) -> PerfStats {
        PerfStats { triangles, materials: 2, bones: 100, blendshapes: 10, texture_bytes: 10 * MIB }
    }

    #[test]
    fn boundaries_are_inclusive_and_monotonic() {
        let l = limits(ContentKind::Avatar, PerfTier::Desktop, PerfMetric::Triangles);
        assert_eq!(rank_value(0, &l), PerfRank::Excellent);
        assert_eq!(rank_value(32_000, &l), PerfRank::Excellent);
        assert_eq!(rank_value(32_001, &l), PerfRank::Good);
        assert_eq!(rank_value(150_000, &l), PerfRank::Poor);
        assert_eq!(rank_value(150_001, &l), PerfRank::VeryPoor);
        for kind in ContentKind::ALL {
            for tier in [PerfTier::Desktop, PerfTier::Mobile] {
                for m in PerfMetric::ALL {
                    assert!(limits(kind, tier, m).windows(2).all(|w| w[0] < w[1]), "{kind:?} {tier:?} {m:?}");
                }
            }
        }
    }

    #[test]
    fn overall_is_the_worst_metric() {
        let r = rank(&stats(10_000), ContentKind::Avatar, PerfTier::Desktop);
        assert_eq!(r.overall, PerfRank::Excellent);
        let heavy = PerfStats { bones: 500, ..stats(10_000) };
        let r = rank(&heavy, ContentKind::Avatar, PerfTier::Desktop);
        assert_eq!(r.overall, PerfRank::Poor);
        let worst = r.worst().unwrap();
        assert_eq!(worst.metric, PerfMetric::Bones);
        assert_eq!(worst.next_better, Some(400));
        assert_eq!(rank(&stats(10_000), ContentKind::Avatar, PerfTier::Desktop).metrics[0].next_better, None);
    }

    #[test]
    fn mobile_is_stricter_than_desktop() {
        let s = stats(20_000);
        let d = rank(&s, ContentKind::Avatar, PerfTier::Desktop).overall;
        let m = rank(&s, ContentKind::Avatar, PerfTier::Mobile).overall;
        assert!(m > d);
    }

    #[test]
    fn policy_hides_by_rank_distance_and_count() {
        let policy = AvatarPolicy { min_rank: PerfRank::Medium, max_shown: 2, cull_distance: 30.0, ..Default::default() };
        let avatars = [
            (PerfRank::Good, 5.0),
            (PerfRank::Poor, 2.0),
            (PerfRank::Good, 40.0),
            (PerfRank::Excellent, 10.0),
            (PerfRank::Medium, 7.0),
        ];
        let out = decide_avatars(&avatars, &policy);
        assert_eq!(out[0], AvatarDisplay::Full);
        assert_eq!(out[1], AvatarDisplay::Fallback(FallbackReason::Rank(PerfRank::Poor)));
        assert_eq!(out[2], AvatarDisplay::Fallback(FallbackReason::Distance));
        assert_eq!(out[3], AvatarDisplay::Fallback(FallbackReason::TooMany), "the farthest of the three eligible loses its slot");
        assert_eq!(out[4], AvatarDisplay::Full);
        assert!(decide_avatars(&avatars, &AvatarPolicy::default()).iter().filter(|d| **d == AvatarDisplay::Full).count() == 5);
    }

    #[test]
    fn far_avatars_become_impostors_inside_the_cull_distance() {
        let policy = AvatarPolicy { min_rank: PerfRank::Poor, impostor_distance: 20.0, cull_distance: 60.0, max_shown: 1 };
        let avatars = [(PerfRank::Good, 5.0), (PerfRank::Good, 25.0), (PerfRank::Good, 70.0), (PerfRank::Poor, 30.0), (PerfRank::VeryPoor, 3.0)];
        let out = decide_avatars(&avatars, &policy);
        assert_eq!(out[0], AvatarDisplay::Full);
        assert_eq!(out[1], AvatarDisplay::Fallback(FallbackReason::Far), "past the impostor distance, inside the cull distance");
        assert_eq!(out[2], AvatarDisplay::Fallback(FallbackReason::Distance), "the cull distance wins");
        assert_eq!(out[3], AvatarDisplay::Fallback(FallbackReason::Far));
        assert_eq!(out[4], AvatarDisplay::Fallback(FallbackReason::Rank(PerfRank::VeryPoor)), "rank wins over distance");
        // A far avatar does not use one of the full-detail slots.
        let two = decide_avatars(&[(PerfRank::Good, 25.0), (PerfRank::Good, 10.0)], &AvatarPolicy { max_shown: 1, impostor_distance: 20.0, ..Default::default() });
        assert_eq!(two, [AvatarDisplay::Fallback(FallbackReason::Far), AvatarDisplay::Full]);
        // Off by default, and an old settings file without the field still loads.
        assert!(decide_avatars(&avatars, &AvatarPolicy::default()).iter().all(|d| *d == AvatarDisplay::Full));
        let old: AvatarPolicy = ron::from_str("(min_rank: Medium, max_shown: 3, cull_distance: 12.0)").unwrap();
        assert_eq!(old.impostor_distance, 0.0);
    }

    #[test]
    fn package_stats_convert() {
        let p = PackageStats { triangles: 5, materials: 3, bones: 4, blendshapes: 2, texture_bytes: 7, ..Default::default() };
        assert_eq!(PerfStats::from(&p), PerfStats { triangles: 5, materials: 3, bones: 4, blendshapes: 2, texture_bytes: 7 });
    }
}
