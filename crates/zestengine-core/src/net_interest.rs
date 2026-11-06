//! Interest management settings and the pure data structures behind them
//! (NET-5): a spatial hash of networked objects, a per-client visible set with
//! hysteresis, update-rate tiers by distance and a starvation-free bandwidth
//! budget.
//!
//! Plain data and plain logic, readable from the scene's `Network Manager` and
//! overridable in the dedicated server's `server.ron`. The engine
//! (`zestengine_engine::net`) drives it from the session; everything here has
//! serde defaults, so old scenes and configs load unchanged. Off by default:
//! a listen host behaves exactly as before until a project opts in.

use std::collections::{HashMap, HashSet};

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// One distance band of the update-rate table.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RateTier {
    /// The band starts at this distance from the player (meters).
    pub distance: f32,
    /// Updates per second for objects in the band (0 = every update the
    /// owner makes).
    pub rate: f32,
}

impl Default for RateTier {
    fn default() -> Self {
        Self { distance: 0.0, rate: 0.0 }
    }
}

/// Interest management: who gets told about what, and how often.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InterestConfig {
    /// Off = every client gets everything (the listen-host behaviour).
    pub enabled: bool,
    /// Edge of one cell of the spatial hash (meters). About the radius / 3.
    pub cell_size: f32,
    /// An object becomes relevant to a player inside this distance (meters).
    pub radius: f32,
    /// ... and stops being relevant only past `radius + hysteresis`, so
    /// objects on the border do not flicker in and out.
    pub hysteresis: f32,
    /// Relevance recomputations per second (the grid and every client's set).
    pub refresh_hz: f32,
    /// This many of the nearest players are always sent at full rate
    /// whatever the tiers say (0 = tiers only).
    pub nearest_full: u32,
    /// Updates per second those nearest players get (0 = every update the owner makes). A 90 Hz
    /// headset otherwise makes the server relay 90 poses a second to each of its neighbours.
    pub full_rate: f32,
    /// Update rate by distance, ascending. Empty = every update.
    pub tiers: Vec<RateTier>,
    /// Outgoing bytes per second and client; the busiest, longest waiting
    /// updates go first and the rest wait (0 = unlimited).
    pub budget: u32,
    /// How much unused budget a client may save up, seconds.
    pub burst: f32,
    /// Voice frames reach listeners within this distance of the speaker (meters).
    pub voice_range: f32,
    /// Face tracking and avatar parameters only reach players within this
    /// distance (meters, capped by `radius`).
    pub avatar_detail_range: f32,
    /// Send 32-bit network ids instead of 128-bit ones in snapshots (the
    /// host teaches each client the mapping when an object becomes relevant).
    pub short_ids: bool,
    /// Per-client delta baselines: a field that has not changed since the last one this client
    /// was sent is left out of its snapshot entry, and an entry with nothing new is not sent
    /// (the owner's own threshold filter works per sender; this one per receiver).
    pub delta: bool,
    /// Seconds after which a full entry is sent anyway, so a lost datagram cannot leave a
    /// client on a stale pose for long (the baseline assumes delivery).
    pub delta_refresh: f32,
}

impl Default for InterestConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cell_size: 50.0,
            radius: 150.0,
            hysteresis: 20.0,
            refresh_hz: 10.0,
            nearest_full: 16,
            full_rate: 30.0,
            tiers: vec![RateTier { distance: 0.0, rate: 30.0 }, RateTier { distance: 20.0, rate: 15.0 }, RateTier { distance: 45.0, rate: 8.0 }, RateTier { distance: 90.0, rate: 4.0 }],
            budget: 0,
            burst: 0.25,
            voice_range: 50.0,
            avatar_detail_range: 25.0,
            short_ids: true,
            delta: true,
            delta_refresh: 1.0,
        }
    }
}

impl InterestConfig {
    /// Clamp to sane values (a bad file must not divide by zero).
    pub fn sanitized(mut self) -> Self {
        let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
        self.cell_size = fin(self.cell_size, 50.0).clamp(1.0, 10_000.0);
        self.radius = fin(self.radius, 150.0).clamp(1.0, 100_000.0);
        self.hysteresis = fin(self.hysteresis, 20.0).clamp(0.0, 10_000.0);
        self.refresh_hz = fin(self.refresh_hz, 10.0).clamp(0.5, 120.0);
        self.burst = fin(self.burst, 0.25).clamp(0.02, 5.0);
        self.voice_range = fin(self.voice_range, 50.0).clamp(0.0, 100_000.0);
        self.avatar_detail_range = fin(self.avatar_detail_range, 25.0).clamp(0.0, 100_000.0);
        self.full_rate = fin(self.full_rate, 30.0).clamp(0.0, 1000.0);
        self.delta_refresh = fin(self.delta_refresh, 1.0).clamp(0.05, 60.0);
        for t in &mut self.tiers {
            t.distance = fin(t.distance, 0.0).max(0.0);
            t.rate = fin(t.rate, 0.0).max(0.0);
        }
        self.tiers.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        self
    }

    /// Seconds between updates of an object `dist` meters away (0 = every one).
    pub fn interval(&self, dist: f32) -> f32 {
        let mut rate = 0.0;
        for t in &self.tiers {
            if t.distance <= dist {
                rate = t.rate;
            }
        }
        if rate > 0.0 { 1.0 / rate } else { 0.0 }
    }

    /// Seconds between updates of one of the nearest players (0 = every one).
    pub fn full_interval(&self) -> f32 {
        if self.full_rate > 0.0 { 1.0 / self.full_rate } else { 0.0 }
    }

    /// Distance past which a visible object leaves the set.
    pub fn leave_radius(&self) -> f32 {
        self.radius + self.hysteresis
    }
}

// ------------------------------------------------------------------ spatial hash

type Cell = (i32, i32, i32);

/// A uniform grid of object positions, rebuilt every refresh. The cell lists
/// keep their capacity, so a steady-state rebuild does not allocate.
#[derive(Default)]
pub struct SpatialGrid {
    cell: f32,
    cells: HashMap<Cell, Vec<(u128, Vec3)>>,
    rebuilds: u32,
}

impl SpatialGrid {
    fn key(&self, p: Vec3) -> Cell {
        let c = |v: f32| (v / self.cell).floor().clamp(-1.0e9, 1.0e9) as i32;
        (c(p.x), c(p.y), c(p.z))
    }

    /// Start a rebuild with this cell size.
    pub fn clear(&mut self, cell: f32) {
        self.cell = cell.max(1.0);
        for v in self.cells.values_mut() {
            v.clear();
        }
        self.rebuilds += 1;
        if self.rebuilds.is_multiple_of(64) {
            self.cells.retain(|_, v| v.capacity() > 0 && !v.is_empty());
        }
    }

    /// Insert (or move) an object to its cell.
    pub fn insert(&mut self, id: u128, pos: Vec3) {
        if !pos.is_finite() {
            return;
        }
        let k = self.key(pos);
        self.cells.entry(k).or_default().push((id, pos));
    }

    /// Number of objects in the grid.
    pub fn len(&self) -> usize {
        self.cells.values().map(Vec::len).sum()
    }

    /// True when no object is in the grid.
    pub fn is_empty(&self) -> bool {
        self.cells.values().all(Vec::is_empty)
    }

    /// Every object in the cells a sphere of `radius` around `center` touches
    /// (a superset of the sphere: the caller checks the distance).
    pub fn query(&self, center: Vec3, radius: f32, mut f: impl FnMut(u128, Vec3)) {
        if self.cell <= 0.0 {
            return;
        }
        let lo = self.key(center - Vec3::splat(radius));
        let hi = self.key(center + Vec3::splat(radius));
        let span = (hi.0 as i64 - lo.0 as i64 + 1) * (hi.1 as i64 - lo.1 as i64 + 1) * (hi.2 as i64 - lo.2 as i64 + 1);
        if span > self.cells.len() as i64 {
            // Fewer occupied cells than cells in the box: walk the occupied ones.
            for (k, v) in &self.cells {
                if k.0 >= lo.0 && k.0 <= hi.0 && k.1 >= lo.1 && k.1 <= hi.1 && k.2 >= lo.2 && k.2 <= hi.2 {
                    for (id, p) in v {
                        f(*id, *p);
                    }
                }
            }
            return;
        }
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    if let Some(v) = self.cells.get(&(x, y, z)) {
                        for (id, p) in v {
                            f(*id, *p);
                        }
                    }
                }
            }
        }
    }

}

// ------------------------------------------------------------------ per client

/// A change of one client's visible set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    /// The object entered the client's visible set.
    Enter(u128),
    /// The object left the client's visible set.
    Leave(u128),
}

/// What one client currently sees, how much it may still be sent, and the
/// numbers the stats show.
#[derive(Default)]
pub struct ClientInterest {
    /// Where the player is (None = no player object yet: it only sees the
    /// always-relevant objects).
    pub center: Option<Vec3>,
    /// What the client currently sees.
    pub visible: HashSet<u128>,
    /// The nearest players, always sent at full rate.
    pub full: HashSet<u128>,
    /// Bytes this client may still be sent now (can dip below zero after
    /// reliable traffic; a deficit is repaid by the refill).
    pub credit: f32,
    scratch: Vec<(f32, u128)>,
    // Stats.
    window_start: f64,
    window_bytes: u64,
    /// Bytes per second over the last full second.
    pub rate_out: f32,
    /// Bytes sent to this client so far.
    pub bytes_total: u64,
    /// Updates that waited or were dropped because the budget was spent.
    pub dropped_budget: u64,
    /// Updates skipped because of the distance tier.
    pub throttled: u64,
    /// Entries left out because the client already had every field (delta baselines).
    pub suppressed: u64,
    /// Objects entered / left since the start.
    pub entered: u64,
    /// Objects that left the set since the start.
    pub left: u64,
}

impl ClientInterest {
    /// A client with this much initial credit (budget times burst).
    pub fn new(cfg: &InterestConfig) -> Self {
        Self { credit: cfg.budget as f32 * cfg.burst, ..Default::default() }
    }

    /// Recompute the visible set against `grid`, appending the changes to `out`.
    /// `always` objects are in the set whatever the distance. `pos_of` finds
    /// a visible object's current position (None = gone or unplaced). `players`
    /// are the player objects, for the nearest-full set.
    pub fn refresh(
        &mut self,
        cfg: &InterestConfig,
        grid: &SpatialGrid,
        always: &HashSet<u128>,
        players: &[u128],
        pos_of: impl Fn(u128) -> Option<Vec3>,
        out: &mut Vec<Change>,
    ) {
        let start = out.len();
        let leave_r = cfg.leave_radius();
        let center = self.center;
        self.visible.retain(|id| {
            let keep = always.contains(id)
                || match (center, pos_of(*id)) {
                    (Some(c), Some(p)) => c.distance(p) <= leave_r,
                    _ => false,
                };
            if !keep {
                out.push(Change::Leave(*id));
            }
            keep
        });
        let mark = out.len();
        if let Some(c) = center {
            grid.query(c, cfg.radius, |id, p| {
                if c.distance(p) <= cfg.radius && !self.visible.contains(&id) {
                    out.push(Change::Enter(id));
                }
            });
        }
        for id in always {
            if !self.visible.contains(id) {
                out.push(Change::Enter(*id));
            }
        }
        for ch in &out[mark..] {
            if let Change::Enter(id) = ch {
                self.visible.insert(*id);
            }
        }
        for ch in &out[start..] {
            match ch {
                Change::Enter(_) => self.entered += 1,
                Change::Leave(_) => self.left += 1,
            }
        }
        self.full.clear();
        if cfg.nearest_full > 0
            && let Some(c) = center
        {
            self.scratch.clear();
            for id in players {
                if self.visible.contains(id)
                    && let Some(p) = pos_of(*id)
                {
                    self.scratch.push((c.distance_squared(p), *id));
                }
            }
            let n = cfg.nearest_full as usize;
            if self.scratch.len() > n {
                self.scratch.select_nth_unstable_by(n - 1, |a, b| a.0.total_cmp(&b.0));
                self.scratch.truncate(n);
            }
            for (_, id) in &self.scratch {
                self.full.insert(*id);
            }
        }
    }

    /// Seconds between updates of `id`, `dist` meters away.
    pub fn interval(&self, cfg: &InterestConfig, id: u128, dist: f32) -> f32 {
        if self.full.contains(&id) { 0.0 } else { cfg.interval(dist) }
    }

    /// Add `dt` seconds of budget.
    pub fn refill(&mut self, cfg: &InterestConfig, dt: f32) {
        if cfg.budget == 0 {
            self.credit = 0.0;
            return;
        }
        let rate = cfg.budget as f32;
        self.credit = (self.credit + rate * dt).clamp(-rate, rate * cfg.burst);
    }

    /// May a send of `bytes` go now?
    pub fn affords(&self, cfg: &InterestConfig, bytes: usize) -> bool {
        cfg.budget == 0 || self.credit >= bytes as f32
    }

    /// Record `bytes` sent (against the budget and the rate window).
    pub fn spend(&mut self, cfg: &InterestConfig, bytes: usize, now: f64) {
        if cfg.budget != 0 {
            self.credit -= bytes as f32;
        }
        self.bytes_total += bytes as u64;
        if now - self.window_start >= 1.0 {
            self.rate_out = self.window_bytes as f32 / (now - self.window_start) as f32;
            self.window_start = now;
            self.window_bytes = 0;
        }
        self.window_bytes += bytes as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> InterestConfig {
        InterestConfig { enabled: true, cell_size: 10.0, radius: 30.0, hysteresis: 10.0, nearest_full: 0, tiers: vec![RateTier { distance: 0.0, rate: 0.0 }, RateTier { distance: 60.0, rate: 10.0 }, RateTier { distance: 110.0, rate: 4.0 }], ..Default::default() }.sanitized()
    }

    fn grid(objs: &[(u128, Vec3)], cell: f32) -> SpatialGrid {
        let mut g = SpatialGrid::default();
        g.clear(cell);
        for (id, p) in objs {
            g.insert(*id, *p);
        }
        g
    }

    fn pos(objs: &[(u128, Vec3)]) -> impl Fn(u128) -> Option<Vec3> + '_ {
        move |id| objs.iter().find(|(i, _)| *i == id).map(|(_, p)| *p)
    }

    #[test]
    fn enter_leave_with_hysteresis() {
        let c = cfg();
        let mut ci = ClientInterest::new(&c);
        ci.center = Some(Vec3::ZERO);
        let mut out = Vec::new();
        let mut objs = vec![(1u128, Vec3::new(25.0, 0.0, 0.0)), (2, Vec3::new(200.0, 0.0, 0.0))];
        ci.refresh(&c, &grid(&objs, 10.0), &HashSet::new(), &[], pos(&objs), &mut out);
        assert_eq!(out, vec![Change::Enter(1)]);
        // Drifts to 35 m: past the enter radius, inside the leave radius: stays.
        objs[0].1.x = 35.0;
        out.clear();
        ci.refresh(&c, &grid(&objs, 10.0), &HashSet::new(), &[], pos(&objs), &mut out);
        assert!(out.is_empty() && ci.visible.contains(&1));
        // 41 m: gone. At 35 m again it does not come back (enter radius is 30).
        objs[0].1.x = 41.0;
        ci.refresh(&c, &grid(&objs, 10.0), &HashSet::new(), &[], pos(&objs), &mut out);
        assert_eq!(out, vec![Change::Leave(1)]);
        objs[0].1.x = 35.0;
        out.clear();
        ci.refresh(&c, &grid(&objs, 10.0), &HashSet::new(), &[], pos(&objs), &mut out);
        assert!(out.is_empty());
        objs[0].1.x = 29.0;
        ci.refresh(&c, &grid(&objs, 10.0), &HashSet::new(), &[], pos(&objs), &mut out);
        assert_eq!(out, vec![Change::Enter(1)]);
    }

    #[test]
    fn always_relevant_and_no_player() {
        let c = cfg();
        let objs = vec![(1u128, Vec3::new(500.0, 0.0, 0.0)), (2, Vec3::new(900.0, 0.0, 0.0))];
        let always: HashSet<u128> = [1u128].into_iter().collect();
        let mut ci = ClientInterest::new(&c);
        ci.center = Some(Vec3::ZERO);
        let mut out = Vec::new();
        ci.refresh(&c, &grid(&objs, 10.0), &always, &[], pos(&objs), &mut out);
        assert_eq!(out, vec![Change::Enter(1)]);
        // Without a player object a client sees only the always-relevant ones.
        let mut blind = ClientInterest::new(&c);
        out.clear();
        blind.refresh(&c, &grid(&objs, 10.0), &always, &[], pos(&objs), &mut out);
        assert_eq!(out, vec![Change::Enter(1)]);
    }

    #[test]
    fn grid_query_matches_brute_force() {
        let mut objs = Vec::new();
        let mut s = 12345u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 400.0 - 200.0
        };
        for i in 0..500u128 {
            objs.push((i, Vec3::new(rnd(), rnd() * 0.1, rnd())));
        }
        let g = grid(&objs, 25.0);
        for c in [Vec3::ZERO, Vec3::new(120.0, 0.0, -80.0), Vec3::new(-300.0, 0.0, 300.0)] {
            let mut got = Vec::new();
            g.query(c, 60.0, |id, p| {
                if c.distance(p) <= 60.0 {
                    got.push(id);
                }
            });
            got.sort();
            let mut want: Vec<u128> = objs.iter().filter(|(_, p)| c.distance(*p) <= 60.0).map(|(i, _)| *i).collect();
            want.sort();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn the_nearest_players_have_a_rate_cap_too() {
        let c = InterestConfig::default();
        assert!((c.full_interval() - 1.0 / 30.0).abs() < 1e-6, "a 90 Hz headset is relayed at 30 Hz to its nearest neighbours");
        assert_eq!(InterestConfig { full_rate: 0.0, ..c.clone() }.full_interval(), 0.0);
        assert!(c.interval(5.0) > 0.0 && c.interval(10.0) < c.interval(30.0) && c.interval(30.0) < c.interval(60.0) && c.interval(60.0) < c.interval(200.0), "the tiers fall with distance");
        assert_eq!(InterestConfig { full_rate: f32::NAN, ..c }.sanitized().full_rate, 30.0);
    }

    #[test]
    fn tiers_and_nearest_full() {
        let c = InterestConfig { nearest_full: 2, ..cfg() };
        assert_eq!(c.interval(5.0), 0.0);
        assert!((c.interval(70.0) - 0.1).abs() < 1e-6);
        assert!((c.interval(500.0) - 0.25).abs() < 1e-6);
        let objs: Vec<(u128, Vec3)> = (1..=4u128).map(|i| (i, Vec3::new(i as f32 * 5.0, 0.0, 0.0))).collect();
        let mut ci = ClientInterest::new(&c);
        ci.center = Some(Vec3::ZERO);
        let mut out = Vec::new();
        ci.refresh(&c, &grid(&objs, 10.0), &HashSet::new(), &[1, 2, 3, 4], pos(&objs), &mut out);
        assert_eq!(ci.full.len(), 2);
        assert!(ci.full.contains(&1) && ci.full.contains(&2));
        assert_eq!(ci.interval(&c, 1, 500.0), 0.0);
        assert!(ci.interval(&c, 4, 500.0) > 0.0);
    }

    #[test]
    fn budget_refills_and_caps() {
        let c = InterestConfig { budget: 1000, burst: 0.5, ..cfg() };
        let mut ci = ClientInterest::new(&c);
        assert!(ci.affords(&c, 500) && !ci.affords(&c, 501));
        ci.spend(&c, 500, 0.0);
        assert!(!ci.affords(&c, 1));
        ci.refill(&c, 0.1);
        assert!(ci.affords(&c, 100) && !ci.affords(&c, 101));
        ci.refill(&c, 10.0);
        assert!(ci.credit <= 500.0 + 1e-3);
        // Unlimited.
        let free = InterestConfig { budget: 0, ..c };
        assert!(ci.affords(&free, 1_000_000));
    }
}
