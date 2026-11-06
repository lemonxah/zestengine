//! Page residency policy for streamed cluster DAGs (GEO-6). Pure CPU: no
//! Vulkan, so the whole policy is unit tested with simulated cameras.
//!
//! A streamed mesh is a list of pages (page 0 is the always-resident root
//! page) with dependencies: a page may only be resident while every page in its
//! `deps` is (see `zestengine-assets::cluster_pages` for why that is enough to
//! keep the cluster cut closed). The GPU cull reports which missing pages the
//! view wants (with a priority, the projected error over its target) and which
//! resident pages it drew; this module turns that into loads and evictions:
//!
//! * a request pulls in its whole missing ancestor chain, parents first, so a
//!   page is never resident without its parents;
//! * the most urgent requests (largest error ratio) go first, under a per-frame
//!   byte limit and a cap on pages in flight;
//! * room is made by evicting the least recently used pages, but only pages
//!   nothing resident depends on, that no pending request needs, and that were
//!   not used for `min_idle_frames` (the hysteresis against thrash: a page that
//!   just arrived stays at least that long);
//! * when nothing can be evicted the request waits (a stall frame): the coarser
//!   parent keeps drawing, so there is never a hole.
//!
//! The budget counts the bytes of resident and loading non-root pages; root
//! pages are not evictable and are outside it.

use std::collections::{HashMap, HashSet};

/// Per-page input of [`PageManager::add_mesh`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageSpec {
    /// Payload bytes (vertices and triangle bytes).
    pub bytes: u32,
    /// Pages that must be resident whenever this one is.
    pub deps: Vec<u32>,
}

/// Limits and hysteresis of the policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamPolicy {
    /// Bytes of resident plus loading non-root pages.
    pub budget_bytes: u64,
    /// Pages being loaded at once.
    pub max_in_flight: u32,
    /// New bytes admitted per frame.
    pub max_bytes_per_frame: u64,
    /// A page is not evicted until it went this many frames unused (and a
    /// freshly loaded page counts as used).
    pub min_idle_frames: u64,
    /// A request not repeated for this many frames is dropped.
    pub request_ttl: u64,
    /// Frames a failed load waits before it is tried again.
    pub retry_frames: u64,
    /// Requests from shadow views queue behind every request from the main view
    /// and may use at most this fraction of the per-frame bytes; they also may
    /// not take the last `shadow_share` of the in-flight slots or of the budget
    /// (kept free for the main view), so shadows never starve it.
    pub shadow_share: f32,
}

impl Default for StreamPolicy {
    fn default() -> Self {
        Self { budget_bytes: 256 << 20, max_in_flight: 24, max_bytes_per_frame: 16 << 20, min_idle_frames: 60, request_ttl: 12, retry_frames: 120, shadow_share: 0.25 }
    }
}

/// Counters, for the profiler window and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StreamStats {
    pub resident_pages: u32,
    /// Bytes of resident non-root pages.
    pub resident_bytes: u64,
    pub root_pages: u32,
    pub root_bytes: u64,
    pub loading_pages: u32,
    pub loading_bytes: u64,
    pub budget_bytes: u64,
    /// Distinct pages requested (a page asked for again after it was dropped counts again).
    pub requests: u64,
    /// Of `requests`, the ones only shadow views asked for.
    pub shadow_requests: u64,
    pub loads: u64,
    pub evictions: u64,
    pub bytes_streamed: u64,
    /// Frames in which a wanted page could not be admitted (budget or nothing evictable).
    pub stall_frames: u64,
    pub failed_loads: u64,
    /// Requests waiting right now.
    pub pending: u32,
    /// Pages loaded again within `2 * min_idle_frames` of their eviction (thrash).
    pub thrash_reloads: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Absent,
    Loading,
    Resident,
}

#[derive(Clone, Debug)]
struct Page {
    bytes: u32,
    deps: Vec<u32>,
    root: bool,
    state: State,
    last_used: u64,
    /// Resident or loading pages that list this one in `deps`.
    refs: u32,
    retry_after: u64,
    evicted_at: u64,
}

#[derive(Clone, Copy, Debug)]
struct Want {
    prio: f32,
    seen: u64,
    /// Only shadow views asked (a main-view request makes it a main one).
    shadow: bool,
}

/// What [`PageManager::plan`] decided.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Pages to read and install, parents before children.
    pub loads: Vec<(u32, u32)>,
    /// Pages to drop now (mesh, page).
    pub evicts: Vec<(u32, u32)>,
}

pub struct PageManager {
    policy: StreamPolicy,
    meshes: HashMap<u32, Vec<Page>>,
    wanted: HashMap<(u32, u32), Want>,
    frame: u64,
    in_flight: u32,
    loading_bytes: u64,
    resident_bytes: u64,
    stats: StreamStats,
}

impl PageManager {
    pub fn new(policy: StreamPolicy) -> Self {
        Self { policy, meshes: HashMap::new(), wanted: HashMap::new(), frame: 0, in_flight: 0, loading_bytes: 0, resident_bytes: 0, stats: StreamStats::default() }
    }

    pub fn policy(&self) -> StreamPolicy {
        self.policy
    }

    pub fn set_policy(&mut self, p: StreamPolicy) {
        self.policy = p;
    }

    /// Register a mesh; page 0 is its resident root page.
    pub fn add_mesh(&mut self, id: u32, specs: &[PageSpec]) {
        let pages: Vec<Page> = specs
            .iter()
            .enumerate()
            .map(|(i, s)| Page {
                bytes: s.bytes,
                deps: s.deps.clone(),
                root: i == 0,
                state: if i == 0 { State::Resident } else { State::Absent },
                last_used: self.frame,
                refs: 0,
                retry_after: 0,
                evicted_at: 0,
            })
            .collect();
        self.meshes.insert(id, pages);
    }

    /// Forget a mesh (its pages are released by the store). Returns the pages that were resident or loading.
    pub fn remove_mesh(&mut self, id: u32) -> Vec<u32> {
        let Some(pages) = self.meshes.remove(&id) else { return Vec::new() };
        let mut out = Vec::new();
        for (i, p) in pages.iter().enumerate() {
            match p.state {
                State::Resident => {
                    if !p.root {
                        self.resident_bytes -= u64::from(p.bytes);
                    }
                    out.push(i as u32);
                }
                State::Loading => {
                    self.loading_bytes -= u64::from(p.bytes);
                    self.in_flight -= 1;
                    out.push(i as u32);
                }
                State::Absent => {}
            }
        }
        self.wanted.retain(|k, _| k.0 != id);
        out
    }

    pub fn begin_frame(&mut self, frame: u64) {
        self.frame = frame;
    }

    pub fn is_loading(&self, id: u32, page: u32) -> bool {
        self.state(id, page) == Some(State::Loading)
    }

    pub fn is_resident(&self, id: u32, page: u32) -> bool {
        self.state(id, page) == Some(State::Resident)
    }

    fn state(&self, id: u32, page: u32) -> Option<State> {
        self.meshes.get(&id)?.get(page as usize).map(|p| p.state)
    }

    /// The GPU drew from this resident page.
    pub fn touch(&mut self, id: u32, page: u32) {
        let frame = self.frame;
        if let Some(p) = self.meshes.get_mut(&id).and_then(|m| m.get_mut(page as usize)) {
            p.last_used = frame;
        }
    }

    /// The view wants an absent page; `prio` is the projected-error ratio (>= 1
    /// is "too coarse right now"). Repeats are merged (largest priority wins).
    pub fn request(&mut self, id: u32, page: u32, prio: f32) {
        self.request_as(id, page, prio, false);
    }

    /// [`request`](Self::request) from a shadow view (`shadow`) or the main view.
    /// Shadow requests rank behind all main ones and have a capped share of the
    /// bandwidth and budget (see [`StreamPolicy::shadow_share`]); a page both ask
    /// for is a main-view page, ranked by the main view's priority alone.
    pub fn request_as(&mut self, id: u32, page: u32, prio: f32, shadow: bool) {
        if self.state(id, page) != Some(State::Absent) {
            return;
        }
        let frame = self.frame;
        let w = self.wanted.entry((id, page)).or_insert_with(|| {
            self.stats.requests += 1;
            if shadow {
                self.stats.shadow_requests += 1;
            }
            Want { prio: 0.0, seen: frame, shadow }
        });
        if w.shadow && !shadow {
            w.shadow = false;
            w.prio = 0.0;
        }
        if w.shadow == shadow {
            w.prio = w.prio.max(prio);
        }
        w.seen = frame;
    }

    /// Missing pages needed to make `page` resident, parents first.
    fn missing_chain(&self, id: u32, page: u32) -> Vec<u32> {
        let Some(pages) = self.meshes.get(&id) else { return Vec::new() };
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        fn visit(pages: &[Page], p: u32, seen: &mut HashSet<u32>, out: &mut Vec<u32>) {
            if !seen.insert(p) {
                return;
            }
            let Some(pg) = pages.get(p as usize) else { return };
            for &d in &pg.deps {
                visit(pages, d, seen, out);
            }
            if pg.state == State::Absent {
                out.push(p);
            }
        }
        visit(pages, page, &mut seen, &mut out);
        out
    }

    fn all_chain(&self, id: u32, page: u32, into: &mut HashSet<(u32, u32)>) {
        let Some(pages) = self.meshes.get(&id) else { return };
        let mut stack = vec![page];
        while let Some(p) = stack.pop() {
            if !into.insert((id, p)) {
                continue;
            }
            if let Some(pg) = pages.get(p as usize) {
                stack.extend(pg.deps.iter().copied());
            }
        }
    }

    /// The least recently used page that may go: resident, not a root, nothing
    /// resident depends on it, idle for the hysteresis, not protected.
    fn victim(&self, protected: &HashSet<(u32, u32)>, force: bool) -> Option<(u32, u32)> {
        type BestVictim = ((u64, u32, u32), (u32, u32));
        let mut best: Option<BestVictim> = None;
        for (&id, pages) in &self.meshes {
            for (i, p) in pages.iter().enumerate() {
                if p.root || p.state != State::Resident || p.refs > 0 || protected.contains(&(id, i as u32)) {
                    continue;
                }
                if !force && self.frame.saturating_sub(p.last_used) < self.policy.min_idle_frames {
                    continue;
                }
                let key = (p.last_used, id, i as u32);
                if best.is_none_or(|(k, _)| key < k) {
                    best = Some((key, (id, i as u32)));
                }
            }
        }
        best.map(|(_, v)| v)
    }

    fn evict(&mut self, id: u32, page: u32) {
        let frame = self.frame;
        let pages = self.meshes.get_mut(&id).expect("mesh exists");
        let deps = pages[page as usize].deps.clone();
        let p = &mut pages[page as usize];
        debug_assert!(p.state == State::Resident && !p.root && p.refs == 0);
        p.state = State::Absent;
        p.evicted_at = frame;
        self.resident_bytes -= u64::from(p.bytes);
        for d in deps {
            pages[d as usize].refs -= 1;
        }
        self.stats.evictions += 1;
    }

    /// Decide this frame's loads and evictions (call once per frame after the
    /// requests and touches of the frame that just retired). The evictions are
    /// already applied to the state; the loads are `Loading` until
    /// [`loaded`](Self::loaded) / [`failed`](Self::failed).
    pub fn plan(&mut self) -> Plan {
        let mut plan = Plan::default();
        let frame = self.frame;
        let ttl = self.policy.request_ttl;
        let meshes = &self.meshes;
        self.wanted.retain(|&(id, page), w| frame.saturating_sub(w.seen) <= ttl && meshes.get(&id).and_then(|m| m.get(page as usize)).is_some_and(|p| p.state == State::Absent));
        let mut order: Vec<((u32, u32), f32, bool)> = self.wanted.iter().map(|(&k, w)| (k, w.prio, w.shadow)).collect();
        order.sort_by(|a, b| a.2.cmp(&b.2).then(b.1.total_cmp(&a.1)).then(a.0.cmp(&b.0)));
        let mut protected: HashSet<(u32, u32)> = HashSet::new();
        for &((id, page), _, _) in &order {
            self.all_chain(id, page, &mut protected);
        }
        // A lowered budget first: shed least recently used pages, those nothing wants
        // before the ones pending requests build on, whatever their age (the budget is a hard limit).
        let none = HashSet::new();
        for set in [&protected, &none] {
            while self.resident_bytes + self.loading_bytes > self.policy.budget_bytes {
                let Some((id, page)) = self.victim(set, true) else { break };
                self.evict(id, page);
                plan.evicts.push((id, page));
            }
        }
        let mut issued = 0u64;
        let mut shadow_issued = 0u64;
        let mut stalled = false;
        let share = f64::from(self.policy.shadow_share.clamp(0.0, 1.0));
        let shadow_bytes = (self.policy.max_bytes_per_frame as f64 * share) as u64;
        let shadow_flight = ((f64::from(self.policy.max_in_flight) * (1.0 - share)).floor() as u32).max(1).min(self.policy.max_in_flight);
        let shadow_budget = (self.policy.budget_bytes as f64 * (1.0 - share)) as u64;
        for &((id, page), _, shadow) in &order {
            if self.in_flight >= self.policy.max_in_flight || issued >= self.policy.max_bytes_per_frame {
                break;
            }
            if shadow && (self.in_flight >= shadow_flight || shadow_issued >= shadow_bytes) {
                continue;
            }
            if self.state(id, page) != Some(State::Absent) {
                continue;
            }
            let chain = self.missing_chain(id, page);
            if chain.is_empty() {
                continue;
            }
            let pages = &self.meshes[&id];
            if chain.iter().any(|&c| pages[c as usize].retry_after > frame) {
                continue;
            }
            let need: u64 = chain.iter().map(|&c| u64::from(pages[c as usize].bytes)).sum();
            if need > self.policy.budget_bytes {
                stalled = true;
                continue;
            }
            // A shadow chain must fit the part of the budget the main view leaves it (and its in-flight slots).
            if shadow
                && (self.resident_bytes + self.loading_bytes + need > shadow_budget
                    || (self.in_flight > 0 && self.in_flight + chain.len() as u32 > shadow_flight))
            {
                continue;
            }
            // A long chain waits for room in the pipeline (but never forever: an empty pipeline takes any chain).
            if self.in_flight > 0 && self.in_flight + chain.len() as u32 > self.policy.max_in_flight {
                continue;
            }
            while self.resident_bytes + self.loading_bytes + need > self.policy.budget_bytes {
                let Some((vid, vpage)) = self.victim(&protected, false) else { break };
                self.evict(vid, vpage);
                plan.evicts.push((vid, vpage));
            }
            if self.resident_bytes + self.loading_bytes + need > self.policy.budget_bytes {
                stalled = true;
                continue;
            }
            for c in chain {
                let pages = self.meshes.get_mut(&id).expect("mesh exists");
                let deps = pages[c as usize].deps.clone();
                let p = &mut pages[c as usize];
                p.state = State::Loading;
                if frame.saturating_sub(p.evicted_at) < 2 * self.policy.min_idle_frames && p.evicted_at > 0 {
                    self.stats.thrash_reloads += 1;
                }
                let bytes = u64::from(p.bytes);
                for d in deps {
                    pages[d as usize].refs += 1;
                }
                self.loading_bytes += bytes;
                self.in_flight += 1;
                issued += bytes;
                if shadow {
                    shadow_issued += bytes;
                }
                plan.loads.push((id, c));
            }
            self.wanted.remove(&(id, page));
        }
        if stalled {
            self.stats.stall_frames += 1;
        }
        plan
    }

    /// The page's payload is installed on the GPU.
    pub fn loaded(&mut self, id: u32, page: u32) {
        let frame = self.frame;
        let Some(p) = self.meshes.get_mut(&id).and_then(|m| m.get_mut(page as usize)) else { return };
        if p.state != State::Loading {
            return;
        }
        p.state = State::Resident;
        p.last_used = frame;
        let bytes = u64::from(p.bytes);
        self.loading_bytes -= bytes;
        self.resident_bytes += bytes;
        self.in_flight -= 1;
        self.stats.loads += 1;
        self.stats.bytes_streamed += bytes;
    }

    /// The load of a page failed (read error, no space): back to absent, retried
    /// later. Loading pages that depend on it fail with it (their parent is gone);
    /// all of them are returned, the page itself first.
    pub fn failed(&mut self, id: u32, page: u32) -> Vec<u32> {
        let frame = self.frame;
        let retry = self.policy.retry_frames;
        let mut out = Vec::new();
        let Some(pages) = self.meshes.get_mut(&id) else { return out };
        let mut queue = vec![page];
        while let Some(c) = queue.pop() {
            let Some(p) = pages.get_mut(c as usize) else { continue };
            if p.state != State::Loading {
                continue;
            }
            p.state = State::Absent;
            p.retry_after = frame + retry;
            let (bytes, deps) = (u64::from(p.bytes), p.deps.clone());
            for d in deps {
                pages[d as usize].refs -= 1;
            }
            self.loading_bytes -= bytes;
            self.in_flight -= 1;
            self.stats.failed_loads += 1;
            out.push(c);
            for (i, q) in pages.iter().enumerate() {
                if q.state == State::Loading && q.deps.contains(&c) {
                    queue.push(i as u32);
                }
            }
        }
        out
    }

    /// Evict everything evictable now (a changed pool: the caller re-creates its ranges).
    pub fn evict_all(&mut self) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        let none = HashSet::new();
        while let Some((id, page)) = self.victim(&none, true) {
            self.evict(id, page);
            out.push((id, page));
        }
        out
    }

    pub fn stats(&self) -> StreamStats {
        let mut s = self.stats;
        s.budget_bytes = self.policy.budget_bytes;
        s.resident_bytes = self.resident_bytes;
        s.loading_bytes = self.loading_bytes;
        s.loading_pages = self.in_flight;
        s.pending = self.wanted.len() as u32;
        for pages in self.meshes.values() {
            for p in pages {
                if p.state == State::Resident {
                    if p.root {
                        s.root_pages += 1;
                        s.root_bytes += u64::from(p.bytes);
                    } else {
                        s.resident_pages += 1;
                    }
                }
            }
        }
        s
    }

    /// Every invariant the renderer relies on (tests call this after every step).
    pub fn check_invariants(&self) -> Result<(), String> {
        let (mut rb, mut lb, mut inflight) = (0u64, 0u64, 0u32);
        for (&id, pages) in &self.meshes {
            if pages.first().is_none_or(|p| !p.root || p.state != State::Resident) {
                return Err(format!("mesh {id}: root page not resident"));
            }
            let mut refs = vec![0u32; pages.len()];
            for (i, p) in pages.iter().enumerate() {
                match p.state {
                    State::Resident if !p.root => rb += u64::from(p.bytes),
                    State::Loading => {
                        lb += u64::from(p.bytes);
                        inflight += 1;
                    }
                    _ => {}
                }
                if p.state != State::Absent {
                    for &d in &p.deps {
                        refs[d as usize] += 1;
                        // A resident page's parents are resident (or arriving no later: loading pages are admitted parents first).
                        let dp = &pages[d as usize];
                        if p.state == State::Resident && dp.state != State::Resident {
                            return Err(format!("mesh {id}: page {i} is resident but its parent page {d} is not (no-hole invariant)"));
                        }
                        if dp.state == State::Absent {
                            return Err(format!("mesh {id}: page {i} is {:?} but its parent page {d} is absent", p.state));
                        }
                    }
                }
            }
            for (i, p) in pages.iter().enumerate() {
                if p.refs != refs[i] {
                    return Err(format!("mesh {id}: page {i} refcount {} != {}", p.refs, refs[i]));
                }
            }
        }
        if rb != self.resident_bytes || lb != self.loading_bytes || inflight != self.in_flight {
            return Err("byte counters drifted".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A binary tree of `n` pages in heap order: page i's parent is (i + 1) / 2 - 1; page 0 is the root.
    fn tree(n: usize, bytes: u32) -> Vec<PageSpec> {
        (0..n).map(|i| PageSpec { bytes, deps: if i == 0 { vec![] } else { vec![(i.div_ceil(2) - 1) as u32] } }).collect()
    }

    fn policy(budget_pages: u64, bytes: u32) -> StreamPolicy {
        StreamPolicy { budget_bytes: budget_pages * u64::from(bytes), max_in_flight: 8, max_bytes_per_frame: u64::MAX, min_idle_frames: 10, request_ttl: 6, retry_frames: 20, shadow_share: 0.25 }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// Run `frames` frames: `want(frame)` lists (page, prio) the "camera" needs,
    /// loads complete after `latency` frames. Returns the manager.
    fn drive(m: &mut PageManager, frames: u64, latency: u64, mut want: impl FnMut(u64, &PageManager) -> Vec<(u32, f32)>, mut touched: impl FnMut(u64) -> Vec<u32>) {
        let mut arriving: Vec<(u64, u32, u32)> = Vec::new();
        for f in 1..=frames {
            m.begin_frame(f);
            let (due, rest): (Vec<_>, Vec<_>) = arriving.into_iter().partition(|a| a.0 <= f);
            arriving = rest;
            for (_, id, p) in due {
                m.loaded(id, p);
            }
            for p in touched(f) {
                if m.is_resident(1, p) {
                    m.touch(1, p);
                }
            }
            for (p, pr) in want(f, m) {
                m.request(1, p, pr);
            }
            let plan = m.plan();
            for (id, p) in plan.loads {
                arriving.push((f + latency, id, p));
            }
            m.check_invariants().unwrap_or_else(|e| panic!("frame {f}: {e}"));
            let s = m.stats();
            assert!(s.resident_bytes + s.loading_bytes <= s.budget_bytes, "frame {f}: budget exceeded ({} + {} > {})", s.resident_bytes, s.loading_bytes, s.budget_bytes);
        }
    }

    #[test]
    fn a_request_loads_its_whole_ancestor_chain_parents_first() {
        let mut m = PageManager::new(policy(100, 1000));
        m.add_mesh(1, &tree(63, 1000));
        m.begin_frame(1);
        m.request(1, 62, 4.0); // a deep leaf
        let plan = m.plan();
        let pages: Vec<u32> = plan.loads.iter().map(|l| l.1).collect();
        assert_eq!(pages, vec![2, 6, 14, 30, 62], "parents first, all missing ancestors, root skipped");
        m.check_invariants().unwrap();
        // Nothing is resident yet; a second request for a sibling asks only for itself.
        for &(id, p) in &plan.loads {
            m.loaded(id, p);
        }
        m.begin_frame(2);
        m.request(1, 61, 1.5);
        let plan = m.plan();
        assert_eq!(plan.loads, vec![(1, 61)]);
    }

    #[test]
    fn shadow_requests_queue_behind_the_main_view_and_cannot_take_all_slots() {
        let pol = StreamPolicy { max_in_flight: 4, shadow_share: 0.5, ..policy(100, 1000) };
        let build = || {
            let mut m = PageManager::new(pol);
            m.add_mesh(1, &tree(7, 1000)); // 1,2 under the root; 3,4 under 1; 5,6 under 2
            m.begin_frame(1);
            m
        };
        // Shadows alone: very urgent, but only half the slots (2 of 4).
        let mut m = build();
        for p in [3, 4, 5, 6] {
            m.request_as(1, p, 1.0e6, true);
        }
        let plan = m.plan();
        assert!(plan.loads.len() <= 2, "shadow loads {:?} exceed their share of the in-flight slots", plan.loads);
        assert!(!plan.loads.is_empty(), "shadows do load when the main view is quiet");
        // Main requests rank first however small their priority, and the shadows wait.
        let mut m = build();
        for p in [3, 4, 5, 6] {
            m.request_as(1, p, 1.0e6, true);
        }
        m.request(1, 2, 0.5);
        m.request(1, 4, 0.01);
        let plan = m.plan();
        assert_eq!(plan.loads, vec![(1, 2), (1, 1), (1, 4)], "main chains first; the shadow share is gone once they are in flight");
        m.check_invariants().unwrap();
        // A page both views ask for is a main page, ranked by the main priority alone.
        let mut m = build();
        m.request_as(1, 5, 1.0e6, true);
        m.request(1, 5, 0.01);
        m.request_as(1, 3, 1.0e6, true);
        let plan = m.plan();
        assert_eq!(plan.loads[0], (1, 2), "the merged page is first, with its parent");
        // The budget a shadow may fill stops short of the main view's headroom.
        let pol = StreamPolicy { budget_bytes: 4000, max_in_flight: 16, shadow_share: 0.25, ..policy(100, 1000) };
        let mut m = PageManager::new(pol);
        m.add_mesh(1, &tree(7, 1000));
        m.begin_frame(1);
        for p in [3, 4, 5, 6] {
            m.request_as(1, p, 1.0e6, true);
        }
        let plan = m.plan();
        let bytes: u64 = plan.loads.len() as u64 * 1000;
        assert!(bytes <= 3000, "shadows took {bytes} of a 4000 byte budget; 25 % is kept for the main view");
    }

    #[test]
    fn requests_are_deduplicated() {
        let mut m = PageManager::new(policy(100, 1000));
        m.add_mesh(1, &tree(15, 1000));
        for f in 1..=5u64 {
            m.begin_frame(f);
            for _ in 0..50 {
                m.request(1, 14, 2.0);
            }
            let plan = m.plan();
            if f == 1 {
                assert_eq!(plan.loads.len(), 3, "page 14 and its two ancestors, once");
            } else {
                assert!(plan.loads.is_empty(), "frame {f}: a page already loading was asked for again");
            }
        }
        assert_eq!(m.stats().requests, 1);
        // A page already resident is not requested at all.
        for p in [2, 6, 14] {
            m.loaded(1, p);
        }
        m.begin_frame(6);
        m.request(1, 14, 9.0);
        assert!(m.plan().loads.is_empty());
        assert_eq!(m.stats().pending, 0);
    }

    #[test]
    fn the_most_urgent_requests_go_first_under_the_per_frame_limit() {
        let mut p = policy(100, 1000);
        p.max_bytes_per_frame = 2000;
        let mut m = PageManager::new(p);
        m.add_mesh(1, &tree(7, 1000)); // root 0; 1, 2; 3, 4, 5, 6
        m.begin_frame(1);
        m.request(1, 3, 1.1);
        m.request(1, 5, 8.0); // needs 2 then 5
        m.request(1, 6, 3.0);
        let plan = m.plan();
        assert_eq!(plan.loads, vec![(1, 2), (1, 5)], "the largest error ratio first");
        assert_eq!(m.stats().pending, 2);
    }

    #[test]
    fn eviction_is_lru_respects_dependents_and_hysteresis() {
        let mut m = PageManager::new(policy(4, 1000));
        m.add_mesh(1, &tree(31, 1000));
        // Fill the budget: chain 1 -> 3 -> 7 and 2.
        m.begin_frame(1);
        m.request(1, 7, 2.0);
        m.request(1, 2, 1.5);
        for (id, p) in m.plan().loads {
            m.loaded(id, p);
        }
        assert_eq!(m.stats().resident_pages, 4);
        // A new request needs room: everything is fresher than min_idle, so it waits (a stall), nothing is evicted.
        m.begin_frame(3);
        m.request(1, 14, 5.0); // chain 2 (resident), 6, 14: needs 2 pages
        let plan = m.plan();
        assert!(plan.loads.is_empty() && plan.evicts.is_empty(), "evicted inside the hysteresis window: {plan:?}");
        assert!(m.stats().stall_frames >= 1);
        // Later the idle pages go: page 7 is a leaf (evictable), page 3 has a dependent until 7 is gone.
        m.begin_frame(30);
        m.touch(1, 2);
        m.request(1, 14, 5.0);
        let plan = m.plan();
        assert!(plan.evicts.contains(&(1, 7)) && plan.evicts.contains(&(1, 3)), "{plan:?}");
        assert!(!plan.evicts.contains(&(1, 2)), "the page the chain needs must stay");
        assert_eq!(plan.loads, vec![(1, 6), (1, 14)]);
        m.check_invariants().unwrap();
    }

    #[test]
    fn failed_loads_back_off_and_free_their_budget() {
        let mut m = PageManager::new(policy(10, 1000));
        m.add_mesh(1, &tree(7, 1000));
        m.begin_frame(1);
        m.request(1, 1, 2.0);
        let plan = m.plan();
        assert_eq!(plan.loads, vec![(1, 1)]);
        m.failed(1, 1);
        assert_eq!(m.stats().loading_bytes, 0);
        m.check_invariants().unwrap();
        m.begin_frame(2);
        m.request(1, 1, 2.0);
        assert!(m.plan().loads.is_empty(), "retried before the backoff");
        m.begin_frame(30);
        m.request(1, 1, 2.0);
        assert_eq!(m.plan().loads, vec![(1, 1)]);
    }

    #[test]
    fn budget_is_respected_and_no_hole_invariant_holds_under_random_requests() {
        for seed in 1..6u64 {
            let mut rng = Rng(seed);
            let mut m = PageManager::new(policy(12, 1000));
            m.add_mesh(1, &tree(255, 1000));
            let mut rng2 = Rng(seed * 77);
            drive(
                &mut m,
                600,
                3,
                |_, _| (0..(rng.next() % 6)).map(|_| (1 + (rng.next() % 254) as u32, 1.0 + (rng.next() % 100) as f32 / 10.0)).collect(),
                |_| (0..8).map(|_| (rng2.next() % 255) as u32).collect(),
            );
            let s = m.stats();
            assert!(s.resident_bytes <= s.budget_bytes);
            assert!(s.loads > 20 && s.evictions > 5, "seed {seed}: the test did not exercise the policy: {s:?}");
        }
    }

    #[test]
    fn a_camera_sweep_does_not_thrash() {
        // 512 leaf pages in a binary tree; the camera sees a window of 16 consecutive leaves and sweeps through all of them twice.
        let leaves = 512u32;
        let n = (2 * leaves - 1) as usize;
        let first_leaf = leaves - 1;
        // Budget: window leaves + their ancestors (~ 16 + 16 + 8 + ... ) with slack.
        let mut m = PageManager::new(policy(120, 1000));
        m.add_mesh(1, &tree(n, 1000));
        let window = 16u32;
        let sweep_frames = 2 * leaves as u64 * 2;
        let pos = move |f: u64| ((f / 2) as u32) % leaves;
        let leaf_page = move |f: u64, k: u32| first_leaf + (pos(f) + k) % leaves;
        drive(
            &mut m,
            sweep_frames,
            2,
            |f, m| {
                (0..window)
                    .map(|k| leaf_page(f, k))
                    .filter(|&p| !m.is_resident(1, p))
                    .map(|p| (p, 2.0))
                    .collect()
            },
            |f| (0..window).map(|k| leaf_page(f, k)).collect(),
        );
        let s = m.stats();
        // Every leaf enters the window at most twice; ancestors are shared. Loads stay near the number of leaves swept.
        let swept = 2 * leaves as u64;
        assert!(s.loads <= swept * 2 + 200, "{} loads for {swept} leaf entries: thrashing ({s:?})", s.loads);
        assert_eq!(s.thrash_reloads, 0, "pages were reloaded right after being evicted: {s:?}");
        assert!(s.evictions + 130 >= s.loads, "memory is not being recycled: {s:?}");
    }

    #[test]
    fn over_budget_working_sets_stall_instead_of_thrashing() {
        // The view wants 40 leaves at once but the budget holds 10 pages: the policy keeps what it has, loads what fits, and counts stalls.
        let mut m = PageManager::new(policy(10, 1000));
        m.add_mesh(1, &tree(127, 1000));
        let want: Vec<(u32, f32)> = (63..103).map(|p| (p, 2.0)).collect();
        drive(&mut m, 300, 2, |_, _| want.clone(), |_| (63..103).collect());
        let s = m.stats();
        assert!(s.stall_frames > 100, "{s:?}");
        assert!(s.loads <= 40, "{} loads for a working set that cannot fit: churn ({s:?})", s.loads);
        assert_eq!(s.thrash_reloads, 0);
    }

    #[test]
    fn removing_a_mesh_releases_its_bytes() {
        let mut m = PageManager::new(policy(50, 1000));
        m.add_mesh(1, &tree(15, 1000));
        m.add_mesh(2, &tree(15, 1000));
        m.begin_frame(1);
        m.request(1, 14, 2.0);
        m.request(2, 7, 2.0);
        let plan = m.plan();
        for &(id, p) in plan.loads.iter().take(3) {
            m.loaded(id, p);
        }
        m.check_invariants().unwrap();
        let freed = m.remove_mesh(1);
        assert!(!freed.is_empty());
        m.check_invariants().unwrap();
        assert!(m.stats().resident_bytes + m.stats().loading_bytes <= 4000);
        m.remove_mesh(2);
        let s = m.stats();
        assert_eq!((s.resident_bytes, s.loading_bytes, s.loading_pages), (0, 0, 0));
    }

    #[test]
    fn a_lowered_budget_sheds_idle_pages() {
        let mut m = PageManager::new(policy(20, 1000));
        m.add_mesh(1, &tree(63, 1000));
        m.begin_frame(1);
        for p in 31..39 {
            m.request(1, p, 2.0);
        }
        for (id, p) in m.plan().loads {
            m.loaded(id, p);
        }
        let before = m.stats().resident_bytes;
        assert!(before >= 8000);
        let mut p = m.policy();
        p.budget_bytes = 4000;
        m.set_policy(p);
        m.begin_frame(200);
        m.plan();
        assert!(m.stats().resident_bytes <= 4000);
        m.check_invariants().unwrap();
    }
}
