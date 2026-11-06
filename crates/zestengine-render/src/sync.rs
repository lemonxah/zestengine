//! Per-resource Vulkan sync2 state tracker (Phase 0).
//!
//! A pass states only the access it needs next (`require`). The tracker remembers
//! the last writer and every reader since that write, and [`Tracker::flush`]
//! submits those hazards as one `vkCmdPipelineBarrier2`. Derivation is pure CPU;
//! [`Tracker::drain`] returns the barrier structs `flush` would submit, so the
//! rules are tested without a device.
//!
//! Check mode ([`Tracker::observe`]) runs beside hand-written barriers and records
//! a mismatch when a source scope misses a real prior use. It is not wired into
//! the frame path yet: that waits until the other hand-off tasks stop editing
//! `lib.rs` (see `docs/design/render-graph-barriers.md`).
//!
//! One tracker per queue. Its state is never reset between frames, which is what
//! makes resources shared across frames in flight correct. Invariant: record
//! order equals submit order on that queue. `flush` must be called outside
//! `cmd_begin_rendering` / `cmd_end_rendering`, and between sequenced uses of the
//! same resource: barriers inside one submission are not ordered against each
//! other.

use std::collections::{HashMap, HashSet};
use std::fmt;

use ash::vk;
use ash::vk::Handle as _;

/// Pipeline stages a caller names for storage, uniform, and acceleration-structure
/// reads. Raw bits, because `ash`'s flag `|` is not `const`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Stages(u64);

impl Stages {
    pub const NONE: Self = Self(0);
    pub const VERTEX: Self = Self(vk::PipelineStageFlags2::VERTEX_SHADER.as_raw());
    pub const FRAGMENT: Self = Self(vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw());
    pub const COMPUTE: Self = Self(vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw());
    pub const RAY_QUERY: Self = Self(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR.as_raw());
    pub const TRANSFER: Self = Self(vk::PipelineStageFlags2::TRANSFER.as_raw());
    pub const DRAW_INDIRECT: Self = Self(vk::PipelineStageFlags2::DRAW_INDIRECT.as_raw());
    pub const ALL_GRAPHICS: Self = Self(vk::PipelineStageFlags2::ALL_GRAPHICS.as_raw());
    pub const ALL_COMMANDS: Self = Self(vk::PipelineStageFlags2::ALL_COMMANDS.as_raw());

    pub const fn bits(self) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::from_raw(self.0)
    }

    pub const fn from_vk(flags: vk::PipelineStageFlags2) -> Self {
        Self(flags.as_raw())
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// One way a pass uses a resource. Each maps to `(stages, access, layout)` via [`info`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    /// Contents are not needed. The barrier's old layout is `UNDEFINED`; prior
    /// users are still waited on. Drops contents, not ordering.
    Discard,
    ColorWrite,
    ColorReadWrite,
    DepthWrite,
    DepthReadOnly,
    SampledFragment,
    SampledCompute,
    SampledVertex,
    SampledRayQuery,
    /// Depth aspect read in a shader (depth-stencil read-only layout).
    DepthSampled,
    StorageRead(Stages),
    StorageWrite(Stages),
    StorageReadWrite(Stages),
    IndirectArgs,
    VertexBuffer,
    IndexBuffer,
    Uniform(Stages),
    TransferSrc,
    TransferDst,
    HostRead,
    AccelBuild,
    AccelRead(Stages),
    Present,
}

/// `(stage, access, layout)` for one [`Access`]. Layout is `UNDEFINED` for
/// buffer-only accesses; image `require` ignores those.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AccessInfo {
    pub stages: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
}

impl fmt::Debug for AccessInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccessInfo")
            .field("stages", &self.stages.as_raw())
            .field("access", &self.access.as_raw())
            .field("layout", &self.layout.as_raw())
            .finish()
    }
}

const fn stage_or(a: vk::PipelineStageFlags2, b: vk::PipelineStageFlags2) -> vk::PipelineStageFlags2 {
    vk::PipelineStageFlags2::from_raw(a.as_raw() | b.as_raw())
}

const fn access_or(a: vk::AccessFlags2, b: vk::AccessFlags2) -> vk::AccessFlags2 {
    vk::AccessFlags2::from_raw(a.as_raw() | b.as_raw())
}

const fn stages_or_default(s: Stages, fallback: vk::PipelineStageFlags2) -> vk::PipelineStageFlags2 {
    if s.is_empty() { fallback } else { s.bits() }
}

/// The NVRHI / vk-sync table, in one place.
pub const fn info(a: Access) -> AccessInfo {
    use vk::AccessFlags2 as A;
    use vk::ImageLayout as L;
    use vk::PipelineStageFlags2 as S;
    match a {
        Access::Discard => AccessInfo { stages: S::BOTTOM_OF_PIPE, access: A::empty(), layout: L::UNDEFINED },
        Access::ColorWrite => AccessInfo {
            stages: S::COLOR_ATTACHMENT_OUTPUT,
            access: A::COLOR_ATTACHMENT_WRITE,
            layout: L::COLOR_ATTACHMENT_OPTIMAL,
        },
        Access::ColorReadWrite => AccessInfo {
            stages: S::COLOR_ATTACHMENT_OUTPUT,
            access: access_or(A::COLOR_ATTACHMENT_READ, A::COLOR_ATTACHMENT_WRITE),
            layout: L::COLOR_ATTACHMENT_OPTIMAL,
        },
        Access::DepthWrite => AccessInfo {
            stages: stage_or(S::EARLY_FRAGMENT_TESTS, S::LATE_FRAGMENT_TESTS),
            access: access_or(A::DEPTH_STENCIL_ATTACHMENT_READ, A::DEPTH_STENCIL_ATTACHMENT_WRITE),
            layout: L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
        Access::DepthReadOnly => AccessInfo {
            stages: stage_or(S::EARLY_FRAGMENT_TESTS, S::LATE_FRAGMENT_TESTS),
            access: A::DEPTH_STENCIL_ATTACHMENT_READ,
            layout: L::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        Access::SampledFragment => AccessInfo {
            stages: S::FRAGMENT_SHADER,
            access: A::SHADER_READ,
            layout: L::SHADER_READ_ONLY_OPTIMAL,
        },
        Access::SampledCompute => AccessInfo {
            stages: S::COMPUTE_SHADER,
            access: A::SHADER_READ,
            layout: L::SHADER_READ_ONLY_OPTIMAL,
        },
        Access::SampledVertex => AccessInfo {
            stages: S::VERTEX_SHADER,
            access: A::SHADER_READ,
            layout: L::SHADER_READ_ONLY_OPTIMAL,
        },
        Access::SampledRayQuery => AccessInfo {
            stages: S::RAY_TRACING_SHADER_KHR,
            access: A::SHADER_READ,
            layout: L::SHADER_READ_ONLY_OPTIMAL,
        },
        Access::DepthSampled => AccessInfo {
            stages: S::FRAGMENT_SHADER,
            access: A::SHADER_READ,
            layout: L::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        Access::StorageRead(s) => AccessInfo {
            stages: stages_or_default(s, S::COMPUTE_SHADER),
            access: A::SHADER_READ,
            layout: L::GENERAL,
        },
        Access::StorageWrite(s) => AccessInfo {
            stages: stages_or_default(s, S::COMPUTE_SHADER),
            access: A::SHADER_WRITE,
            layout: L::GENERAL,
        },
        Access::StorageReadWrite(s) => AccessInfo {
            stages: stages_or_default(s, S::COMPUTE_SHADER),
            access: access_or(A::SHADER_READ, A::SHADER_WRITE),
            layout: L::GENERAL,
        },
        Access::IndirectArgs => AccessInfo {
            stages: S::DRAW_INDIRECT,
            access: A::INDIRECT_COMMAND_READ,
            layout: L::UNDEFINED,
        },
        Access::VertexBuffer => AccessInfo {
            stages: S::VERTEX_ATTRIBUTE_INPUT,
            access: A::VERTEX_ATTRIBUTE_READ,
            layout: L::UNDEFINED,
        },
        Access::IndexBuffer => AccessInfo {
            stages: S::INDEX_INPUT,
            access: A::INDEX_READ,
            layout: L::UNDEFINED,
        },
        Access::Uniform(s) => AccessInfo {
            stages: stages_or_default(s, S::ALL_COMMANDS),
            access: A::UNIFORM_READ,
            layout: L::UNDEFINED,
        },
        Access::TransferSrc => AccessInfo {
            stages: S::TRANSFER,
            access: A::TRANSFER_READ,
            layout: L::TRANSFER_SRC_OPTIMAL,
        },
        Access::TransferDst => AccessInfo {
            stages: S::TRANSFER,
            access: A::TRANSFER_WRITE,
            layout: L::TRANSFER_DST_OPTIMAL,
        },
        Access::HostRead => AccessInfo { stages: S::HOST, access: A::HOST_READ, layout: L::UNDEFINED },
        Access::AccelBuild => AccessInfo {
            stages: S::ACCELERATION_STRUCTURE_BUILD_KHR,
            access: access_or(A::ACCELERATION_STRUCTURE_READ_KHR, A::ACCELERATION_STRUCTURE_WRITE_KHR),
            layout: L::UNDEFINED,
        },
        Access::AccelRead(s) => AccessInfo {
            stages: stages_or_default(s, S::RAY_TRACING_SHADER_KHR),
            access: A::ACCELERATION_STRUCTURE_READ_KHR,
            layout: L::UNDEFINED,
        },
        Access::Present => AccessInfo {
            stages: S::ALL_COMMANDS,
            access: A::MEMORY_READ,
            layout: L::PRESENT_SRC_KHR,
        },
    }
}

pub const fn is_write(a: Access) -> bool {
    match a {
        Access::Discard
        | Access::ColorWrite
        | Access::ColorReadWrite
        | Access::DepthWrite
        | Access::StorageWrite(_)
        | Access::StorageReadWrite(_)
        | Access::TransferDst
        | Access::AccelBuild => true,
        Access::DepthReadOnly
        | Access::SampledFragment
        | Access::SampledCompute
        | Access::SampledVertex
        | Access::SampledRayQuery
        | Access::DepthSampled
        | Access::StorageRead(_)
        | Access::IndirectArgs
        | Access::VertexBuffer
        | Access::IndexBuffer
        | Access::Uniform(_)
        | Access::TransferSrc
        | Access::HostRead
        | Access::AccelRead(_)
        | Access::Present => false,
    }
}

/// Buffer-only accesses have no image layout. Using one in [`Tracker::require`] is a no-op.
pub const fn is_buffer_only(a: Access) -> bool {
    match a {
        Access::IndirectArgs
        | Access::VertexBuffer
        | Access::IndexBuffer
        | Access::Uniform(_)
        | Access::HostRead
        | Access::AccelBuild
        | Access::AccelRead(_) => true,
        Access::Discard
        | Access::ColorWrite
        | Access::ColorReadWrite
        | Access::DepthWrite
        | Access::DepthReadOnly
        | Access::SampledFragment
        | Access::SampledCompute
        | Access::SampledVertex
        | Access::SampledRayQuery
        | Access::DepthSampled
        | Access::StorageRead(_)
        | Access::StorageWrite(_)
        | Access::StorageReadWrite(_)
        | Access::TransferSrc
        | Access::TransferDst
        | Access::Present => false,
    }
}

fn is_uav(a: Access) -> bool {
    matches!(a, Access::StorageWrite(_) | Access::StorageReadWrite(_))
}

/// Shader-read plus depth-read resolves to `DEPTH_STENCIL_READ_ONLY_OPTIMAL`.
/// Any other layout pair is a conflict (`None`), except identical layouts and
/// `UNDEFINED` (which yields the other side).
pub fn resolve_layout(a: vk::ImageLayout, b: vk::ImageLayout) -> Option<vk::ImageLayout> {
    let undefined = vk::ImageLayout::UNDEFINED;
    let depth_stencil_ro = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    let depth_ro = vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL;
    let sampled = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    if a == b || a == undefined {
        return Some(b);
    }
    if b == undefined {
        return Some(a);
    }
    let a_depth = a == depth_stencil_ro || a == depth_ro;
    let b_depth = b == depth_stencil_ro || b == depth_ro;
    if (a_depth && b == sampled) || (b_depth && a == sampled) {
        return Some(if a == depth_ro || b == depth_ro { depth_ro } else { depth_stencil_ro });
    }
    None
}

/// Which subresources [`Tracker::require`] applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Range {
    All,
    /// All mips of layers `base..base+count`.
    Layers { base: u32, count: u32 },
    /// All layers of mips `base..base+count`.
    Mips { base: u32, count: u32 },
}

/// How a resource is ordered across frames in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// Indexed by the frame slot. The slot fence retires last frame's use, but
    /// barriers still name the real prior stages so there is one rule set.
    PerSlot,
    /// One instance, reused every frame. Only barriers order it.
    Shared,
    /// Created outside the tracker (swapchain, XR swapchain, egui, CPU-fenced upload).
    Imported,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDesc {
    pub name: &'static str,
    pub aspect: vk::ImageAspectFlags,
    pub mips: u32,
    pub layers: u32,
    pub lifetime: Lifetime,
}

/// `Off`: [`Tracker::observe`] is one branch and returns. `Check`: compare
/// hand-written barriers and log mismatches. `Emit`: the tracker is the source
/// of truth (`require` + `flush`); observe stays quiet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Check,
    Emit,
}

/// Parse `ZESTENGINE_SYNC_CHECK`. Absent means Check in debug builds, Off in release.
/// `panic` is Check plus panic-on-mismatch (for `gpu_tests`, once wired).
pub fn mode_from_env() -> (Mode, bool) {
    mode_from_var(std::env::var("ZESTENGINE_SYNC_CHECK").ok().as_deref())
}

pub fn mode_from_var(val: Option<&str>) -> (Mode, bool) {
    match val.map(str::trim) {
        None => {
            if cfg!(debug_assertions) {
                (Mode::Check, false)
            } else {
                (Mode::Off, false)
            }
        }
        Some("0" | "off" | "false" | "OFF" | "Off") => (Mode::Off, false),
        Some("emit" | "EMIT" | "Emit") => (Mode::Emit, false),
        Some("panic" | "PANIC" | "Panic") => (Mode::Check, true),
        Some(_) => (Mode::Check, false),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MismatchKind {
    OldLayout,
    SourceMissesPriorUse,
    TopOfPipeWithPriorUse,
    BeforeAcquireStage,
    SignalNarrowerThanPresent,
    OwnerConflict,
    PermanentMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub image: vk::Image,
    pub name: &'static str,
    pub kind: MismatchKind,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncStats {
    /// Image barriers plus one if a global memory barrier was drained.
    pub barriers: u64,
    pub mismatches: u64,
    pub observes: u64,
}

/// What [`Tracker::flush`] submits in a single `vkCmdPipelineBarrier2`.
#[derive(Clone, Debug, Default)]
pub struct SyncBatch {
    pub images: Vec<vk::ImageMemoryBarrier2<'static>>,
    pub memory: Option<vk::MemoryBarrier2<'static>>,
}

impl SyncBatch {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.memory.is_none()
    }

    /// `flush` makes one Vulkan call when this is non-empty, and none when it is.
    pub fn submission_count(&self) -> u32 {
        if self.is_empty() { 0 } else { 1 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SubState {
    layout: vk::ImageLayout,
    /// Last writer, including layout transitions. `None` means no prior use.
    write: Option<(vk::PipelineStageFlags2, vk::AccessFlags2)>,
    /// Union of every reader since that write.
    read_stages: vk::PipelineStageFlags2,
    /// Accesses the last write is already visible to.
    visible: vk::AccessFlags2,
    visible_stages: vk::PipelineStageFlags2,
}

impl Default for SubState {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            write: None,
            read_stages: vk::PipelineStageFlags2::empty(),
            visible: vk::AccessFlags2::empty(),
            visible_stages: vk::PipelineStageFlags2::empty(),
        }
    }
}

impl SubState {
    fn prior_stages(self) -> vk::PipelineStageFlags2 {
        let write = self.write.map(|(stage, _)| stage).unwrap_or(vk::PipelineStageFlags2::empty());
        write | self.read_stages
    }

    fn has_prior(self) -> bool {
        self.write.is_some() || !self.read_stages.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct BarrierSpec {
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
}

fn layout_changes(prev: &SubState, next: Access) -> bool {
    if matches!(next, Access::Discard) || is_buffer_only(next) {
        return false;
    }
    let layout = info(next).layout;
    layout != vk::ImageLayout::UNDEFINED && prev.layout != layout
}

/// Pure derivation. `None` means no barrier (the reader is still accumulated by [`apply`]).
fn derive(prev: &SubState, next: Access) -> Option<BarrierSpec> {
    let n = info(next);
    if matches!(next, Access::Discard) {
        if !prev.has_prior() {
            return None;
        }
        return Some(BarrierSpec {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::UNDEFINED,
            src_stage: prev.prior_stages(),
            src_access: prev.write.map(|(_, access)| access).unwrap_or(vk::AccessFlags2::empty()),
            dst_stage: n.stages,
            dst_access: n.access,
        });
    }
    if layout_changes(prev, next) {
        return Some(BarrierSpec {
            old_layout: prev.layout,
            new_layout: n.layout,
            src_stage: prev.prior_stages(),
            src_access: prev.write.map(|(_, access)| access).unwrap_or(vk::AccessFlags2::empty()),
            dst_stage: n.stages,
            dst_access: n.access,
        });
    }
    if is_write(next) {
        if !prev.has_prior() {
            return None;
        }
        let only_war = prev.write.is_none();
        return Some(BarrierSpec {
            old_layout: prev.layout,
            new_layout: prev.layout,
            src_stage: prev.prior_stages(),
            src_access: if only_war {
                vk::AccessFlags2::empty()
            } else {
                prev.write.map(|(_, access)| access).unwrap_or(vk::AccessFlags2::empty())
            },
            dst_stage: n.stages,
            dst_access: n.access,
        });
    }
    if prev.visible_stages.contains(n.stages) && prev.visible.contains(n.access) {
        return None;
    }
    let (src_stage, src_access) = prev.write?;
    Some(BarrierSpec {
        old_layout: prev.layout,
        new_layout: prev.layout,
        src_stage,
        src_access,
        dst_stage: n.stages,
        dst_access: n.access,
    })
}

fn apply(state: &mut SubState, next: Access) {
    let n = info(next);
    if matches!(next, Access::Discard) {
        *state = SubState {
            layout: vk::ImageLayout::UNDEFINED,
            write: Some((n.stages, n.access)),
            ..SubState::default()
        };
        return;
    }
    let writing = is_write(next);
    if writing || layout_changes(state, next) {
        if n.layout != vk::ImageLayout::UNDEFINED {
            state.layout = n.layout;
        }
        state.write = Some((n.stages, n.access));
        if writing {
            state.read_stages = vk::PipelineStageFlags2::empty();
            state.visible = vk::AccessFlags2::empty();
            state.visible_stages = vk::PipelineStageFlags2::empty();
        } else {
            state.read_stages = n.stages;
            state.visible = n.access;
            state.visible_stages = n.stages;
        }
        return;
    }
    state.read_stages |= n.stages;
    state.visible |= n.access;
    state.visible_stages |= n.stages;
}

fn transition(prev: &SubState, next: Access) -> (SubState, Option<BarrierSpec>) {
    let spec = derive(prev, next);
    let mut state = *prev;
    apply(&mut state, next);
    (state, spec)
}

/// `ALL_COMMANDS` is its own bit, not the union of the other stage bits.
fn stage_covers(have: vk::PipelineStageFlags2, need: vk::PipelineStageFlags2) -> bool {
    if need.is_empty() || have.contains(vk::PipelineStageFlags2::ALL_COMMANDS) {
        return true;
    }
    if need.contains(vk::PipelineStageFlags2::ALL_COMMANDS) {
        return have.contains(vk::PipelineStageFlags2::ALL_COMMANDS);
    }
    have.contains(need)
}

fn is_vacuous_src(src: vk::PipelineStageFlags2) -> bool {
    src.is_empty() || src == vk::PipelineStageFlags2::TOP_OF_PIPE
}

struct ImageTrack {
    desc: ImageDesc,
    compact: SubState,
    /// `Some` once a partial range was required. Collapses when every
    /// subresource is identical again (a later full-image require).
    expanded: Option<Vec<SubState>>,
    permanent: Option<Access>,
    /// Acquire-semaphore wait stage for an imported image. The first transition
    /// must include it (swapchain bug 1).
    acquire_stage: Option<vk::PipelineStageFlags2>,
    /// Opt-in feature tag. A write under a different tag, after readers, is bug 5.
    owner: Option<&'static str>,
    writer_owner: Option<&'static str>,
    uav_barriers: bool,
    first_uav_placed: bool,
}

impl ImageTrack {
    fn fresh(desc: ImageDesc, state: SubState, acquire: Option<vk::PipelineStageFlags2>) -> Self {
        Self {
            desc,
            compact: state,
            expanded: None,
            permanent: None,
            acquire_stage: acquire,
            owner: None,
            writer_owner: None,
            uav_barriers: true,
            first_uav_placed: false,
        }
    }

    fn mips(&self) -> u32 {
        self.desc.mips.max(1)
    }

    fn layers(&self) -> u32 {
        self.desc.layers.max(1)
    }

    fn index(&self, mip: u32, layer: u32) -> usize {
        mip as usize + layer as usize * self.mips() as usize
    }

    fn expand(&mut self) {
        if self.expanded.is_some() {
            return;
        }
        let n = self.mips() as usize * self.layers() as usize;
        self.expanded = Some(vec![self.compact; n]);
    }

    fn try_collapse(&mut self) {
        let Some(subs) = &self.expanded else { return };
        let Some(first) = subs.first().copied() else { return };
        if subs.iter().all(|s| *s == first) {
            self.compact = first;
            self.expanded = None;
        }
    }

    fn state(&self, mip: u32, layer: u32) -> SubState {
        match &self.expanded {
            Some(subs) => subs[self.index(mip, layer)],
            None => self.compact,
        }
    }

    fn state_mut(&mut self, mip: u32, layer: u32) -> &mut SubState {
        let index = self.index(mip, layer);
        &mut self.expanded.as_mut().expect("expanded before per-subresource write")[index]
    }

    fn any_present(&self) -> bool {
        let present = vk::ImageLayout::PRESENT_SRC_KHR;
        match &self.expanded {
            Some(subs) => subs.iter().any(|s| s.layout == present),
            None => self.compact.layout == present,
        }
    }

    fn write_stage(&self) -> Option<vk::PipelineStageFlags2> {
        let mut acc = vk::PipelineStageFlags2::empty();
        let mut any = false;
        let fold = |state: SubState, acc: &mut vk::PipelineStageFlags2, any: &mut bool| {
            if let Some((stage, _)) = state.write {
                *acc |= stage;
                *any = true;
            }
        };
        match &self.expanded {
            Some(subs) => {
                for state in subs {
                    fold(*state, &mut acc, &mut any);
                }
            }
            None => fold(self.compact, &mut acc, &mut any),
        }
        any.then_some(acc)
    }

    fn has_readers(&self) -> bool {
        match &self.expanded {
            Some(subs) => subs.iter().any(|s| !s.read_stages.is_empty()),
            None => !self.compact.read_stages.is_empty(),
        }
    }
}

struct BufferTrack {
    state: SubState,
    uav_barriers: bool,
    first_uav_placed: bool,
}

impl Default for BufferTrack {
    fn default() -> Self {
        Self { state: SubState::default(), uav_barriers: true, first_uav_placed: false }
    }
}

const MISMATCH_CAP: usize = 64;

/// Per-queue resource-state tracker.
///
/// Keyed by Vulkan handle so existing images and buffers need no new id.
/// `std::HashMap` rather than `FxHashMap`: `rustc-hash` is not a dependency of
/// this crate, and a couple of hundred lookups a frame is noise.
pub struct Tracker {
    images: HashMap<vk::Image, ImageTrack>,
    buffers: HashMap<vk::Buffer, BufferTrack>,
    pending_img: Vec<vk::ImageMemoryBarrier2<'static>>,
    pending_mem: Option<vk::MemoryBarrier2<'static>>,
    mode: Mode,
    panic_on_mismatch: bool,
    stats: SyncStats,
    mismatches: Vec<Mismatch>,
    warned: HashSet<(u64, u8)>,
    unknown_warned: HashSet<u64>,
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracker {
    /// Mode from `ZESTENGINE_SYNC_CHECK` (Check in debug, Off in release, unless set).
    pub fn new() -> Self {
        let (mode, panic_on_mismatch) = mode_from_env();
        Self::build(mode, panic_on_mismatch)
    }

    pub fn with_mode(mode: Mode) -> Self {
        Self::build(mode, false)
    }

    /// Check mode that panics on the first mismatch. `ZESTENGINE_SYNC_CHECK=panic`.
    pub fn panicking() -> Self {
        Self::build(Mode::Check, true)
    }

    fn build(mode: Mode, panic_on_mismatch: bool) -> Self {
        Self {
            images: HashMap::new(),
            buffers: HashMap::new(),
            pending_img: Vec::new(),
            pending_mem: None,
            mode,
            panic_on_mismatch,
            stats: SyncStats::default(),
            mismatches: Vec::new(),
            warned: HashSet::new(),
            unknown_warned: HashSet::new(),
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn stats(&self) -> SyncStats {
        self.stats
    }

    pub fn mismatches(&self) -> &[Mismatch] {
        &self.mismatches
    }

    pub fn take_mismatches(&mut self) -> Vec<Mismatch> {
        std::mem::take(&mut self.mismatches)
    }

    pub fn register_image(&mut self, image: vk::Image, desc: ImageDesc) {
        self.forget_image(image);
        self.images.insert(image, ImageTrack::fresh(desc, SubState::default(), None));
    }

    pub fn forget_image(&mut self, image: vk::Image) {
        self.images.remove(&image);
        self.pending_img.retain(|barrier| barrier.image != image);
    }

    /// Imported with a known prior state (NVRHI `beginTracking`).
    /// `after` is the stage the previous owner released at — for a swapchain
    /// image, the acquire semaphore's wait stage.
    pub fn import_image(&mut self, image: vk::Image, desc: ImageDesc, layout: vk::ImageLayout, after: vk::PipelineStageFlags2) {
        self.forget_image(image);
        let mut state = SubState { layout, ..Default::default() };
        let acquire = if after.is_empty() { None } else { Some(after) };
        if let Some(stage) = acquire {
            state.write = Some((stage, vk::AccessFlags2::empty()));
        }
        self.images.insert(image, ImageTrack::fresh(desc, state, acquire));
    }

    /// Fix the image in one access. Later `require` calls only verify; they never emit.
    pub fn set_permanent(&mut self, image: vk::Image, access: Access) {
        let Some(track) = self.images.get_mut(&image) else {
            self.warn_unknown(image.as_raw(), "set_permanent");
            return;
        };
        let n = info(access);
        track.expanded = None;
        track.compact = SubState {
            layout: n.layout,
            write: Some((n.stages, n.access)),
            read_stages: if is_write(access) { vk::PipelineStageFlags2::empty() } else { n.stages },
            visible: n.access,
            visible_stages: n.stages,
        };
        track.permanent = Some(access);
    }

    /// Opt-in owner tag (bug 5). A write under a different tag, after readers, is logged in check mode.
    pub fn set_owner(&mut self, image: vk::Image, owner: &'static str) {
        if let Some(track) = self.images.get_mut(&image) {
            track.owner = Some(owner);
        }
    }

    /// Skip UAV-to-UAV barriers after the first UAV use. The first use still barriers.
    /// Disjoint dispatches only; a write-after-read is never skipped.
    pub fn set_image_uav_barriers(&mut self, image: vk::Image, enabled: bool) {
        if let Some(track) = self.images.get_mut(&image) {
            track.uav_barriers = enabled;
        }
    }

    pub fn set_buffer_uav_barriers(&mut self, buffer: vk::Buffer, enabled: bool) {
        self.buffers.entry(buffer).or_default().uav_barriers = enabled;
    }

    pub fn is_expanded(&self, image: vk::Image) -> bool {
        self.images.get(&image).is_some_and(|track| track.expanded.is_some())
    }

    pub fn require(&mut self, image: vk::Image, range: Range, access: Access) {
        if is_buffer_only(access) {
            self.warn_unknown(image.as_raw(), "require of a buffer-only access");
            return;
        }
        if !self.images.contains_key(&image) {
            self.warn_unknown(image.as_raw(), "require");
            return;
        }
        if self.images[&image].permanent.is_some() {
            self.verify_permanent(image, access);
            return;
        }
        self.note_owner(image, access);
        let (base_mip, mip_count, base_layer, layer_count) = self.clamp_range(image, range);
        if mip_count == 0 || layer_count == 0 {
            return;
        }
        let whole = matches!(range, Range::All) && self.images[&image].expanded.is_none();
        if whole {
            self.require_compact(image, access);
        } else {
            self.require_range(image, access, base_mip, mip_count, base_layer, layer_count);
        }
    }

    pub fn require_buffer(&mut self, buffer: vk::Buffer, access: Access) {
        self.buffers.entry(buffer).or_default();
        let track = &self.buffers[&buffer];
        if should_skip_uav(&track.state, track.uav_barriers, track.first_uav_placed, access) {
            let n = info(access);
            let track = self.buffers.get_mut(&buffer).expect("buffer inserted above");
            if let Some((stage, acc)) = track.state.write {
                track.state.write = Some((stage | n.stages, acc | n.access));
            }
            return;
        }
        let prev = self.buffers[&buffer].state;
        let (new_state, spec) = transition(&prev, access);
        if let Some(spec) = spec {
            self.queue_mem(&spec);
        }
        let track = self.buffers.get_mut(&buffer).expect("buffer inserted above");
        track.state = new_state;
        if is_uav(access) {
            track.first_uav_placed = true;
        }
    }

    /// Stage the submit's signal semaphore must cover.
    /// A layout transition to `PRESENT_SRC` needs `ALL_COMMANDS` (bug 2).
    pub fn present_signal_stage(&self, image: vk::Image) -> vk::PipelineStageFlags2 {
        let Some(track) = self.images.get(&image) else {
            return vk::PipelineStageFlags2::ALL_COMMANDS;
        };
        if track.any_present() {
            vk::PipelineStageFlags2::ALL_COMMANDS
        } else {
            track.write_stage().unwrap_or(vk::PipelineStageFlags2::ALL_COMMANDS)
        }
    }

    /// Compare a hand-written barrier with what `require(next)` would derive,
    /// then adopt the hand barrier's destination so later checks follow the GPU.
    /// No-op in [`Mode::Off`] (one branch) and for images this tracker does not know.
    pub fn observe(&mut self, barrier: &vk::ImageMemoryBarrier2<'_>, next: Access) {
        if self.mode != Mode::Check {
            return;
        }
        self.stats.observes += 1;
        let image = barrier.image;
        if !self.images.contains_key(&image) {
            return;
        }
        let (base_mip, mip_count, base_layer, layer_count) = self.observe_span(image, barrier);
        self.check_barrier(image, barrier, next, base_mip, mip_count, base_layer, layer_count);
        self.adopt_barrier(image, barrier, next, base_mip, mip_count, base_layer, layer_count);
    }

    /// Check-mode hook for the submit signal (bug 2). No-op unless the image is in `PRESENT_SRC`.
    pub fn observe_signal(&mut self, image: vk::Image, signal: vk::PipelineStageFlags2) {
        if self.mode != Mode::Check {
            return;
        }
        let Some(track) = self.images.get(&image) else { return };
        if !track.any_present() {
            return;
        }
        let name = track.desc.name;
        if !stage_covers(signal, vk::PipelineStageFlags2::ALL_COMMANDS) {
            self.flag(
                image,
                name,
                MismatchKind::SignalNarrowerThanPresent,
                format!("signal 0x{:x} does not cover ALL_COMMANDS after a present transition", signal.as_raw()),
            );
        }
    }

    /// Drop pending barriers into the structs [`flush`](Self::flush) submits.
    /// Tests assert these. Does not call Vulkan.
    pub fn drain(&mut self) -> SyncBatch {
        let images = std::mem::take(&mut self.pending_img);
        let memory = self.pending_mem.take();
        if !images.is_empty() || memory.is_some() {
            self.stats.barriers += images.len() as u64 + u64::from(memory.is_some());
        }
        SyncBatch { images, memory }
    }

    /// Submit every pending barrier as one sync2 pipeline barrier.
    ///
    /// `cb` must be recording, and this must be outside a dynamic rendering pass.
    /// No-op when nothing is pending (no Vulkan call).
    pub fn flush(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        let batch = self.drain();
        if batch.is_empty() {
            return;
        }
        let memory = batch.memory.map(|barrier| [barrier]);
        let memory = memory.as_ref().map(|barrier| barrier.as_slice()).unwrap_or(&[]);
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&batch.images).memory_barriers(memory);
        unsafe { device.cmd_pipeline_barrier2(cb, &dependency) };
    }

    fn require_compact(&mut self, image: vk::Image, access: Access) {
        let (skip, prev, desc, mips, layers) = {
            let track = self.images.get(&image).expect("image checked present");
            (
                should_skip_uav(&track.compact, track.uav_barriers, track.first_uav_placed, access),
                track.compact,
                track.desc,
                track.mips(),
                track.layers(),
            )
        };
        if skip {
            let n = info(access);
            let track = self.images.get_mut(&image).expect("image checked present");
            if let Some((stage, acc)) = track.compact.write {
                track.compact.write = Some((stage | n.stages, acc | n.access));
            }
            return;
        }
        let (new_state, spec) = transition(&prev, access);
        {
            let track = self.images.get_mut(&image).expect("image checked present");
            if is_uav(access) {
                track.first_uav_placed = true;
            }
            track.compact = new_state;
        }
        if let Some(spec) = spec {
            self.push_image(image, desc, spec, 0, mips, 0, layers);
        }
    }

    fn require_range(&mut self, image: vk::Image, access: Access, base_mip: u32, mip_count: u32, base_layer: u32, layer_count: u32) {
        {
            let track = self.images.get_mut(&image).expect("image checked present");
            track.expand();
        }
        let mut specs: Vec<(u32, u32, BarrierSpec)> = Vec::new();
        for layer in base_layer..base_layer + layer_count {
            for mip in base_mip..base_mip + mip_count {
                let (prev, skip) = {
                    let track = self.images.get(&image).expect("image checked present");
                    let prev = track.state(mip, layer);
                    (prev, should_skip_uav(&prev, track.uav_barriers, track.first_uav_placed, access))
                };
                if skip {
                    let n = info(access);
                    let track = self.images.get_mut(&image).expect("image checked present");
                    let state = track.state_mut(mip, layer);
                    if let Some((stage, acc)) = state.write {
                        state.write = Some((stage | n.stages, acc | n.access));
                    }
                    continue;
                }
                let (new_state, spec) = transition(&prev, access);
                {
                    let track = self.images.get_mut(&image).expect("image checked present");
                    if is_uav(access) {
                        track.first_uav_placed = true;
                    }
                    *track.state_mut(mip, layer) = new_state;
                }
                if let Some(spec) = spec {
                    specs.push((mip, layer, spec));
                }
            }
        }
        let desc = self.images[&image].desc;
        self.push_coalesced(image, desc, &specs);
        if let Some(track) = self.images.get_mut(&image) {
            track.try_collapse();
        }
    }

    fn push_coalesced(&mut self, image: vk::Image, desc: ImageDesc, specs: &[(u32, u32, BarrierSpec)]) {
        let mut groups: Vec<(BarrierSpec, Vec<(u32, u32)>)> = Vec::new();
        for &(mip, layer, spec) in specs {
            if let Some((_, cells)) = groups.iter_mut().find(|(have, _)| *have == spec) {
                cells.push((mip, layer));
            } else {
                groups.push((spec, vec![(mip, layer)]));
            }
        }
        for (spec, cells) in groups {
            if let Some((base_mip, mip_count, base_layer, layer_count)) = rectangle(&cells) {
                self.push_image(image, desc, spec, base_mip, mip_count, base_layer, layer_count);
            } else {
                for (mip, layer) in cells {
                    self.push_image(image, desc, spec, mip, 1, layer, 1);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_image(&mut self, image: vk::Image, desc: ImageDesc, spec: BarrierSpec, base_mip: u32, mip_count: u32, base_layer: u32, layer_count: u32) {
        let barrier = vk::ImageMemoryBarrier2::default()
            .image(image)
            .old_layout(spec.old_layout)
            .new_layout(spec.new_layout)
            .src_stage_mask(spec.src_stage)
            .src_access_mask(spec.src_access)
            .dst_stage_mask(spec.dst_stage)
            .dst_access_mask(spec.dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(desc.aspect)
                    .base_mip_level(base_mip)
                    .level_count(mip_count)
                    .base_array_layer(base_layer)
                    .layer_count(layer_count),
            );
        self.pending_img.push(barrier);
    }

    fn queue_mem(&mut self, spec: &BarrierSpec) {
        if spec.src_stage.is_empty() && spec.dst_stage.is_empty() {
            return;
        }
        match &mut self.pending_mem {
            Some(barrier) => {
                barrier.src_stage_mask |= spec.src_stage;
                barrier.src_access_mask |= spec.src_access;
                barrier.dst_stage_mask |= spec.dst_stage;
                barrier.dst_access_mask |= spec.dst_access;
            }
            None => {
                self.pending_mem = Some(
                    vk::MemoryBarrier2::default()
                        .src_stage_mask(spec.src_stage)
                        .src_access_mask(spec.src_access)
                        .dst_stage_mask(spec.dst_stage)
                        .dst_access_mask(spec.dst_access),
                );
            }
        }
    }

    fn clamp_range(&self, image: vk::Image, range: Range) -> (u32, u32, u32, u32) {
        let track = &self.images[&image];
        let mips = track.mips();
        let layers = track.layers();
        match range {
            Range::All => (0, mips, 0, layers),
            Range::Mips { base, count } => {
                let base = base.min(mips);
                (base, count.min(mips - base), 0, layers)
            }
            Range::Layers { base, count } => {
                let base = base.min(layers);
                (0, mips, base, count.min(layers - base))
            }
        }
    }

    fn verify_permanent(&mut self, image: vk::Image, access: Access) {
        let track = &self.images[&image];
        let Some(permanent) = track.permanent else { return };
        let name = track.desc.name;
        let want = info(permanent).layout;
        let got = info(access).layout;
        let layout_ok = resolve_layout(want, got) == Some(want) || want == got;
        let write_ok = !(is_write(access) && !is_write(permanent));
        if layout_ok && write_ok {
            return;
        }
        self.flag(
            image,
            name,
            MismatchKind::PermanentMismatch,
            format!("permanent layout {} rejected require layout {}", want.as_raw(), got.as_raw()),
        );
    }

    fn note_owner(&mut self, image: vk::Image, access: Access) {
        if self.mode != Mode::Check || !is_write(access) {
            return;
        }
        let Some(track) = self.images.get(&image) else { return };
        let Some(owner) = track.owner else { return };
        let conflict = track.writer_owner.is_some_and(|prev| prev != owner) && track.has_readers();
        let name = track.desc.name;
        let prev = track.writer_owner.unwrap_or("");
        if conflict {
            self.flag(
                image,
                name,
                MismatchKind::OwnerConflict,
                format!("owner {owner} writes while {prev} still has readers"),
            );
        }
        if let Some(track) = self.images.get_mut(&image) {
            track.writer_owner = Some(owner);
        }
    }

    fn observe_span(&self, image: vk::Image, barrier: &vk::ImageMemoryBarrier2<'_>) -> (u32, u32, u32, u32) {
        let track = &self.images[&image];
        let range = barrier.subresource_range;
        if range.level_count == 0 && range.layer_count == 0 {
            return (0, track.mips(), 0, track.layers());
        }
        let mip_count = if range.level_count == vk::REMAINING_MIP_LEVELS {
            track.mips().saturating_sub(range.base_mip_level)
        } else {
            range.level_count
        };
        let layer_count = if range.layer_count == vk::REMAINING_ARRAY_LAYERS {
            track.layers().saturating_sub(range.base_array_layer)
        } else {
            range.layer_count
        };
        (range.base_mip_level, mip_count, range.base_array_layer, layer_count)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_barrier(
        &mut self,
        image: vk::Image,
        barrier: &vk::ImageMemoryBarrier2<'_>,
        next: Access,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
        layer_count: u32,
    ) {
        let track = &self.images[&image];
        let name = track.desc.name;
        let lifetime = track.desc.lifetime;
        let acquire = track.acquire_stage;
        let strict = lifetime != Lifetime::PerSlot;
        let need_readers = is_write(next) || barrier.old_layout != barrier.new_layout || matches!(next, Access::Discard);
        let mut required = vk::PipelineStageFlags2::empty();
        let mut has_prior = false;
        let mut layout_mismatch = false;
        let mut acquire_layout = false;
        for layer in base_layer..base_layer.saturating_add(layer_count).min(track.layers()) {
            for mip in base_mip..base_mip.saturating_add(mip_count).min(track.mips()) {
                let state = track.state(mip, layer);
                if barrier.old_layout != state.layout && barrier.old_layout != vk::ImageLayout::UNDEFINED {
                    layout_mismatch = true;
                }
                if state.has_prior() {
                    has_prior = true;
                }
                if state.layout == vk::ImageLayout::UNDEFINED && acquire.is_some() {
                    acquire_layout = true;
                }
                if need_readers {
                    required |= state.prior_stages();
                } else if let Some((stage, _)) = state.write {
                    required |= stage;
                }
            }
        }
        let src = barrier.src_stage_mask;
        if layout_mismatch {
            self.flag(
                image,
                name,
                MismatchKind::OldLayout,
                format!("old layout {} is neither tracked nor UNDEFINED", barrier.old_layout.as_raw()),
            );
        }
        if strict && has_prior && !stage_covers(src, required) {
            self.flag(
                image,
                name,
                MismatchKind::SourceMissesPriorUse,
                format!("src 0x{:x} misses prior 0x{:x}", src.as_raw(), required.as_raw()),
            );
        }
        if strict && has_prior && is_vacuous_src(src) {
            self.flag(
                image,
                name,
                MismatchKind::TopOfPipeWithPriorUse,
                format!("src 0x{:x} waits for nothing", src.as_raw()),
            );
        }
        if acquire_layout && let Some(acquire) = acquire && !stage_covers(src, acquire) {
            self.flag(
                image,
                name,
                MismatchKind::BeforeAcquireStage,
                format!("src 0x{:x} misses acquire stage 0x{:x}", src.as_raw(), acquire.as_raw()),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn adopt_barrier(
        &mut self,
        image: vk::Image,
        barrier: &vk::ImageMemoryBarrier2<'_>,
        next: Access,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
        layer_count: u32,
    ) {
        let whole = {
            let track = &self.images[&image];
            track.expanded.is_none() && base_mip == 0 && base_layer == 0 && mip_count >= track.mips() && layer_count >= track.layers()
        };
        if !whole
            && let Some(track) = self.images.get_mut(&image) {
                track.expand();
            }
        let writing = is_write(next) || barrier.old_layout != barrier.new_layout;
        let apply_one = |state: &mut SubState| {
            state.layout = barrier.new_layout;
            if writing && is_write(next) {
                state.write = Some((barrier.dst_stage_mask, barrier.dst_access_mask));
                state.read_stages = vk::PipelineStageFlags2::empty();
                state.visible = vk::AccessFlags2::empty();
                state.visible_stages = vk::PipelineStageFlags2::empty();
            } else if barrier.old_layout != barrier.new_layout {
                state.write = Some((barrier.dst_stage_mask, barrier.dst_access_mask));
                if is_write(next) {
                    state.read_stages = vk::PipelineStageFlags2::empty();
                    state.visible = vk::AccessFlags2::empty();
                    state.visible_stages = vk::PipelineStageFlags2::empty();
                } else {
                    state.read_stages = barrier.dst_stage_mask;
                    state.visible = barrier.dst_access_mask;
                    state.visible_stages = barrier.dst_stage_mask;
                }
            } else if !is_write(next) {
                state.read_stages |= barrier.dst_stage_mask;
                state.visible |= barrier.dst_access_mask;
                state.visible_stages |= barrier.dst_stage_mask;
            } else {
                state.write = Some((barrier.dst_stage_mask, barrier.dst_access_mask));
                state.read_stages = vk::PipelineStageFlags2::empty();
                state.visible = vk::AccessFlags2::empty();
                state.visible_stages = vk::PipelineStageFlags2::empty();
            }
        };
        let Some(track) = self.images.get_mut(&image) else { return };
        if whole && track.expanded.is_none() {
            apply_one(&mut track.compact);
            return;
        }
        let mips = track.mips();
        let layers = track.layers();
        for layer in base_layer..base_layer.saturating_add(layer_count).min(layers) {
            for mip in base_mip..base_mip.saturating_add(mip_count).min(mips) {
                apply_one(track.state_mut(mip, layer));
            }
        }
        track.try_collapse();
    }

    fn flag(&mut self, image: vk::Image, name: &'static str, kind: MismatchKind, detail: String) {
        self.stats.mismatches += 1;
        let key = (image.as_raw(), kind as u8);
        if self.warned.insert(key) {
            tracing::warn!(target: "sync", image = name, kind = ?kind, "{detail}");
        }
        if self.mismatches.len() < MISMATCH_CAP {
            self.mismatches.push(Mismatch { image, name, kind, detail: detail.clone() });
        }
        if self.panic_on_mismatch {
            panic!("sync check: {name}: {kind:?}: {detail}");
        }
    }

    fn warn_unknown(&mut self, raw: u64, what: &str) {
        if self.unknown_warned.insert(raw) {
            tracing::warn!(target: "sync", "sync tracker: {what} on unregistered handle 0x{raw:x}");
        }
    }
}

fn should_skip_uav(state: &SubState, uav_barriers: bool, first_placed: bool, access: Access) -> bool {
    is_uav(access) && !uav_barriers && first_placed && is_write(access) && state.read_stages.is_empty() && state.write.is_some() && !layout_changes(state, access)
}

fn rectangle(cells: &[(u32, u32)]) -> Option<(u32, u32, u32, u32)> {
    let mut min_mip = u32::MAX;
    let mut max_mip = 0;
    let mut min_layer = u32::MAX;
    let mut max_layer = 0;
    for &(mip, layer) in cells {
        min_mip = min_mip.min(mip);
        max_mip = max_mip.max(mip);
        min_layer = min_layer.min(layer);
        max_layer = max_layer.max(layer);
    }
    let mip_count = max_mip - min_mip + 1;
    let layer_count = max_layer - min_layer + 1;
    if cells.len() == mip_count as usize * layer_count as usize {
        Some((min_mip, mip_count, min_layer, layer_count))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(n: u64) -> vk::Image {
        vk::Image::from_raw(n)
    }

    fn buf(n: u64) -> vk::Buffer {
        vk::Buffer::from_raw(n)
    }

    fn desc(lifetime: Lifetime, mips: u32, layers: u32) -> ImageDesc {
        ImageDesc { name: "test-image", aspect: vk::ImageAspectFlags::COLOR, mips, layers, lifetime }
    }

    fn depth_desc(lifetime: Lifetime) -> ImageDesc {
        ImageDesc { name: "test-depth", aspect: vk::ImageAspectFlags::DEPTH, mips: 1, layers: 1, lifetime }
    }

    fn tracker() -> Tracker {
        Tracker::with_mode(Mode::Emit)
    }

    fn one_image(batch: &SyncBatch) -> vk::ImageMemoryBarrier2<'static> {
        assert!(batch.memory.is_none(), "unexpected memory barrier");
        assert_eq!(batch.images.len(), 1, "expected one image barrier, got {}", batch.images.len());
        batch.images[0]
    }

    fn kinds(tracker: &Tracker) -> Vec<MismatchKind> {
        tracker.mismatches().iter().map(|m| m.kind).collect()
    }

    fn hand_barrier(
        image: vk::Image,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        src: vk::PipelineStageFlags2,
        dst: vk::PipelineStageFlags2,
    ) -> vk::ImageMemoryBarrier2<'static> {
        vk::ImageMemoryBarrier2::default()
            .image(image)
            .old_layout(old)
            .new_layout(new)
            .src_stage_mask(src)
            .dst_stage_mask(dst)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_access_mask(vk::AccessFlags2::empty())
            .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1))
    }

    #[test]
    fn info_table_no_undefined_layout_and_depth_sampled_resolves() {
        let image_accesses = [
            Access::ColorWrite,
            Access::ColorReadWrite,
            Access::DepthWrite,
            Access::DepthReadOnly,
            Access::SampledFragment,
            Access::SampledCompute,
            Access::SampledVertex,
            Access::SampledRayQuery,
            Access::DepthSampled,
            Access::StorageRead(Stages::COMPUTE),
            Access::StorageWrite(Stages::COMPUTE),
            Access::StorageReadWrite(Stages::FRAGMENT),
            Access::TransferSrc,
            Access::TransferDst,
            Access::Present,
        ];
        for access in image_accesses {
            assert!(!is_buffer_only(access));
            assert_ne!(info(access).layout, vk::ImageLayout::UNDEFINED, "{access:?}");
        }
        assert_eq!(info(Access::Discard).layout, vk::ImageLayout::UNDEFINED);
        assert!(is_write(Access::ColorWrite));
        assert!(!is_write(Access::SampledCompute));
        assert_eq!(info(Access::ColorWrite).stages, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        assert_eq!(info(Access::ColorWrite).access, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
        assert_eq!(info(Access::Present).stages, vk::PipelineStageFlags2::ALL_COMMANDS);
        assert_eq!(info(Access::Present).layout, vk::ImageLayout::PRESENT_SRC_KHR);
        assert_eq!(info(Access::SampledCompute).stages, vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert_eq!(info(Access::DepthReadOnly).layout, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        assert_eq!(info(Access::DepthSampled).layout, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        assert_eq!(
            resolve_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            Some(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL),
        );
        assert_eq!(
            resolve_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL),
            Some(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL),
        );
        assert!(resolve_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).is_none());
        let _ = depth_desc(Lifetime::Shared);
    }

    #[test]
    fn acquired_swapchain_waits_on_semaphore_stage() {
        let mut t = tracker();
        let image = img(1);
        t.import_image(image, desc(Lifetime::Imported, 1, 1), vk::ImageLayout::UNDEFINED, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        t.require(image, Range::All, Access::ColorWrite);
        let barrier = one_image(&t.drain());
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barrier.new_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert!(barrier.src_stage_mask.contains(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT), "src 0x{:x}", barrier.src_stage_mask.as_raw());
        assert_ne!(barrier.src_stage_mask, vk::PipelineStageFlags2::TOP_OF_PIPE);
        assert_eq!(barrier.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
    }

    #[test]
    fn present_needs_all_commands_signal() {
        let mut t = tracker();
        let image = img(2);
        t.import_image(image, desc(Lifetime::Imported, 1, 1), vk::ImageLayout::UNDEFINED, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        t.require(image, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.require(image, Range::All, Access::Present);
        let barrier = one_image(&t.drain());
        assert_eq!(barrier.new_layout, vk::ImageLayout::PRESENT_SRC_KHR);
        assert_eq!(barrier.dst_stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
        assert!(barrier.src_stage_mask.contains(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT));
        assert_eq!(t.present_signal_stage(image), vk::PipelineStageFlags2::ALL_COMMANDS);
    }

    #[test]
    fn discard_after_compute_read_waits_for_compute() {
        let mut t = tracker();
        let image = img(3);
        t.register_image(image, desc(Lifetime::Shared, 1, 1));
        t.require(image, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledFragment);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledCompute);
        let _ = t.drain();
        t.require(image, Range::All, Access::Discard);
        let barrier = one_image(&t.drain());
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED, "discard drops contents");
        assert!(
            barrier.src_stage_mask.contains(vk::PipelineStageFlags2::COMPUTE_SHADER),
            "src 0x{:x} dropped the compute reader",
            barrier.src_stage_mask.as_raw(),
        );
        assert!(barrier.src_stage_mask.contains(vk::PipelineStageFlags2::FRAGMENT_SHADER));
    }

    #[test]
    fn shared_target_reuse_waits_for_last_frame_readers() {
        let mut t = tracker();
        let image = img(4);
        t.register_image(image, desc(Lifetime::Shared, 1, 1));
        t.require(image, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledFragment);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledCompute);
        let _ = t.drain();
        t.require(image, Range::All, Access::ColorWrite);
        let barrier = one_image(&t.drain());
        assert_ne!(barrier.src_stage_mask, vk::PipelineStageFlags2::TOP_OF_PIPE);
        assert!(!barrier.src_stage_mask.is_empty());
        assert!(barrier.src_stage_mask.contains(vk::PipelineStageFlags2::COMPUTE_SHADER), "src 0x{:x}", barrier.src_stage_mask.as_raw());
        assert!(barrier.src_stage_mask.contains(vk::PipelineStageFlags2::FRAGMENT_SHADER));
        assert_eq!(barrier.new_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    #[test]
    fn second_reader_in_new_stage_gets_visibility() {
        let mut t = tracker();
        let image = img(5);
        t.register_image(image, desc(Lifetime::Shared, 1, 1));
        t.require(image, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledFragment);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledCompute);
        let barrier = one_image(&t.drain());
        assert!(barrier.dst_stage_mask.contains(vk::PipelineStageFlags2::COMPUTE_SHADER));
        assert!(barrier.src_stage_mask.contains(vk::PipelineStageFlags2::FRAGMENT_SHADER));
        t.require(image, Range::All, Access::SampledCompute);
        assert!(t.drain().is_empty(), "a repeat reader in a stage already made visible needs no barrier");
    }

    #[test]
    fn per_layer_require_expands_then_full_collapses() {
        let mut t = tracker();
        let image = img(6);
        t.register_image(image, desc(Lifetime::Shared, 1, 4));
        assert!(!t.is_expanded(image));
        t.require(image, Range::Layers { base: 0, count: 1 }, Access::ColorWrite);
        assert!(t.is_expanded(image), "a partial require expands to per-subresource state");
        let first = t.drain();
        assert_eq!(first.images.len(), 1);
        assert_eq!(first.images[0].subresource_range.layer_count, 1);
        t.require(image, Range::All, Access::SampledFragment);
        assert!(!t.is_expanded(image), "a full require that unifies subresources collapses");
        let batch = t.drain();
        assert!(!batch.images.is_empty());
        assert!(batch.images.iter().all(|b| b.new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
    }

    #[test]
    fn full_image_require_is_one_barrier() {
        let mut t = tracker();
        let image = img(7);
        t.register_image(image, desc(Lifetime::PerSlot, 4, 6));
        t.require(image, Range::All, Access::ColorWrite);
        let barrier = one_image(&t.drain());
        assert_eq!(barrier.subresource_range.level_count, 4);
        assert_eq!(barrier.subresource_range.layer_count, 6);
        assert!(!t.is_expanded(image));
    }

    #[test]
    fn permanent_texture_never_emits() {
        let mut t = Tracker::with_mode(Mode::Check);
        let image = img(8);
        t.register_image(image, desc(Lifetime::Shared, 1, 1));
        t.set_permanent(image, Access::SampledFragment);
        t.require(image, Range::All, Access::SampledFragment);
        t.require(image, Range::All, Access::SampledCompute);
        assert!(t.drain().is_empty());
        assert!(t.mismatches().is_empty(), "same-layout reads verify clean");
        t.require(image, Range::All, Access::ColorWrite);
        assert!(t.drain().is_empty(), "permanent verify never emits");
        assert!(kinds(&t).contains(&MismatchKind::PermanentMismatch));
    }

    #[test]
    fn buffer_needs_merge_into_one_global_barrier() {
        let mut t = tracker();
        let a = buf(1);
        let b = buf(2);
        t.require_buffer(a, Access::StorageWrite(Stages::COMPUTE));
        t.require_buffer(a, Access::IndirectArgs);
        t.require_buffer(b, Access::TransferDst);
        t.require_buffer(b, Access::StorageRead(Stages::COMPUTE));
        let batch = t.drain();
        assert!(batch.images.is_empty());
        assert_eq!(batch.submission_count(), 1);
        let memory = batch.memory.expect("one global memory barrier");
        assert!(memory.src_stage_mask.contains(vk::PipelineStageFlags2::COMPUTE_SHADER));
        assert!(memory.src_stage_mask.contains(vk::PipelineStageFlags2::TRANSFER));
        assert!(memory.dst_stage_mask.contains(vk::PipelineStageFlags2::DRAW_INDIRECT));
        assert!(memory.dst_stage_mask.contains(vk::PipelineStageFlags2::COMPUTE_SHADER));
        assert!(memory.src_access_mask.contains(vk::AccessFlags2::SHADER_WRITE));
        assert!(memory.src_access_mask.contains(vk::AccessFlags2::TRANSFER_WRITE));
    }

    #[test]
    fn drain_is_one_submission_for_images_and_buffers() {
        let mut t = tracker();
        let image = img(9);
        t.register_image(image, desc(Lifetime::PerSlot, 1, 1));
        t.require(image, Range::All, Access::ColorWrite);
        t.require_buffer(buf(3), Access::TransferDst);
        t.require_buffer(buf(3), Access::StorageRead(Stages::COMPUTE));
        let batch = t.drain();
        assert_eq!(batch.submission_count(), 1);
        assert_eq!(batch.images.len(), 1);
        assert!(batch.memory.is_some());
        assert!(t.drain().is_empty());
    }

    #[test]
    fn observe_flags_prefix_bugs_1_through_4() {
        let color = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let sampled = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let undef = vk::ImageLayout::UNDEFINED;
        let top = vk::PipelineStageFlags2::TOP_OF_PIPE;
        let color_stage = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        let frag = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;

        let mut t = Tracker::with_mode(Mode::Check);
        let swap = img(11);
        t.import_image(swap, desc(Lifetime::Imported, 1, 1), undef, color_stage);
        t.observe(&hand_barrier(swap, undef, color, top, color_stage), Access::ColorWrite);
        let found = kinds(&t);
        assert!(found.contains(&MismatchKind::BeforeAcquireStage), "{found:?}");
        assert!(found.contains(&MismatchKind::TopOfPipeWithPriorUse), "{found:?}");
        assert!(found.contains(&MismatchKind::SourceMissesPriorUse), "{found:?}");

        let mut t = Tracker::with_mode(Mode::Check);
        let shared = img(14);
        t.register_image(shared, desc(Lifetime::Shared, 1, 1));
        t.require(shared, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.observe(&hand_barrier(shared, color, color, top, color_stage), Access::ColorWrite);
        let found = kinds(&t);
        assert!(found.contains(&MismatchKind::TopOfPipeWithPriorUse), "{found:?}");
        assert!(found.contains(&MismatchKind::SourceMissesPriorUse), "{found:?}");

        let mut t = Tracker::with_mode(Mode::Check);
        let hdr = img(13);
        t.register_image(hdr, desc(Lifetime::Shared, 1, 1));
        t.observe(&hand_barrier(hdr, undef, color, vk::PipelineStageFlags2::empty(), color_stage), Access::ColorWrite);
        t.observe(&hand_barrier(hdr, color, sampled, color_stage, frag), Access::SampledFragment);
        t.observe(&hand_barrier(hdr, sampled, sampled, frag, compute), Access::SampledCompute);
        assert!(t.mismatches().is_empty(), "the pre-fix sequence's correct half must not flag: {:?}", kinds(&t));
        t.observe(&hand_barrier(hdr, sampled, color, frag, color_stage), Access::ColorWrite);
        assert!(kinds(&t).contains(&MismatchKind::SourceMissesPriorUse), "{:?}", kinds(&t));

        let mut t = Tracker::with_mode(Mode::Check);
        let present = img(12);
        t.import_image(present, desc(Lifetime::Imported, 1, 1), undef, color_stage);
        t.observe(&hand_barrier(present, undef, color, color_stage, color_stage), Access::ColorWrite);
        t.observe(&hand_barrier(present, color, vk::ImageLayout::PRESENT_SRC_KHR, color_stage, color_stage), Access::Present);
        assert!(t.mismatches().is_empty(), "{:?}", kinds(&t));
        t.observe_signal(present, color_stage);
        assert!(kinds(&t).contains(&MismatchKind::SignalNarrowerThanPresent), "{:?}", kinds(&t));
        t.observe_signal(present, vk::PipelineStageFlags2::ALL_COMMANDS);
        assert_eq!(kinds(&t).iter().filter(|k| **k == MismatchKind::SignalNarrowerThanPresent).count(), 1);
    }

    #[test]
    fn observe_flags_old_layout_mismatch() {
        let mut t = Tracker::with_mode(Mode::Check);
        let image = img(15);
        t.register_image(image, desc(Lifetime::Shared, 1, 1));
        t.observe(
            &hand_barrier(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            Access::ColorWrite,
        );
        assert!(kinds(&t).contains(&MismatchKind::OldLayout), "{:?}", kinds(&t));
    }

    #[test]
    fn per_slot_top_of_pipe_is_not_flagged() {
        let mut t = Tracker::with_mode(Mode::Check);
        let image = img(16);
        t.register_image(image, desc(Lifetime::PerSlot, 1, 1));
        t.require(image, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.observe(
            &hand_barrier(image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::TOP_OF_PIPE, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            Access::ColorWrite,
        );
        assert!(t.mismatches().is_empty(), "per-slot TOP_OF_PIPE is legal: {:?}", kinds(&t));
    }

    #[test]
    fn all_commands_source_covers_prior_use() {
        let mut t = Tracker::with_mode(Mode::Check);
        let image = img(17);
        t.register_image(image, desc(Lifetime::Shared, 1, 1));
        t.require(image, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledCompute);
        let _ = t.drain();
        t.observe(
            &hand_barrier(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::ALL_COMMANDS, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            Access::ColorWrite,
        );
        assert!(t.mismatches().is_empty(), "ALL_COMMANDS covers the compute reader: {:?}", kinds(&t));
    }

    #[test]
    fn check_off_is_silent() {
        let mut t = Tracker::with_mode(Mode::Off);
        let image = img(18);
        t.import_image(image, desc(Lifetime::Imported, 1, 1), vk::ImageLayout::UNDEFINED, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        t.observe(
            &hand_barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::TOP_OF_PIPE, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            Access::ColorWrite,
        );
        t.observe_signal(image, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        assert!(t.mismatches().is_empty());
        assert_eq!(t.stats().observes, 0);
    }

    #[test]
    fn panic_mode_panics_on_mismatch() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut t = Tracker::panicking();
            let image = img(19);
            t.import_image(image, desc(Lifetime::Imported, 1, 1), vk::ImageLayout::UNDEFINED, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
            t.observe(
                &hand_barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::TOP_OF_PIPE, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                Access::ColorWrite,
            );
        }));
        assert!(result.is_err());
    }

    #[test]
    fn mode_from_var_matches_env_spec() {
        assert_eq!(mode_from_var(Some("0")).0, Mode::Off);
        assert_eq!(mode_from_var(Some("off")).0, Mode::Off);
        assert_eq!(mode_from_var(Some("1")), (Mode::Check, false));
        assert_eq!(mode_from_var(Some("panic")), (Mode::Check, true));
        assert_eq!(mode_from_var(Some("emit")).0, Mode::Emit);
        let (mode, panic) = mode_from_var(None);
        assert!(!panic);
        if cfg!(debug_assertions) {
            assert_eq!(mode, Mode::Check);
        } else {
            assert_eq!(mode, Mode::Off);
        }
    }

    #[test]
    fn uav_opt_out_skips_second_write_keeps_the_first() {
        let mut t = tracker();
        let buffer = buf(4);
        t.set_buffer_uav_barriers(buffer, false);
        t.require_buffer(buffer, Access::TransferDst);
        let _ = t.drain();
        t.require_buffer(buffer, Access::StorageWrite(Stages::COMPUTE));
        assert!(t.drain().memory.is_some(), "the first UAV use still barriers");
        t.require_buffer(buffer, Access::StorageWrite(Stages::COMPUTE));
        assert!(t.drain().is_empty(), "a later UAV-to-UAV write is skipped when opted out");
        t.require_buffer(buffer, Access::StorageRead(Stages::FRAGMENT));
        let memory = t.drain().memory.expect("a reader still waits for the writes");
        assert!(memory.src_stage_mask.contains(vk::PipelineStageFlags2::COMPUTE_SHADER));
    }

    #[test]
    fn owner_conflict_is_opt_in() {
        let mut quiet = tracker();
        let image = img(20);
        quiet.register_image(image, desc(Lifetime::Shared, 1, 1));
        quiet.require(image, Range::All, Access::ColorWrite);
        let _ = quiet.drain();
        quiet.require(image, Range::All, Access::SampledFragment);
        let _ = quiet.drain();
        quiet.require(image, Range::All, Access::ColorWrite);
        assert!(quiet.mismatches().is_empty(), "no owner tag, no bug-5 report");

        let mut t = Tracker::with_mode(Mode::Check);
        t.register_image(image, desc(Lifetime::Shared, 1, 1));
        t.set_owner(image, "sky");
        t.require(image, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledFragment);
        let _ = t.drain();
        t.set_owner(image, "capture");
        t.require(image, Range::All, Access::TransferDst);
        assert!(kinds(&t).contains(&MismatchKind::OwnerConflict), "{:?}", kinds(&t));
    }

    #[test]
    fn read_after_read_same_stage_emits_nothing() {
        let mut t = tracker();
        let image = img(21);
        t.register_image(image, desc(Lifetime::Shared, 1, 1));
        t.require(image, Range::All, Access::ColorWrite);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledFragment);
        let _ = t.drain();
        t.require(image, Range::All, Access::SampledFragment);
        assert!(t.drain().is_empty());
    }
}
