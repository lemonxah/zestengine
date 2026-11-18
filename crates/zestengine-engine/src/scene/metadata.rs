//! Typed object metadata shared by scene handles and native engine queries.

use zestengine_core::ecs::ObjectMetadata;
use super::RenderInfo;

pub struct Render(pub Option<RenderInfo>);
impl ObjectMetadata for Render {}

/// Static objects contribute geometry to lighting bakes.
pub struct StaticGeometry(pub bool);
impl ObjectMetadata for StaticGeometry {}

/// Camera culling and collision layer, in the range 0..32.
pub struct Layer(pub u8);
impl ObjectMetadata for Layer {}

pub struct NetRole(pub zestengine_core::NetRole);
impl ObjectMetadata for NetRole {}

pub struct Source(pub zestengine_assets::ObjectSource);
impl ObjectMetadata for Source {}

pub struct ExtraRender(pub Vec<RenderInfo>);
impl ObjectMetadata for ExtraRender {}

pub struct ComponentRoles(pub std::collections::BTreeMap<String, zestengine_core::NetRole>);
impl ObjectMetadata for ComponentRoles {}

/// A constructor/import command waiting for its scene order to be installed.
/// Resolved objects use only native Parent entity links; this stays empty.
#[derive(Clone, Copy, Default)]
pub(super) struct ParentRequest {
    pub row: Option<Option<usize>>,
    pub saved: Option<zestengine_core::ObjectId>,
}
impl ObjectMetadata for ParentRequest {}

/// Keep the extra-slot lease alive without borrowing an iterator from itself.
/// RenderInfo is small Copy data; iterating never clones the backing vector.
pub(super) struct RenderSlots<'a> {
    pub(super) primary: Option<RenderInfo>,
    pub(super) extra: zestengine_core::ecs::ComponentRef<'a, Vec<RenderInfo>>,
    pub(super) index: usize,
}

impl Iterator for RenderSlots<'_> {
    type Item = RenderInfo;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(primary) = self.primary.take() { return Some(primary); }
        let value = self.extra.get(self.index).copied()?;
        self.index += 1;
        Some(value)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::from(self.primary.is_some()) + self.extra.len() - self.index;
        (remaining, Some(remaining))
    }
    fn count(self) -> usize { self.size_hint().0 }
}
impl ExactSizeIterator for RenderSlots<'_> {}

macro_rules! borrowed_metadata {
    (@native _, $get:ident, $metadata:ty, $value:ty) => {};
    (@native $get_in:ident, $get:ident, $metadata:ty, $value:ty) => {
        pub(crate) fn $get_in<'a>(&self, world: &'a zestengine_core::ecs::SystemWorld<'_>) -> zestengine_core::ecs::Ref<'a, $value> {
            zestengine_core::ecs::Ref::map(world.get::<$metadata>(self.entity()).expect(concat!("object ", stringify!($get))), |value| &value.0)
        }
    };
    ($get:ident, $get_in:tt, $write:ident, $with:ident, $metadata:ty, $value:ty) => {
        pub fn $get(&self) -> zestengine_core::ecs::ComponentRef<'_, $value> {
            zestengine_core::ecs::ComponentRef::map(self.components.metadata::<$metadata>().expect(concat!("object ", stringify!($get))), |value| &value.0)
        }
        $crate::scene::metadata::borrowed_metadata!(@native $get_in, $get, $metadata, $value);
        pub fn $write(&self) -> zestengine_core::ecs::ComponentMut<'_, $value> {
            zestengine_core::ecs::ComponentMut::map(self.components.metadata_mut::<$metadata>().expect(concat!("object ", stringify!($get))), |value| &mut value.0)
        }
        pub fn $with(mut self, value: $value) -> Self {
            self.components.insert_metadata(<$metadata>::from(value));
            self
        }
    };
}
pub(super) use borrowed_metadata;

impl From<zestengine_assets::ObjectSource> for Source { fn from(value: zestengine_assets::ObjectSource) -> Self { Self(value) } }
impl From<Vec<RenderInfo>> for ExtraRender { fn from(value: Vec<RenderInfo>) -> Self { Self(value) } }
impl From<std::collections::BTreeMap<String, zestengine_core::NetRole>> for ComponentRoles {
    fn from(value: std::collections::BTreeMap<String, zestengine_core::NetRole>) -> Self { Self(value) }
}

macro_rules! copy_metadata {
    (@native _, $get:ident, $metadata:ty, $value:ty) => {};
    (@native $get_in:ident, $get:ident, $metadata:ty, $value:ty) => {
        pub(crate) fn $get_in(&self, world: &zestengine_core::ecs::SystemWorld<'_>) -> $value {
            world.get::<$metadata>(self.entity()).expect(concat!("object ", stringify!($get))).0
        }
    };
    ($get:ident, $get_in:tt, $write:ident, $with:ident, $metadata:ty, $value:ty) => {
        pub fn $get(&self) -> $value { self.components.metadata::<$metadata>().expect(concat!("object ", stringify!($get))).0 }
        $crate::scene::metadata::copy_metadata!(@native $get_in, $get, $metadata, $value);
        pub fn $write(&self) -> zestengine_core::ecs::ComponentMut<'_, $value> {
            zestengine_core::ecs::ComponentMut::map(self.components.metadata_mut::<$metadata>().expect(concat!("object ", stringify!($get))), |value| &mut value.0)
        }
        pub fn $with(mut self, value: $value) -> Self {
            self.components.insert_metadata(<$metadata>::from(value));
            self
        }
    };
}
pub(super) use copy_metadata;

impl From<Option<RenderInfo>> for Render { fn from(value: Option<RenderInfo>) -> Self { Self(value) } }
impl From<bool> for StaticGeometry { fn from(value: bool) -> Self { Self(value) } }
impl From<u8> for Layer { fn from(value: u8) -> Self { Self(value) } }
impl From<zestengine_core::NetRole> for NetRole { fn from(value: zestengine_core::NetRole) -> Self { Self(value) } }
