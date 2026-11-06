//! New Project template catalogue: the id, name, description and picture of
//! every starter project. Metadata only, so the Project Hub can list templates
//! without linking the engine; the content each template writes lives in
//! `zestengine_engine::templates`.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TemplateCategory {
    Desktop,
    Vr,
}

impl TemplateCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::Desktop => "Desktop",
            Self::Vr => "VR",
        }
    }
}

/// Picture shown on a template card (mapped to the editor icon set by the UI).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TemplateIcon {
    Empty,
    Crosshair,
    Camera,
    Pointer,
    Grid,
    Headset,
    Social,
}

/// One entry in the New Project template picker.
#[derive(Clone, Copy, Debug)]
pub struct ProjectTemplate {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub category: TemplateCategory,
    /// True for templates that ship a player avatar and the locomotion kit
    /// (the F6 avatar switch + `.animctrl` controller). UI / board templates
    /// are false.
    pub gameplay: bool,
    pub icon: TemplateIcon,
    /// Short feature list for the picker's detail line.
    pub features: &'static [&'static str],
}

pub const DEFAULT_TEMPLATE: &str = "empty";

pub const TEMPLATES: &[ProjectTemplate] = &[
    ProjectTemplate {
        id: "empty",
        name: "Empty",
        description: "A camera, a cube and a ground plane. Start from scratch.",
        category: TemplateCategory::Desktop,
        gameplay: false,
        icon: TemplateIcon::Empty,
        features: &["Starter scene", "Default input bindings"],
    },
    ProjectTemplate {
        id: "fps",
        name: "First Person Shooter",
        description: "Mouse-look FPS controller, hitscan rifle, health, target dummies, HUD and pause menu.",
        category: TemplateCategory::Desktop,
        gameplay: true,
        icon: TemplateIcon::Crosshair,
        features: &["WASD + mouse look, jump, sprint, crouch", "Hitscan weapon with reload", "Target range", "Crosshair, ammo and health HUD"],
    },
    ProjectTemplate {
        id: "tps",
        name: "Third Person Shooter",
        description: "Over-the-shoulder character with a collision-aware camera boom, aim mode, weapon and HUD.",
        category: TemplateCategory::Desktop,
        gameplay: true,
        icon: TemplateIcon::Camera,
        features: &["Orbit camera that avoids walls", "Aim (right mouse) zooms over the shoulder", "Hitscan weapon, health, targets"],
    },
    ProjectTemplate {
        id: "topdown",
        name: "Top-Down Click to Move",
        description: "Isometric follow camera; click the ground to walk over a baked navmesh, click switches to use them.",
        category: TemplateCategory::Desktop,
        gameplay: true,
        icon: TemplateIcon::Pointer,
        features: &["NavMesh Surface + Nav Agent", "Click marker", "Interactable switch and lamp", "Wheel zoom"],
    },
    ProjectTemplate {
        id: "strategy",
        name: "Strategy (RTS)",
        description: "RTS camera with edge pan, zoom and rotate; box-select units and send them in formation.",
        category: TemplateCategory::Desktop,
        gameplay: true,
        icon: TemplateIcon::Grid,
        features: &["Edge / WASD pan, wheel zoom, Q/E rotate", "Box and click selection", "Right-click formation moves", "Resource HUD"],
    },
    ProjectTemplate {
        id: "vr_shooter",
        name: "VR Shooter",
        description: "VR rig with hands, a grabbable pistol (trigger fires), teleport and smooth locomotion, targets, wrist HUD.",
        category: TemplateCategory::Vr,
        gameplay: true,
        icon: TemplateIcon::Headset,
        features: &["Head + hand tracking", "Teleport arc and snap turn", "Grab + trigger weapons", "Desktop fallback in the editor"],
    },
    ProjectTemplate {
        id: "vr_social",
        name: "VR Social",
        description: "Hangout room: avatar body, mirror, world-space welcome panel, player spawn points and synced grabbables.",
        category: TemplateCategory::Vr,
        gameplay: true,
        icon: TemplateIcon::Social,
        features: &["Avatar that follows your head", "Planar reflection mirror", "Spawn points + Sync objects for multiplayer", "Push-to-talk voice binding"],
    },
    ProjectTemplate {
        id: "novel",
        name: "Visual Novel",
        description: "A fixed camera on a speaker with a dialogue box and choice buttons. Build branching stories.",
        category: TemplateCategory::Desktop,
        gameplay: false,
        icon: TemplateIcon::Social,
        features: &["Dialogue + speaker name UI", "Choice buttons", "Advance / choice bindings"],
    },
    ProjectTemplate {
        id: "cardgame",
        name: "Trading Card Game",
        description: "A shared table, two seats and a hand of cards, with the Network Manager ready for LAN multiplayer.",
        category: TemplateCategory::Desktop,
        gameplay: false,
        icon: TemplateIcon::Grid,
        features: &["Network Manager + spawn points", "Shared table", "Hand + turn HUD", "LAN discovery"],
    },
];

pub fn find(id: &str) -> Option<&'static ProjectTemplate> {
    TEMPLATES.iter().find(|t| t.id == id)
}
