//! Transform gizmo driver.
//!
//! Drives `transform_gizmo::Gizmo` directly instead of via the crate's egui
//! wrapper. The wrapper registers an interact widget under the cursor every
//! frame, which would make egui claim the pointer over the whole viewport
//! and break click-to-pick. Here the gizmo hit-tests itself
//! (`pick_preview`) and the viewport decides who gets the click.

use egui::{Pos2, Ui};
use glam::{DQuat, DVec3, Mat4, Quat, Vec3};
use transform_gizmo_egui::math::Transform;
use transform_gizmo_egui::{
    EnumSet, Gizmo, GizmoConfig, GizmoInteraction, GizmoMode, GizmoOrientation, GizmoVisuals,
    enum_set,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GizmoTool {
    /// Pick only, no transform handles.
    Select,
    Move,
    Rotate,
    Scale,
    /// 2D rect handles (UI elements); 3D objects get no handles.
    Rect,
}

/// Where the gizmo sits and what transformations pivot around.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PivotMode {
    /// The object's own origin (pivot point).
    Origin,
    /// Center of the mesh bounds (Unity's "Center").
    Center,
}

impl GizmoTool {
    fn modes(self) -> EnumSet<GizmoMode> {
        match self {
            // TranslateView: a camera-facing screen-plane handle (the center
            // square) for dragging in the view plane, alongside the axis/plane
            // handles.
            Self::Move => enum_set!(
                GizmoMode::TranslateX
                    | GizmoMode::TranslateY
                    | GizmoMode::TranslateZ
                    | GizmoMode::TranslateXY
                    | GizmoMode::TranslateXZ
                    | GizmoMode::TranslateYZ
                    | GizmoMode::TranslateView
            ),
            // RotateView: a camera-facing ring (screen-aligned rotation) plus
            // the three world axes.
            Self::Rotate => enum_set!(
                GizmoMode::RotateX | GizmoMode::RotateY | GizmoMode::RotateZ | GizmoMode::RotateView
            ),
            Self::Scale => enum_set!(
                GizmoMode::ScaleX | GizmoMode::ScaleY | GizmoMode::ScaleZ | GizmoMode::ScaleUniform
            ),
            Self::Select | Self::Rect => EnumSet::empty(),
        }
    }
}

pub struct GizmoState {
    gizmo: Gizmo,
    pub tool: GizmoTool,
    pub pivot: PivotMode,
    /// Gizmo axes follow the object (local) or the world (global).
    pub local_orientation: bool,
    /// Snap translations to the grid (also active while Ctrl is held).
    pub snap: bool,
    /// Grid cell size in meters (translation snap increment).
    pub grid_size: f32,
    /// Asset placement: align a dragged-in asset's up axis to the surface normal.
    pub align_to_surface: bool,
    /// During a move: the pivot where the drag began and where it would be
    /// without snapping (snapping rounds this, so small steps still add up).
    drag: Option<(Vec3, Vec3)>,
}

impl GizmoState {
    pub fn new() -> Self {
        Self {
            gizmo: Gizmo::default(),
            tool: GizmoTool::Move,
            pivot: PivotMode::Origin,
            local_orientation: false,
            snap: false,
            grid_size: 0.5,
            align_to_surface: true,
            drag: None,
        }
    }

    pub fn is_focused(&self) -> bool {
        self.gizmo.is_focused()
    }

    pub fn pick_preview(&self, pos: Pos2) -> bool {
        self.gizmo.pick_preview((pos.x, pos.y))
    }

    /// Update + draw the gizmo for one target. `pivot_local` is the pivot
    /// point in object space (zero = object origin). Returns true when the
    /// transform was modified this frame. Holding Shift moves and rotates at
    /// half speed for fine placement; Ctrl (or the Snap toggle) snaps moves
    /// onto the grid and rotations / scales to their steps.
    #[allow(clippy::too_many_arguments)]
    pub fn interact(
        &mut self,
        ui: &Ui,
        viewport: egui::Rect,
        view: Mat4,
        proj: Mat4,
        trs: (&mut Vec3, &mut Quat, &mut Vec3),
        pivot_local: Vec3,
        cursor: Option<Pos2>,
        drag_started: bool,
        dragging: bool,
    ) -> bool {
        let (translation, rotation, scale) = trs;
        let (ctrl, shift) = ui.input(|i| (i.modifiers.ctrl, i.modifiers.shift));
        let snap = self.snap || ctrl;
        let moving = self.tool == GizmoTool::Move;
        if !dragging {
            self.drag = None;
        }
        self.gizmo.update_config(GizmoConfig {
            view_matrix: view.as_dmat4().into(),
            projection_matrix: proj.as_dmat4().into(),
            viewport,
            modes: self.tool.modes(),
            orientation: if self.local_orientation {
                GizmoOrientation::Local
            } else {
                GizmoOrientation::Global
            },
            // Moves snap here to absolute grid lines (below), not in steps.
            snapping: snap && !moving,
            snap_distance: self.grid_size.max(0.001),
            snap_angle: 15f32.to_radians(),
            snap_scale: 0.1,
            pixels_per_point: ui.ctx().pixels_per_point(),
            visuals: GizmoVisuals {
                // Fat, easy-to-grab handles. The pick tolerance scales with
                // `gizmo_size * 0.1 + stroke_width * 2`, so a thick stroke both
                // widens the clickable band and makes grabbing reliable (much
                // less accidental orbit).
                gizmo_size: 110.0,
                stroke_width: 11.0,
                // Per-handle hover emphasis. Idle axes are dimmed so the axis
                // under the cursor (full alpha, its own color) stands out as
                // the one you'll grab. The crate highlights only the
                // hovered/active sub-gizmo, so the rest of the widget is
                // unaffected (no whole-gizmo resize).
                inactive_alpha: 0.5,
                highlight_alpha: 1.0,
                ..Default::default()
            },
            ..Default::default()
        });

        // The gizmo edits a frame placed at the pivot; the object's origin
        // is carried along as an offset within that frame.
        let pivot_world = *translation + *rotation * (*scale * pivot_local);
        let target = Transform::from_scale_rotation_translation(
            scale.as_dvec3(),
            rotation.as_dquat(),
            pivot_world.as_dvec3(),
        );
        let cursor_pos = cursor.map(|p| (p.x, p.y)).unwrap_or_default();
        let result = self.gizmo.update(
            GizmoInteraction {
                cursor_pos,
                hovered: cursor.is_some(),
                drag_started,
                dragging,
            },
            &[target],
        );

        // Draw through the viewport tab's painter (clipped to the tab).
        let draw_data = self.gizmo.draw();
        if !draw_data.indices.is_empty() {
            let mesh = egui::Mesh {
                indices: draw_data.indices,
                vertices: draw_data
                    .vertices
                    .into_iter()
                    .zip(draw_data.colors)
                    .map(|(pos, [r, g, b, a])| egui::epaint::Vertex {
                        pos: pos.into(),
                        uv: Pos2::default(),
                        color: egui::Rgba::from_rgba_premultiplied(r, g, b, a).into(),
                    })
                    .collect(),
                ..Default::default()
            };
            ui.painter().add(mesh);
        }

        if let Some((_, transforms)) = result
            && let Some(t) = transforms.first()
        {
            let mut new_pivot = DVec3::from(t.translation).as_vec3();
            let mut new_rotation = DQuat::from(t.rotation).as_quat();
            let new_scale = DVec3::from(t.scale).as_vec3();
            let factor = if shift { 0.5 } else { 1.0 };
            if moving {
                let (start, raw) = self.drag.get_or_insert((pivot_world, pivot_world));
                *raw += (new_pivot - pivot_world) * factor;
                new_pivot = if snap { snap_moved(*raw, *start, self.grid_size) } else { *raw };
            } else if shift {
                // Half of this frame's turn.
                new_rotation = (Quat::IDENTITY.slerp(new_rotation * rotation.inverse(), 0.5) * *rotation).normalize();
            }

            // Re-derive the object origin from the edited pivot frame:
            // origin = pivot + R_new * (ratio * (R_old⁻¹ * offset)).
            let offset = *translation - pivot_world;
            let ratio = Vec3::select(scale.cmpne(Vec3::ZERO), new_scale / *scale, Vec3::ONE);
            let local_offset = rotation.inverse() * offset;
            *translation = new_pivot + new_rotation * (local_offset * ratio);
            *rotation = new_rotation;
            *scale = new_scale;
            return true;
        }
        false
    }
}

/// `raw` with every axis that moved since `start` rounded to the grid (axes
/// the drag isn't moving stay exactly where they were).
fn snap_moved(raw: Vec3, start: Vec3, grid: f32) -> Vec3 {
    let g = grid.max(0.001);
    Vec3::from_array(std::array::from_fn(|k| if (raw[k] - start[k]).abs() > 1e-6 { (raw[k] / g).round() * g } else { start[k] }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapping_rounds_only_the_moved_axes_onto_the_grid() {
        let start = Vec3::new(0.3, 1.27, -2.0);
        let raw = Vec3::new(1.12, 1.27, -2.0);
        assert_eq!(snap_moved(raw, start, 0.5), Vec3::new(1.0, 1.27, -2.0));
        // Small steps add up in `raw` until they cross a grid line.
        assert_eq!(snap_moved(start + Vec3::X * 0.1, start, 0.5).x, 0.5);
    }
}
