//! Free-fly editor camera.
//!
//! Controls: hold right mouse to look, WASD to fly (Q/E down/up, Shift =
//! fast) while looking, middle mouse drag to pan, scroll to dolly.

use glam::{Mat4, Vec3};

/// Orthographic near/far distance (symmetric about the eye).
const ORTHO_DEPTH: f32 = 500.0;

pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    /// Orthographic ("2D") view: no foreshortening; `ortho_size` is the
    /// visible half-height in world units. Scroll zooms the size.
    pub ortho: bool,
    pub ortho_size: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self::looking_at(Vec3::new(3.9, 2.2, 2.7), Vec3::new(0.0, 0.5, 0.0))
    }
}

impl FlyCamera {
    pub fn looking_at(position: Vec3, target: Vec3) -> Self {
        let dir = (target - position).normalize_or(Vec3::NEG_Z);
        Self {
            position,
            yaw: dir.z.atan2(dir.x),
            pitch: dir.y.asin(),
            // 50 deg vertical (~80 deg horizontal at 16:9): wide enough to frame a
            // scene, narrow enough that objects near the edges aren't stretched by
            // perspective the way a 60+ deg FOV does.
            fov_y: 50f32.to_radians(),
            ortho: false,
            ortho_size: 5.0,
        }
    }

    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or(Vec3::X)
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.0035;
        self.pitch = (self.pitch - dy * 0.0035).clamp(-1.55, 1.55);
    }

    /// Turntable orbit around `pivot` (left drag): the camera revolves around
    /// the pivot, looking at it. The pivot is locked for the whole drag (see
    /// engine), so the rotation is stable.
    pub fn orbit(&mut self, pivot: Vec3, dx: f32, dy: f32) {
        let offset = self.position - pivot;
        let distance = offset.length().max(0.05);
        let azimuth = offset.z.atan2(offset.x);
        let elevation = (offset.y / distance).clamp(-1.0, 1.0).asin();

        let new_azimuth = azimuth + dx * 0.008;
        let new_elevation = (elevation + dy * 0.008).clamp(-1.54, 1.54);

        let (sy, cy) = new_azimuth.sin_cos();
        let (sp, cp) = new_elevation.sin_cos();
        self.position = pivot + Vec3::new(cy * cp, sp, sy * cp) * distance;

        let dir = (pivot - self.position).normalize_or(Vec3::NEG_Z);
        self.yaw = dir.z.atan2(dir.x);
        self.pitch = dir.y.asin();
    }

    /// Screen-space pan (middle mouse drag), in pixels.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let up = self.right().cross(self.forward()).normalize_or(Vec3::Y);
        // Orthographic: a pixel is a fixed fraction of the visible height
        // (about 900 px tall), so the scene follows the cursor at any zoom.
        let k = if self.ortho { self.ortho_size * 2.0 / 900.0 } else { 0.005 };
        self.position += self.right() * (-dx * k) + up * (dy * k);
    }

    /// Scroll dolly along the view direction.
    pub fn dolly(&mut self, scroll: f32) {
        if self.ortho {
            self.ortho_size = (self.ortho_size * (1.0 - scroll * 0.1)).clamp(0.05, 5000.0);
            return;
        }
        self.position += self.forward() * scroll * 0.6;
    }

    /// Frame an object: keep the view direction, move so the object fills a
    /// comfortable portion of the screen (F-focus).
    pub fn focus(&mut self, center: Vec3, radius: f32) {
        if self.ortho {
            self.ortho_size = (radius * 1.5).clamp(0.05, 5000.0);
            self.position = center - self.forward() * ORTHO_DEPTH * 0.5;
            return;
        }
        self.position = center - self.forward() * (radius * 3.0).max(0.5);
    }

    /// `local` is (right, up, forward) movement intent, e.g. from WASD.
    pub fn fly(&mut self, local: Vec3, dt: f32, fast: bool) {
        let speed = if fast { 10.0 } else { 3.0 };
        let movement = self.right() * local.x + Vec3::Y * local.y + self.forward() * local.z;
        self.position += movement.normalize_or(Vec3::ZERO) * speed * dt;
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        if self.ortho {
            // The volume reaches behind the eye too, so geometry beside and
            // behind the (arbitrary) camera position stays visible.
            return zestengine_core::orthographic_projection(self.ortho_size, aspect, -ORTHO_DEPTH, ORTHO_DEPTH);
        }
        Mat4::perspective_rh(self.fov_y, aspect, 0.05, 500.0)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthographic_scroll_zooms_the_size_and_keeps_the_position() {
        let mut cam = FlyCamera { ortho: true, ..FlyCamera::default() };
        let (pos, size) = (cam.position, cam.ortho_size);
        cam.dolly(1.0);
        assert_eq!(cam.position, pos);
        assert!(cam.ortho_size < size);
        cam.dolly(-100.0);
        assert!(cam.ortho_size <= 5000.0);
    }

    #[test]
    fn orthographic_projection_sees_behind_the_eye() {
        let cam = FlyCamera { ortho: true, ..FlyCamera::default() };
        let vp = cam.proj(1.0) * cam.view();
        let behind = cam.position - cam.forward() * 10.0;
        let p = vp.project_point3(behind);
        assert!((0.0..=1.0).contains(&p.z), "depth {}", p.z);
    }
}
