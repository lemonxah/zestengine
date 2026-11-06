//! Shared numeric transitions without scene, rendering or audio dependencies.

/// Linear, interruptible transition. Retargeting starts from the current value.
#[derive(Clone, Copy, Debug)]
pub struct Transition {
    current: f32,
    from: f32,
    target: f32,
    elapsed: f32,
    duration: f32,
}

impl Transition {
    pub fn new(value: f32) -> Self {
        Self { current: value, from: value, target: value, elapsed: 0.0, duration: 0.0 }
    }
    pub fn step(&mut self, target: f32, seconds: f32, dt: f32) -> f32 {
        if target != self.target {
            self.from = self.current;
            self.target = target;
            self.elapsed = 0.0;
            self.duration = seconds.max(0.0);
        }
        self.elapsed += dt.max(0.0);
        let t = if self.duration > 0.0 { (self.elapsed / self.duration).min(1.0) } else { 1.0 };
        self.current = self.from + (self.target - self.from) * t;
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transition_retargets_without_jumping_and_finishes() {
        let mut fade = Transition::new(1.0);
        assert_eq!(fade.step(0.0, 1.0, 0.25), 0.75);
        assert_eq!(fade.step(1.0, 0.5, 0.25), 0.875);
        assert_eq!(fade.step(1.0, 0.5, 0.25), 1.0);
        assert_eq!(fade.step(0.0, 0.0, 0.0), 0.0);
    }
}
