//! Pinhole camera model: projection, unprojection, intrinsic parameters.

/// Pinhole camera with focal lengths `fx`, `fy` and principal point `(cx, cy)`.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Camera {
    /// Creates a new camera model with given focal length `(fx, fy)` and principal point `(cx, cy)`.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Projects 3D camera coordinates `(x, y, z)` into 2D pixel coordinates `(u, v)`.
    pub fn project(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        (self.fx * x / z + self.cx, self.fy * y / z + self.cy)
    }

    /// Unprojects 2D pixel coordinates `(u, v)` at depth `z` back to 3D camera coordinates `(x, y, z)`.
    pub fn unproject(&self, u: f64, v: f64, z: f64) -> (f64, f64) {
        ((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn projection() {
        let c = Camera::new(500.0, 500.0, 320.0, 240.0);
        let (u, v) = c.project(1.0, 2.0, 10.0);
        assert!((u - 370.0).abs() < EPS && (v - 340.0).abs() < EPS);
    }

    #[test]
    fn unproject_roundtrip() {
        let c = Camera::new(800.0, 800.0, 400.0, 300.0);
        let (u, v) = c.project(0.5, -0.25, 2.0);
        let (x, y) = c.unproject(u, v, 2.0);
        assert!((x - 0.5).abs() < EPS && (y + 0.25).abs() < EPS);
    }

    #[test]
    fn depth_effect() {
        let c = Camera::new(500.0, 500.0, 320.0, 240.0);
        let (u1, _) = c.project(1.0, 0.0, 5.0);
        let (u2, _) = c.project(1.0, 0.0, 10.0);
        assert!(u2 < u1);
    }
}
