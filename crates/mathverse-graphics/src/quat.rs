//! Quaternions for 3D rotation.

use mathverse_vector::Vector;

/// Unit quaternion `(w, x, y, z)` representing a rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Quat {
        Quat { w, x, y, z }
    }

    pub fn identity() -> Quat {
        Quat::new(1.0, 0.0, 0.0, 0.0)
    }

    /// From axis-angle (radians), normalized axis.
    pub fn from_axis_angle(axis: [f64; 3], a: f64) -> Quat {
        let v = Vector::new(axis.to_vec()).normalized().unwrap();
        let (s, c) = (a / 2.0).sin_cos();
        Quat::new(c, s * v.get(0), s * v.get(1), s * v.get(2))
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(&self) -> Quat {
        let n = self.norm();
        Quat::new(self.w / n, self.x / n, self.y / n, self.z / n)
    }

    pub fn conjugate(&self) -> Quat {
        Quat::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Hamilton product (composition of rotations).
    pub fn mul(&self, o: &Quat) -> Quat {
        Quat::new(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )
    }

    /// Rotate a 3D vector: `p' = q·p·q⁻¹`.
    ///
    /// ```
    /// use mathverse_graphics::quat::Quat;
    /// let q = Quat::from_axis_angle([0.0, 0.0, 1.0], core::f64::consts::FRAC_PI_2);
    /// let p = q.rotate([1.0, 0.0, 0.0]);
    /// assert!(p[0].abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12);
    /// ```
    pub fn rotate(&self, p: [f64; 3]) -> [f64; 3] {
        let q = self.normalized();
        let pq = Quat::new(0.0, p[0], p[1], p[2]);
        let r = q.mul(&pq).mul(&q.conjugate());
        [r.x, r.y, r.z]
    }

    /// Spherical linear interpolation, `t ∈ [0, 1]`.
    pub fn slerp(&self, other: &Quat, t: f64) -> Quat {
        let a = self.normalized();
        let mut b = other.normalized();
        let mut dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
        if dot < 0.0 {
            b = Quat::new(-b.w, -b.x, -b.y, -b.z);
            dot = -dot;
        }
        if dot > 0.9995 {
            // linear fallback near identity
            let r = Quat::new(
                a.w + t * (b.w - a.w),
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
            );
            return r.normalized();
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let (s0, s1) = (((1.0 - t) * theta).sin(), (t * theta).sin());
        let sin = theta.sin();
        Quat::new(
            (a.w * s0 + b.w * s1) / sin,
            (a.x * s0 + b.x * s1) / sin,
            (a.y * s0 + b.y * s1) / sin,
            (a.z * s0 + b.z * s1) / sin,
        )
        .normalized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_composition() {
        // 90° about z twice = 180°
        let q = Quat::from_axis_angle([0.0, 0.0, 1.0], core::f64::consts::FRAC_PI_2);
        let p = q.mul(&q).rotate([1.0, 0.0, 0.0]);
        assert!(p[0] + 1.0 < 1e-12 && p[1].abs() < 1e-12);
        // inverse rotation round-trips
        let back = q.conjugate().rotate(q.rotate([1.0, 2.0, 3.0]));
        assert!((back[0] - 1.0).abs() < 1e-12 && (back[1] - 2.0).abs() < 1e-12 && (back[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn slerp_endpoints() {
        let a = Quat::identity();
        let b = Quat::from_axis_angle([1.0, 0.0, 0.0], 1.0);
        let mid = a.slerp(&b, 0.5);
        let p = mid.rotate([0.0, 1.0, 0.0]);
        let expected = Quat::from_axis_angle([1.0, 0.0, 0.0], 0.5).rotate([0.0, 1.0, 0.0]);
        assert!((p[0] - expected[0]).abs() < 1e-9 && (p[1] - expected[1]).abs() < 1e-9 && (p[2] - expected[2]).abs() < 1e-9);
        // t = 1 gives b
        let end = a.slerp(&b, 1.0);
        assert!((end.w - b.w).abs() < 1e-9);
    }
}
