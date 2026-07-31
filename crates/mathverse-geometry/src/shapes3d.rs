//! 3D shapes: point, sphere, cube, cylinder, cone, plane, line.

use mathverse_core::constants::PI;

/// 3D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Point3 { x, y, z }
    }
    pub fn distance_to(self, other: Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    pub fn translate(self, dx: f64, dy: f64, dz: f64) -> Point3 {
        Point3::new(self.x + dx, self.y + dy, self.z + dz)
    }
    pub fn scale(self, s: f64) -> Point3 {
        Point3::new(self.x * s, self.y * s, self.z * s)
    }
}

fn dot(a: Point3, b: Point3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn cross(a: Point3, b: Point3) -> Point3 {
    Point3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn norm(v: Point3) -> f64 {
    dot(v, v).sqrt()
}

/// Sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Point3,
    pub radius: f64,
}

impl Sphere {
    /// Panics if `radius < 0`.
    pub fn new(center: Point3, radius: f64) -> Sphere {
        assert!(radius >= 0.0, "sphere radius must be non-negative");
        Sphere { center, radius }
    }
    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * PI * self.radius.powi(3)
    }
    pub fn surface_area(&self) -> f64 {
        4.0 * PI * self.radius.powi(2)
    }
    pub fn contains(&self, p: Point3) -> bool {
        self.center.distance_to(p) <= self.radius
    }
    /// Sphere-sphere collision (touch counts as collision).
    pub fn collides_with(&self, other: &Sphere) -> bool {
        self.center.distance_to(other.center) <= self.radius + other.radius
    }
}

/// Cube with a center and side length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cube {
    pub center: Point3,
    pub side: f64,
}

impl Cube {
    /// Panics if `side < 0`.
    pub fn new(center: Point3, side: f64) -> Cube {
        assert!(side >= 0.0, "cube side must be non-negative");
        Cube { center, side }
    }
    pub fn volume(&self) -> f64 {
        self.side.powi(3)
    }
    pub fn surface_area(&self) -> f64 {
        6.0 * self.side.powi(2)
    }
}

/// Right circular cylinder, `height` along the z axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    pub center: Point3,
    pub radius: f64,
    pub height: f64,
}

impl Cylinder {
    /// Panics if `radius < 0` or `height < 0`.
    pub fn new(center: Point3, radius: f64, height: f64) -> Cylinder {
        assert!(radius >= 0.0 && height >= 0.0, "cylinder dimensions must be non-negative");
        Cylinder { center, radius, height }
    }
    pub fn volume(&self) -> f64 {
        PI * self.radius.powi(2) * self.height
    }
    /// Lateral surface plus both caps.
    pub fn surface_area(&self) -> f64 {
        2.0 * PI * self.radius * (self.radius + self.height)
    }
}

/// Right circular cone, `height` along the z axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    pub center: Point3,
    pub radius: f64,
    pub height: f64,
}

impl Cone {
    /// Panics if `radius < 0` or `height < 0`.
    pub fn new(center: Point3, radius: f64, height: f64) -> Cone {
        assert!(radius >= 0.0 && height >= 0.0, "cone dimensions must be non-negative");
        Cone { center, radius, height }
    }
    pub fn volume(&self) -> f64 {
        PI * self.radius.powi(2) * self.height / 3.0
    }
    /// Lateral surface plus base.
    pub fn surface_area(&self) -> f64 {
        PI * self.radius * (self.radius + (self.radius.powi(2) + self.height.powi(2)).sqrt())
    }
}

/// Plane `normal · p + d = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: Point3,
    pub d: f64,
}

impl Plane {
    /// Panics if `normal` is the zero vector.
    pub fn new(normal: Point3, d: f64) -> Plane {
        assert!(norm(normal) > 0.0, "plane normal must be non-zero");
        Plane { normal, d }
    }
    /// Signed distance (positive on the side the normal points to).
    pub fn signed_distance(&self, p: Point3) -> f64 {
        (dot(self.normal, p) + self.d) / norm(self.normal)
    }
    /// Absolute distance.
    pub fn distance(&self, p: Point3) -> f64 {
        self.signed_distance(p).abs()
    }
    /// Orthogonal projection of `p` onto the plane.
    pub fn project(&self, p: Point3) -> Point3 {
        let n = self.normal.scale(1.0 / norm(self.normal));
        p.translate(-n.x * self.signed_distance(p), -n.y * self.signed_distance(p), -n.z * self.signed_distance(p))
    }
}

/// Line through `point` with direction `dir`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3 {
    pub point: Point3,
    pub dir: Point3,
}

impl Line3 {
    /// Panics if `dir` is the zero vector.
    pub fn new(point: Point3, dir: Point3) -> Line3 {
        assert!(norm(dir) > 0.0, "line direction must be non-zero");
        Line3 { point, dir }
    }
    /// Distance from a point to the line.
    pub fn distance(&self, p: Point3) -> f64 {
        let v = p.translate(-self.point.x, -self.point.y, -self.point.z);
        norm(cross(v, self.dir)) / norm(self.dir)
    }
    /// Orthogonal projection of `p` onto the line.
    pub fn project(&self, p: Point3) -> Point3 {
        let v = p.translate(-self.point.x, -self.point.y, -self.point.z);
        let t = dot(v, self.dir) / dot(self.dir, self.dir);
        self.point.translate(self.dir.x * t, self.dir.y * t, self.dir.z * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere() {
        let s = Sphere::new(Point3::new(0.0, 0.0, 0.0), 1.0);
        assert!((s.volume() - 4.0 / 3.0 * PI).abs() < 1e-12);
        assert!((s.surface_area() - 4.0 * PI).abs() < 1e-12);
        assert!(s.contains(Point3::new(0.5, 0.5, 0.5)));
        assert!(!s.contains(Point3::new(1.0, 1.0, 1.0)));
        let a = Sphere::new(Point3::new(0.0, 0.0, 0.0), 1.0);
        let b = Sphere::new(Point3::new(1.5, 0.0, 0.0), 1.0);
        assert!(a.collides_with(&b));
        let c = Sphere::new(Point3::new(3.0, 0.0, 0.0), 1.0);
        assert!(!a.collides_with(&c));
    }

    #[test]
    fn cube() {
        let c = Cube::new(Point3::new(0.0, 0.0, 0.0), 2.0);
        assert_eq!(c.volume(), 8.0);
        assert_eq!(c.surface_area(), 24.0);
    }

    #[test]
    fn cylinder_and_cone() {
        let cy = Cylinder::new(Point3::new(0.0, 0.0, 0.0), 1.0, 3.0);
        assert!((cy.volume() - 3.0 * PI).abs() < 1e-12);
        let co = Cone::new(Point3::new(0.0, 0.0, 0.0), 1.0, 3.0);
        assert!((co.volume() - PI).abs() < 1e-12); // exactly 1/3 of the cylinder
    }

    #[test]
    fn plane() {
        let p = Plane::new(Point3::new(0.0, 0.0, 1.0), -2.0); // z = 2
        assert_eq!(p.distance(Point3::new(1.0, 1.0, 5.0)), 3.0);
        assert_eq!(p.project(Point3::new(1.0, 1.0, 5.0)), Point3::new(1.0, 1.0, 2.0));
        assert_eq!(p.signed_distance(Point3::new(0.0, 0.0, 0.0)), -2.0);
    }

    #[test]
    fn line() {
        // x axis through origin
        let l = Line3::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0));
        assert_eq!(l.distance(Point3::new(0.0, 3.0, 4.0)), 5.0);
        assert_eq!(l.project(Point3::new(7.0, 3.0, 4.0)), Point3::new(7.0, 0.0, 0.0));
    }
}
