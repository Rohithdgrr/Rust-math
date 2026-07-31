//! 2D shapes: point, circle, triangle, rectangle, polygon, ellipse.

use mathverse_core::constants::{PI, TAU};

/// 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Point2 { x, y }
    }
    /// Euclidean distance to another point.
    pub fn distance_to(self, other: Point2) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
    /// Translate by `(dx, dy)`.
    pub fn translate(self, dx: f64, dy: f64) -> Point2 {
        Point2::new(self.x + dx, self.y + dy)
    }
    /// Uniform scale about the origin.
    pub fn scale(self, s: f64) -> Point2 {
        Point2::new(self.x * s, self.y * s)
    }
    /// Rotate by `angle` radians counterclockwise about the origin.
    pub fn rotate(self, angle: f64) -> Point2 {
        let (s, c) = angle.sin_cos();
        Point2::new(c * self.x - s * self.y, s * self.x + c * self.y)
    }
    /// Rotate by `angle` radians about `center`.
    pub fn rotate_around(self, angle: f64, center: Point2) -> Point2 {
        self.translate(-center.x, -center.y).rotate(angle).translate(center.x, center.y)
    }
}

/// Circle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Point2,
    pub radius: f64,
}

impl Circle {
    /// Panics if `radius < 0`.
    pub fn new(center: Point2, radius: f64) -> Circle {
        assert!(radius >= 0.0, "circle radius must be non-negative");
        Circle { center, radius }
    }
    pub fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }
    pub fn perimeter(&self) -> f64 {
        TAU * self.radius
    }
    pub fn centroid(&self) -> Point2 {
        self.center
    }
    pub fn contains(&self, p: Point2) -> bool {
        self.center.distance_to(p) <= self.radius
    }
    /// Whether this circle overlaps `other` (touch counts).
    pub fn intersects(&self, other: &Circle) -> bool {
        self.center.distance_to(other.center) <= self.radius + other.radius
    }
    /// Intersection points with another circle: 0, 1, or 2.
    /// Coincident circles yield `[]`.
    pub fn intersection_points(&self, other: &Circle) -> Vec<Point2> {
        let d = self.center.distance_to(other.center);
        let (r1, r2) = (self.radius, other.radius);
        if d > r1 + r2 || d < (r1 - r2).abs() || d == 0.0 {
            return vec![];
        }
        let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
        let h2 = r1 * r1 - a * a;
        let p = self.center.translate(a * (other.center.x - self.center.x) / d, a * (other.center.y - self.center.y) / d);
        if h2 <= 1e-14 {
            return vec![p];
        }
        let h = h2.sqrt();
        let dx = h * (other.center.y - self.center.y) / d;
        let dy = h * (other.center.x - self.center.x) / d;
        vec![Point2::new(p.x + dx, p.y - dy), Point2::new(p.x - dx, p.y + dy)]
    }
}

/// Triangle given by three vertices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    pub a: Point2,
    pub b: Point2,
    pub c: Point2,
}

impl Triangle {
    pub fn new(a: Point2, b: Point2, c: Point2) -> Triangle {
        Triangle { a, b, c }
    }
    /// Shoelace formula; area is always positive.
    pub fn area(&self) -> f64 {
        ((self.b.x - self.a.x) * (self.c.y - self.a.y) - (self.c.x - self.a.x) * (self.b.y - self.a.y)).abs() / 2.0
    }
    pub fn perimeter(&self) -> f64 {
        self.a.distance_to(self.b) + self.b.distance_to(self.c) + self.c.distance_to(self.a)
    }
    pub fn centroid(&self) -> Point2 {
        Point2::new((self.a.x + self.b.x + self.c.x) / 3.0, (self.a.y + self.b.y + self.c.y) / 3.0)
    }
}

/// Axis-aligned rectangle, `(x, y)` is the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangle {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rectangle {
    /// Panics if `width < 0` or `height < 0`.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Rectangle {
        assert!(width >= 0.0 && height >= 0.0, "rectangle dimensions must be non-negative");
        Rectangle { x, y, width, height }
    }
    pub fn area(&self) -> f64 {
        self.width * self.height
    }
    pub fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }
    pub fn centroid(&self) -> Point2 {
        Point2::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
    pub fn contains(&self, p: Point2) -> bool {
        p.x >= self.x && p.x <= self.x + self.width && p.y >= self.y && p.y <= self.y + self.height
    }
    /// Overlap of two axis-aligned rectangles.
    pub fn intersects(&self, other: &Rectangle) -> bool {
        self.x < other.x + other.width
            && other.x < self.x + self.width
            && self.y < other.y + other.height
            && other.y < self.y + self.height
    }
}

/// Polygon from an ordered vertex list (not closed; the last vertex joins the first).
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    pub points: Vec<Point2>,
}

impl Polygon {
    /// Panics if fewer than 3 points.
    pub fn new(points: Vec<Point2>) -> Polygon {
        assert!(points.len() >= 3, "polygon needs at least 3 points");
        Polygon { points }
    }
    /// Shoelace formula; works for convex and simple concave polygons.
    pub fn area(&self) -> f64 {
        let mut s = 0.0;
        let n = self.points.len();
        for i in 0..n {
            let (a, b) = (self.points[i], self.points[(i + 1) % n]);
            s += a.x * b.y - b.x * a.y;
        }
        s.abs() / 2.0
    }
    pub fn perimeter(&self) -> f64 {
        let n = self.points.len();
        (0..n).map(|i| self.points[i].distance_to(self.points[(i + 1) % n])).sum()
    }
    /// Area-weighted centroid.
    pub fn centroid(&self) -> Point2 {
        let n = self.points.len();
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut a2 = 0.0;
        for i in 0..n {
            let (p, q) = (self.points[i], self.points[(i + 1) % n]);
            let cross = p.x * q.y - q.x * p.y;
            a2 += cross;
            cx += (p.x + q.x) * cross;
            cy += (p.y + q.y) * cross;
        }
        if a2 == 0.0 {
            return Point2::new(0.0, 0.0);
        }
        Point2::new(cx / (3.0 * a2), cy / (3.0 * a2))
    }
}

/// Ellipse with semi-axes `rx`, `ry`, axis-aligned about `center`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    pub center: Point2,
    pub rx: f64,
    pub ry: f64,
}

impl Ellipse {
    /// Panics if `rx < 0` or `ry < 0`.
    pub fn new(center: Point2, rx: f64, ry: f64) -> Ellipse {
        assert!(rx >= 0.0 && ry >= 0.0, "ellipse semi-axes must be non-negative");
        Ellipse { center, rx, ry }
    }
    pub fn area(&self) -> f64 {
        PI * self.rx * self.ry
    }
    /// Ramanujan's approximation (error < 0.04% for all eccentricities).
    pub fn perimeter(&self) -> f64 {
        let s = self.rx + self.ry;
        PI * (3.0 * s - ((3.0 * self.rx + self.ry) * (self.rx + 3.0 * self.ry)).sqrt())
    }
    pub fn centroid(&self) -> Point2 {
        self.center
    }
    pub fn contains(&self, p: Point2) -> bool {
        let dx = (p.x - self.center.x) / self.rx;
        let dy = (p.y - self.center.y) / self.ry;
        dx * dx + dy * dy <= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_ops() {
        let p = Point2::new(1.0, 2.0);
        assert_eq!(p.distance_to(Point2::new(4.0, 6.0)), 5.0);
        assert_eq!(p.translate(2.0, -1.0), Point2::new(3.0, 1.0));
        assert_eq!(p.scale(2.0), Point2::new(2.0, 4.0));
        let r = Point2::new(1.0, 0.0).rotate(PI / 2.0);
        assert!((r.x - 0.0).abs() < 1e-12 && (r.y - 1.0).abs() < 1e-12);
        assert_eq!(p.rotate_around(PI, Point2::new(1.0, 2.0)), Point2::new(1.0, 2.0));
    }

    #[test]
    fn circle() {
        let c = Circle::new(Point2::new(0.0, 0.0), 2.0);
        assert!((c.area() - 4.0 * PI).abs() < 1e-12);
        assert!((c.perimeter() - 4.0 * PI).abs() < 1e-12);
        assert!(c.contains(Point2::new(1.0, 1.0)));
        assert!(!c.contains(Point2::new(2.0, 2.0)));
        // intersection: unit circles at (0,0) and (1,0) cross at (0.5, ±√3/2)
        let a = Circle::new(Point2::new(0.0, 0.0), 1.0);
        let b = Circle::new(Point2::new(1.0, 0.0), 1.0);
        assert!(a.intersects(&b));
        let pts = a.intersection_points(&b);
        assert_eq!(pts.len(), 2);
        for p in &pts {
            assert!((p.x - 0.5).abs() < 1e-12);
            assert!((p.y.abs() - 3.0f64.sqrt() / 2.0).abs() < 1e-12);
        }
        // tangent circles: one point
        let c = Circle::new(Point2::new(2.0, 0.0), 1.0);
        assert_eq!(a.intersection_points(&c).len(), 1);
        // separated: none
        let d = Circle::new(Point2::new(5.0, 0.0), 1.0);
        assert!(!a.intersects(&d));
        assert!(a.intersection_points(&d).is_empty());
    }

    #[test]
    fn triangle() {
        let t = Triangle::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0), Point2::new(0.0, 3.0));
        assert_eq!(t.area(), 6.0);
        assert_eq!(t.perimeter(), 12.0);
        let c = t.centroid();
        assert!((c.x - 4.0 / 3.0).abs() < 1e-12 && (c.y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rectangle() {
        let r = Rectangle::new(0.0, 0.0, 3.0, 4.0);
        assert_eq!(r.area(), 12.0);
        assert_eq!(r.perimeter(), 14.0);
        assert_eq!(r.centroid(), Point2::new(1.5, 2.0));
        assert!(r.contains(Point2::new(1.0, 2.0)));
        assert!(!r.contains(Point2::new(-1.0, 0.0)));
        assert!(r.intersects(&Rectangle::new(2.0, 2.0, 2.0, 2.0)));
        assert!(!r.intersects(&Rectangle::new(4.0, 4.0, 1.0, 1.0)));
    }

    #[test]
    fn polygon() {
        let sq = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ]);
        assert_eq!(sq.area(), 4.0);
        assert_eq!(sq.perimeter(), 8.0);
        assert_eq!(sq.centroid(), Point2::new(1.0, 1.0));
    }

    #[test]
    fn ellipse() {
        let e = Ellipse::new(Point2::new(0.0, 0.0), 2.0, 1.0);
        assert!((e.area() - 2.0 * PI).abs() < 1e-12);
        assert!(e.contains(Point2::new(1.9, 0.0)));
        assert!(!e.contains(Point2::new(2.1, 0.0)));
        assert!(!e.contains(Point2::new(0.0, 1.1)));
    }
}
