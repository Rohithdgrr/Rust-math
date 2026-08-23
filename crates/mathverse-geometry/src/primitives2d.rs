//! 2D primitives: line segment, ray, arc, sector, circular segment, polyline, Bézier curve.

use mathverse_core::constants::TAU;

use super::shapes2d::Point2;

// ---------------------------------------------------------------------------
// LineSegment2
// ---------------------------------------------------------------------------

/// Directed line segment from `a` to `b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSegment2 {
    pub a: Point2,
    pub b: Point2,
}

impl LineSegment2 {
    #[must_use]
    pub const fn new(a: Point2, b: Point2) -> Self {
        Self { a, b }
    }
    #[must_use]
    pub fn length(self) -> f64 {
        self.a.distance_to(self.b)
    }
    #[must_use]
    pub const fn midpoint(self) -> Point2 {
        Point2::new(f64::midpoint(self.a.x, self.b.x), f64::midpoint(self.a.y, self.b.y))
    }
    /// Direction angle in radians.
    #[must_use]
    pub fn angle(self) -> f64 {
        (self.b.y - self.a.y).atan2(self.b.x - self.a.x)
    }
    /// Closest point on this segment to `p`.
    #[must_use]
    pub fn closest_point(self, p: Point2) -> Point2 {
        let dx = self.b.x - self.a.x;
        let dy = self.b.y - self.a.y;
        let len_sq = dy.mul_add(dy, dx * dx);
        if len_sq < 1e-30 {
            return self.a;
        }
        let t = (p.y - self.a.y).mul_add(dy, (p.x - self.a.x) * dx) / len_sq;
        let t = t.clamp(0.0, 1.0);
        Point2::new(self.a.x + t * dx, self.a.y + t * dy)
    }
    /// Distance from point `p` to this segment.
    #[must_use]
    pub fn distance_to_point(self, p: Point2) -> f64 {
        p.distance_to(self.closest_point(p))
    }
}

// ---------------------------------------------------------------------------
// Ray2
// ---------------------------------------------------------------------------

/// Ray starting at `origin` going in direction `dir`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray2 {
    pub origin: Point2,
    pub dir: Point2,
}

impl Ray2 {
    /// Panics if `dir` is zero.
    #[must_use]
    pub fn new(origin: Point2, dir: Point2) -> Self {
        assert!(
            dir.y.mul_add(dir.y, dir.x * dir.x) > 1e-30,
            "ray direction must be non-zero"
        );
        Self { origin, dir }
    }
    #[must_use]
    pub fn point_at(self, t: f64) -> Point2 {
        Point2::new(t.mul_add(self.dir.x, self.origin.x), t.mul_add(self.dir.y, self.origin.y))
    }
    /// Distance from point `p` to the ray (closest point on the half-line).
    #[must_use]
    pub fn distance_to_point(self, p: Point2) -> f64 {
        let vx = p.x - self.origin.x;
        let vy = p.y - self.origin.y;
        let t = vy.mul_add(self.dir.y, vx * self.dir.x) / self.dir.y.mul_add(self.dir.y, self.dir.x * self.dir.x);
        if t <= 0.0 {
            return p.distance_to(self.origin);
        }
        p.distance_to(self.point_at(t))
    }
}

// ---------------------------------------------------------------------------
// Arc
// ---------------------------------------------------------------------------

/// Circular arc defined by center, radius, start angle, and sweep angle (radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc {
    pub center: Point2,
    pub radius: f64,
    pub start: f64,
    pub sweep: f64,
}

impl Arc {
    /// Panics if `radius < 0`. `sweep` may be negative (clockwise).
    #[must_use]
    pub fn new(center: Point2, radius: f64, start: f64, sweep: f64) -> Self {
        assert!(radius >= 0.0, "arc radius must be non-negative");
        Self {
            center,
            radius,
            start,
            sweep,
        }
    }
    #[must_use]
    pub fn start_point(self) -> Point2 {
        Point2::new(
            self.radius.mul_add(self.start.cos(), self.center.x),
            self.radius.mul_add(self.start.sin(), self.center.y),
        )
    }
    #[must_use]
    pub fn end_point(self) -> Point2 {
        let a = self.start + self.sweep;
        Point2::new(
            self.radius.mul_add(a.cos(), self.center.x),
            self.radius.mul_add(a.sin(), self.center.y),
        )
    }
    /// Arc length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.radius * self.sweep.abs()
    }
    /// Area of the arc (sector minus triangle if sweep < 2π).
    #[must_use]
    pub fn area(self) -> f64 {
        0.5 * self.radius * self.radius * self.sweep.abs()
    }
    /// Whether point `p` lies inside the arc (within the sector and on the arc).
    #[must_use]
    pub fn contains(self, p: Point2) -> bool {
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;
        let dist = dx.hypot(dy);
        if (dist - self.radius).abs() > 1e-10 {
            return false;
        }
        let mut angle = dy.atan2(dx);
        let mut start = self.start;
        let mut end = self.start + self.sweep;
        // Normalize angles to [0, 2π)
        while angle < 0.0 {
            angle += TAU;
        }
        while angle >= TAU {
            angle -= TAU;
        }
        while start < 0.0 {
            start += TAU;
        }
        while start >= TAU {
            start -= TAU;
        }
        while end < 0.0 {
            end += TAU;
        }
        while end >= TAU {
            end -= TAU;
        }
        if self.sweep >= 0.0 {
            if start <= end {
                angle >= start && angle <= end
            } else {
                angle >= start || angle <= end
            }
        } else {
            if end <= start {
                angle >= end && angle <= start
            } else {
                angle >= end || angle <= start
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sector
// ---------------------------------------------------------------------------

/// Circular sector (pie slice) defined by center, radius, start angle, sweep angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sector {
    pub center: Point2,
    pub radius: f64,
    pub start: f64,
    pub sweep: f64,
}

impl Sector {
    #[must_use]
    pub fn new(center: Point2, radius: f64, start: f64, sweep: f64) -> Self {
        assert!(radius >= 0.0, "sector radius must be non-negative");
        Self {
            center,
            radius,
            start,
            sweep,
        }
    }
    #[must_use]
    pub fn area(self) -> f64 {
        0.5 * self.radius * self.radius * self.sweep.abs()
    }
    #[must_use]
    pub fn perimeter(self) -> f64 {
        self.radius.mul_add(self.sweep.abs(), 2.0 * self.radius)
    }
    #[must_use]
    pub fn contains(self, p: Point2) -> bool {
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;
        let dist = dx.hypot(dy);
        if dist > self.radius + 1e-10 {
            return false;
        }
        let angle = dy.atan2(dx);
        let start = self.start;
        let sweep = self.sweep;
        if sweep >= 0.0 {
            let mut a = angle - start;
            while a < 0.0 {
                a += TAU;
            }
            while a >= TAU {
                a -= TAU;
            }
            a <= sweep
        } else {
            let mut a = angle - start;
            while a < 0.0 {
                a += TAU;
            }
            while a >= TAU {
                a -= TAU;
            }
            a >= sweep + TAU || a == 0.0
        }
    }
    /// Arc portion of the sector.
    #[must_use]
    pub fn arc(self) -> Arc {
        Arc::new(self.center, self.radius, self.start, self.sweep)
    }
}

// ---------------------------------------------------------------------------
// CircularSegment
// ---------------------------------------------------------------------------

/// Region between a chord and the arc it subtends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularSegment {
    pub center: Point2,
    pub radius: f64,
    pub start: f64,
    pub sweep: f64,
}

impl CircularSegment {
    #[must_use]
    pub fn new(center: Point2, radius: f64, start: f64, sweep: f64) -> Self {
        assert!(radius >= 0.0, "segment radius must be non-negative");
        Self {
            center,
            radius,
            start,
            sweep,
        }
    }
    /// Area = sector area - triangle area.
    #[must_use]
    pub fn area(self) -> f64 {
        let sector = 0.5 * self.radius * self.radius * self.sweep.abs();
        let triangle =
            0.5 * self.radius * self.radius * self.sweep.abs().sin();
        sector - triangle
    }
    /// Chord length.
    #[must_use]
    pub fn chord_length(self) -> f64 {
        2.0 * self.radius * (self.sweep.abs() / 2.0).sin()
    }
    /// Arc length.
    #[must_use]
    pub fn arc_length(self) -> f64 {
        self.radius * self.sweep.abs()
    }
    /// Perimeter = chord + arc.
    #[must_use]
    pub fn perimeter(self) -> f64 {
        self.chord_length() + self.arc_length()
    }
}

// ---------------------------------------------------------------------------
// Polyline
// ---------------------------------------------------------------------------

/// Open or closed polyline (ordered vertex list).
#[derive(Debug, Clone, PartialEq)]
pub struct Polyline {
    pub points: Vec<Point2>,
    pub closed: bool,
}

impl Polyline {
    #[must_use]
    pub const fn open(points: Vec<Point2>) -> Self {
        Self {
            points,
            closed: false,
        }
    }
    #[must_use]
    pub const fn closed(points: Vec<Point2>) -> Self {
        Self {
            points,
            closed: true,
        }
    }
    #[must_use]
    pub fn total_length(&self) -> f64 {
        self.points
            .windows(2)
            .map(|w| w[0].distance_to(w[1]))
            .sum::<f64>()
            + if self.closed && self.points.len() > 1 {
                self.points.last().unwrap().distance_to(self.points[0])
            } else {
                0.0
            }
    }
    /// Point at parameter `t` ∈ [0, 1] along the polyline.
    #[must_use]
    pub fn point_at(&self, t: f64) -> Point2 {
        let total = self.total_length();
        let target = t * total;
        let mut acc = 0.0;
        for w in self.points.windows(2) {
            let seg = w[0].distance_to(w[1]);
            if acc + seg >= target {
                let frac = if seg > 1e-30 {
                    (target - acc) / seg
                } else {
                    0.0
                };
                return Point2::new(
                    w[0].x + frac * (w[1].x - w[0].x),
                    w[0].y + frac * (w[1].y - w[0].y),
                );
            }
            acc += seg;
        }
        if self.closed && self.points.len() > 1 {
            *self.points.last().unwrap()
        } else {
            *self.points.last().unwrap()
        }
    }
}

// ---------------------------------------------------------------------------
// BezierCurve
// ---------------------------------------------------------------------------

/// Cubic Bézier curve with 4 control points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BezierCurve {
    pub p0: Point2,
    pub p1: Point2,
    pub p2: Point2,
    pub p3: Point2,
}

impl BezierCurve {
    #[must_use]
    pub const fn new(p0: Point2, p1: Point2, p2: Point2, p3: Point2) -> Self {
        Self { p0, p1, p2, p3 }
    }
    /// Evaluate at parameter `t` ∈ [0, 1].
    #[must_use]
    pub fn point_at(self, t: f64) -> Point2 {
        let u = 1.0 - t;
        let u2 = u * u;
        let u3 = u2 * u;
        let t2 = t * t;
        let t3 = t2 * t;
        Point2::new(
            t3.mul_add(self.p3.x, (3.0 * u * t2).mul_add(self.p2.x, (3.0 * u2 * t).mul_add(self.p1.x, u3 * self.p0.x))),
            t3.mul_add(self.p3.y, (3.0 * u * t2).mul_add(self.p2.y, (3.0 * u2 * t).mul_add(self.p1.y, u3 * self.p0.y))),
        )
    }
    /// Approximate length via adaptive subdivision.
    #[must_use]
    pub fn length(self) -> f64 {
        fn approx_len(p0: Point2, p1: Point2, p2: Point2, p3: Point2, depth: u32) -> f64 {
            let d = p0.distance_to(p3);
            let chord = d;
            let poly = p0.distance_to(p1) + p1.distance_to(p2) + p2.distance_to(p3);
            if (poly - chord).abs() < 1e-6 || depth > 10 {
                return f64::midpoint(chord, poly);
            }
            let m01 = Point2::new(
                f64::midpoint(p0.x, p1.x),
                f64::midpoint(p0.y, p1.y),
            );
            let m12 = Point2::new(
                f64::midpoint(p1.x, p2.x),
                f64::midpoint(p1.y, p2.y),
            );
            let m23 = Point2::new(
                f64::midpoint(p2.x, p3.x),
                f64::midpoint(p2.y, p3.y),
            );
            let m012 = Point2::new(
                f64::midpoint(m01.x, m12.x),
                f64::midpoint(m01.y, m12.y),
            );
            let m123 = Point2::new(
                f64::midpoint(m12.x, m23.x),
                f64::midpoint(m12.y, m23.y),
            );
            let mid = Point2::new(
                f64::midpoint(m012.x, m123.x),
                f64::midpoint(m012.y, m123.y),
            );
            approx_len(p0, m01, m012, mid, depth + 1)
                + approx_len(mid, m123, m23, p3, depth + 1)
        }
        approx_len(self.p0, self.p1, self.p2, self.p3, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mathverse_core::constants::{PI, TAU};
    use std::f64::consts::FRAC_PI_2;

    fn pt(x: f64, y: f64) -> Point2 {
        Point2::new(x, y)
    }

    #[test]
    fn segment_basics() {
        let s = LineSegment2::new(pt(0.0, 0.0), pt(3.0, 4.0));
        assert!((s.length() - 5.0).abs() < 1e-12);
        assert_eq!(s.midpoint(), pt(1.5, 2.0));
    }

    #[test]
    fn segment_closest_point() {
        let s = LineSegment2::new(pt(0.0, 0.0), pt(1.0, 0.0));
        assert_eq!(s.closest_point(pt(0.5, 3.0)), pt(0.5, 0.0));
        assert_eq!(s.closest_point(pt(-1.0, 0.0)), pt(0.0, 0.0));
        assert_eq!(s.closest_point(pt(2.0, 0.0)), pt(1.0, 0.0));
    }

    #[test]
    fn ray_distance() {
        let r = Ray2::new(pt(0.0, 0.0), pt(1.0, 0.0));
        assert!((r.distance_to_point(pt(0.0, 1.0)) - 1.0).abs() < 1e-12);
        assert!((r.distance_to_point(pt(-1.0, 0.0)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn arc_area() {
        let a = Arc::new(pt(0.0, 0.0), 2.0, 0.0, core::f64::consts::FRAC_PI_2);
        assert!((a.area() - PI).abs() < 1e-12);
    }

    #[test]
    fn sector_contains() {
        let s = Sector::new(pt(0.0, 0.0), 1.0, 0.0, FRAC_PI_2);
        assert!(s.contains(pt(0.5, 0.5)));
        assert!(!s.contains(pt(2.0, 2.0)));
    }

    #[test]
    fn segment_area() {
        let s = CircularSegment::new(pt(0.0, 0.0), 1.0, 0.0, PI);
        let expected = PI / 2.0 - 0.0; // sector - triangle = π/2 - 0
        // sector = 0.5*1*π = π/2, triangle = 0.5*1*sin(π) = 0
        assert!((s.area() - expected).abs() < 1e-12);
    }

    #[test]
    fn polyline_length() {
        let pl = Polyline::open(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0)]);
        assert!((pl.total_length() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bezier_at_zero_one() {
        let b = BezierCurve::new(pt(0.0, 0.0), pt(1.0, 2.0), pt(3.0, 2.0), pt(4.0, 0.0));
        assert_eq!(b.point_at(0.0), pt(0.0, 0.0));
        assert_eq!(b.point_at(1.0), pt(4.0, 0.0));
    }
}
