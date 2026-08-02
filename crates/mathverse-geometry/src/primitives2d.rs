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
    pub fn new(a: Point2, b: Point2) -> Self {
        Self { a, b }
    }
    pub fn length(self) -> f64 {
        self.a.distance_to(self.b)
    }
    pub fn midpoint(self) -> Point2 {
        Point2::new((self.a.x + self.b.x) / 2.0, (self.a.y + self.b.y) / 2.0)
    }
    /// Direction angle in radians.
    pub fn angle(self) -> f64 {
        (self.b.y - self.a.y).atan2(self.b.x - self.a.x)
    }
    /// Closest point on this segment to `p`.
    pub fn closest_point(self, p: Point2) -> Point2 {
        let dx = self.b.x - self.a.x;
        let dy = self.b.y - self.a.y;
        let len_sq = dx * dx + dy * dy;
        if len_sq < 1e-30 {
            return self.a;
        }
        let t = ((p.x - self.a.x) * dx + (p.y - self.a.y) * dy) / len_sq;
        let t = t.clamp(0.0, 1.0);
        Point2::new(self.a.x + t * dx, self.a.y + t * dy)
    }
    /// Distance from point `p` to this segment.
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
    pub fn new(origin: Point2, dir: Point2) -> Self {
        assert!(
            (dir.x * dir.x + dir.y * dir.y) > 1e-30,
            "ray direction must be non-zero"
        );
        Self { origin, dir }
    }
    pub fn point_at(self, t: f64) -> Point2 {
        Point2::new(self.origin.x + t * self.dir.x, self.origin.y + t * self.dir.y)
    }
    /// Distance from point `p` to the ray (closest point on the half-line).
    pub fn distance_to_point(self, p: Point2) -> f64 {
        let vx = p.x - self.origin.x;
        let vy = p.y - self.origin.y;
        let t = (vx * self.dir.x + vy * self.dir.y) / (self.dir.x * self.dir.x + self.dir.y * self.dir.y);
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
    pub fn new(center: Point2, radius: f64, start: f64, sweep: f64) -> Self {
        assert!(radius >= 0.0, "arc radius must be non-negative");
        Self {
            center,
            radius,
            start,
            sweep,
        }
    }
    pub fn start_point(self) -> Point2 {
        Point2::new(
            self.center.x + self.radius * self.start.cos(),
            self.center.y + self.radius * self.start.sin(),
        )
    }
    pub fn end_point(self) -> Point2 {
        let a = self.start + self.sweep;
        Point2::new(
            self.center.x + self.radius * a.cos(),
            self.center.y + self.radius * a.sin(),
        )
    }
    /// Arc length.
    pub fn length(self) -> f64 {
        self.radius * self.sweep.abs()
    }
    /// Area of the arc (sector minus triangle if sweep < 2π).
    pub fn area(self) -> f64 {
        0.5 * self.radius * self.radius * self.sweep.abs()
    }
    /// Whether point `p` lies inside the arc (within the sector and on the arc).
    pub fn contains(self, p: Point2) -> bool {
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;
        let dist = (dx * dx + dy * dy).sqrt();
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
    pub fn new(center: Point2, radius: f64, start: f64, sweep: f64) -> Self {
        assert!(radius >= 0.0, "sector radius must be non-negative");
        Self {
            center,
            radius,
            start,
            sweep,
        }
    }
    pub fn area(self) -> f64 {
        0.5 * self.radius * self.radius * self.sweep.abs()
    }
    pub fn perimeter(self) -> f64 {
        2.0 * self.radius + self.radius * self.sweep.abs()
    }
    pub fn contains(self, p: Point2) -> bool {
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;
        let dist = (dx * dx + dy * dy).sqrt();
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
    pub fn area(self) -> f64 {
        let sector = 0.5 * self.radius * self.radius * self.sweep.abs();
        let triangle =
            0.5 * self.radius * self.radius * self.sweep.abs().sin();
        sector - triangle
    }
    /// Chord length.
    pub fn chord_length(self) -> f64 {
        2.0 * self.radius * (self.sweep.abs() / 2.0).sin()
    }
    /// Arc length.
    pub fn arc_length(self) -> f64 {
        self.radius * self.sweep.abs()
    }
    /// Perimeter = chord + arc.
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
    pub fn open(points: Vec<Point2>) -> Self {
        Self {
            points,
            closed: false,
        }
    }
    pub fn closed(points: Vec<Point2>) -> Self {
        Self {
            points,
            closed: true,
        }
    }
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
    pub fn new(p0: Point2, p1: Point2, p2: Point2, p3: Point2) -> Self {
        Self { p0, p1, p2, p3 }
    }
    /// Evaluate at parameter `t` ∈ [0, 1].
    pub fn point_at(self, t: f64) -> Point2 {
        let u = 1.0 - t;
        let u2 = u * u;
        let u3 = u2 * u;
        let t2 = t * t;
        let t3 = t2 * t;
        Point2::new(
            u3 * self.p0.x + 3.0 * u2 * t * self.p1.x + 3.0 * u * t2 * self.p2.x + t3 * self.p3.x,
            u3 * self.p0.y + 3.0 * u2 * t * self.p1.y + 3.0 * u * t2 * self.p2.y + t3 * self.p3.y,
        )
    }
    /// Approximate length via adaptive subdivision.
    pub fn length(self) -> f64 {
        fn approx_len(p0: Point2, p1: Point2, p2: Point2, p3: Point2, depth: u32) -> f64 {
            let d = p0.distance_to(p3);
            let chord = d;
            let poly = p0.distance_to(p1) + p1.distance_to(p2) + p2.distance_to(p3);
            if (poly - chord).abs() < 1e-6 || depth > 10 {
                return (chord + poly) / 2.0;
            }
            let m01 = Point2::new(
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
            );
            let m12 = Point2::new(
                (p1.x + p2.x) / 2.0,
                (p1.y + p2.y) / 2.0,
            );
            let m23 = Point2::new(
                (p2.x + p3.x) / 2.0,
                (p2.y + p3.y) / 2.0,
            );
            let m012 = Point2::new(
                (m01.x + m12.x) / 2.0,
                (m01.y + m12.y) / 2.0,
            );
            let m123 = Point2::new(
                (m12.x + m23.x) / 2.0,
                (m12.y + m23.y) / 2.0,
            );
            let mid = Point2::new(
                (m012.x + m123.x) / 2.0,
                (m012.y + m123.y) / 2.0,
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
