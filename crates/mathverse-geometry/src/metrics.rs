//! Geometry metrics: angle between lines, cross product area, signed area, winding number, moment of inertia.

use mathverse_core::constants::PI;

use super::shapes2d::{Point2, Polygon};
use super::transforms::Transform2D;

// ---------------------------------------------------------------------------
// Angle between vectors
// ---------------------------------------------------------------------------

/// Angle between two vectors in radians [0, π].
pub fn angle_between(a: Point2, b: Point2) -> f64 {
    let dot = a.x * b.x + a.y * b.y;
    let cross = a.x * b.y - a.y * b.x;
    cross.atan2(dot).abs()
}

/// Signed angle from vector `a` to vector `b` in radians (-π, π].
pub fn signed_angle(a: Point2, b: Point2) -> f64 {
    let cross = a.x * b.y - a.y * b.x;
    let dot = a.x * b.x + a.y * b.y;
    cross.atan2(dot)
}

// ---------------------------------------------------------------------------
// Cross product area (2D)
// ---------------------------------------------------------------------------

/// Signed area of the parallelogram formed by vectors `a` and `b`.
pub fn cross_product_area_2d(a: Point2, b: Point2) -> f64 {
    a.x * b.y - a.y * b.x
}

// ---------------------------------------------------------------------------
// Signed area
// ---------------------------------------------------------------------------

/// Signed area of a polygon (positive = counterclockwise).
pub fn signed_area(poly: &Polygon) -> f64 {
    let pts = &poly.points;
    let n = pts.len();
    let mut s = 0.0;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        s += a.x * b.y - b.x * a.y;
    }
    s / 2.0
}

/// Whether polygon vertices are in counterclockwise order.
pub fn is_counterclockwise(poly: &Polygon) -> bool {
    signed_area(poly) > 0.0
}

/// Whether polygon vertices are in clockwise order.
pub fn is_clockwise(poly: &Polygon) -> bool {
    signed_area(poly) < 0.0
}

/// Force polygon to be counterclockwise by reversing if needed.
pub fn ensure_counterclockwise(poly: &Polygon) -> Polygon {
    if signed_area(poly) < 0.0 {
        let mut pts = poly.points.clone();
        pts.reverse();
        Polygon::new(pts)
    } else {
        poly.clone()
    }
}

// ---------------------------------------------------------------------------
// Winding number (re-export from intersection for convenience)
// ---------------------------------------------------------------------------

/// Winding number of polygon around point `p`. 0 = outside.
pub fn winding_number(p: Point2, poly: &Polygon) -> i32 {
    let pts = &poly.points;
    let n = pts.len();
    let mut wn = 0i32;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        if a.y <= p.y {
            if b.y > p.y {
                let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
                if cross > 0.0 {
                    wn += 1;
                }
            }
        } else if b.y <= p.y {
            let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
            if cross < 0.0 {
                wn -= 1;
            }
        }
    }
    wn
}

// ---------------------------------------------------------------------------
// Moment of inertia (2D, about origin)
// ---------------------------------------------------------------------------

/// Moment of inertia of a uniform polygon about the origin.
/// For a polygon with vertices in order, using the shoelace-based formula.
pub fn moment_of_inertia_origin(poly: &Polygon) -> f64 {
    let pts = &poly.points;
    let n = pts.len();
    let mut area2 = 0.0;
    let mut moi = 0.0;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        let cross = a.x * b.y - b.x * a.y;
        area2 += cross;
        moi += cross * (a.x * a.x + a.x * b.x + b.x * b.x + a.y * a.y + a.y * b.y + b.y * b.y);
    }
    moi / (6.0 * area2.abs())
}

/// Moment of inertia of a uniform polygon about its centroid.
pub fn moment_of_inertia_centroid(poly: &Polygon) -> f64 {
    let c = poly.centroid();
    let translated = poly.clone().translate(-c.x, -c.y);
    moment_of_inertia_origin(&translated)
}

/// Moment of inertia of a circle of radius `r` about the origin.
pub fn moment_of_inertia_circle(radius: f64) -> f64 {
    PI * radius.powi(4) / 2.0
}

/// Moment of inertia of a rectangle about its centroid.
pub fn moment_of_inertia_rectangle(width: f64, height: f64) -> f64 {
    width * height * (width * width + height * height) / 12.0
}

/// Moment of inertia of a triangle about its centroid.
pub fn moment_of_inertia_triangle(a: Point2, b: Point2, c: Point2) -> f64 {
    let poly = Polygon::new(vec![a, b, c]);
    moment_of_inertia_centroid(&poly)
}

// ---------------------------------------------------------------------------
// Area between shapes
// ---------------------------------------------------------------------------

/// Approximate area of polygon by Monte Carlo sampling inside `bounds`.
pub fn monte_carlo_area(poly: &Polygon, bounds: (Point2, Point2), samples: usize) -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let (min, max) = bounds;
    let width = max.x - min.x;
    let height = max.y - min.y;
    let mut inside = 0usize;
    let mut hasher = DefaultHasher::new();
    for i in 0..samples {
        i.hash(&mut hasher);
        let h1 = hasher.finish();
        i.wrapping_add(1).hash(&mut hasher);
        let h2 = hasher.finish();
        let x = min.x + (h1 as f64 / u64::MAX as f64) * width;
        let y = min.y + (h2 as f64 / u64::MAX as f64) * height;
        let p = Point2::new(x, y);
        let mut j = poly.points.len() - 1;
        let mut inside_ray = false;
        for (k, pt) in poly.points.iter().enumerate() {
            let yi_gt = pt.y > p.y;
            let yj_gt = poly.points[j].y > p.y;
            if yi_gt != yj_gt {
                let x_intersect = (poly.points[j].x - pt.x) * (p.y - pt.y) / (poly.points[j].y - pt.y) + pt.x;
                if p.x < x_intersect {
                    inside_ray = !inside_ray;
                }
            }
            j = k;
        }
        if inside_ray {
            inside += 1;
        }
    }
    (inside as f64 / samples as f64) * width * height
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::shapes2d::Point2;

    fn pt(x: f64, y: f64) -> Point2 { Point2::new(x, y) }

    #[test]
    fn angle_between_test() {
        let a = pt(1.0, 0.0);
        let b = pt(0.0, 1.0);
        assert!((angle_between(a, b) - FRAC_PI_2).abs() < 1e-12);
    }

    const FRAC_PI_2: f64 = PI / 2.0;

    #[test]
    fn signed_area_test() {
        let sq = Polygon::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 1.0)]);
        assert!(signed_area(&sq) > 0.0);
        assert!(is_counterclockwise(&sq));
    }

    #[test]
    fn moment_of_inertia_circle_test() {
        let moi = moment_of_inertia_circle(1.0);
        assert!((moi - PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn moment_of_inertia_rectangle_test() {
        let moi = moment_of_inertia_rectangle(2.0, 3.0);
        // I = w*h*(w²+h²)/12 = 2*3*(4+9)/12 = 6*13/12 = 6.5
        assert!((moi - 6.5).abs() < 1e-12);
    }
}
