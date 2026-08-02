//! Collision and intersection: point-in-polygon, segment-polygon, circle-polygon,
//! polygon-polygon (SAT), convex hull, convexity test.

use super::shapes2d::{Circle, Point2, Polygon};
use super::primitives2d::LineSegment2;

// ---------------------------------------------------------------------------
// Point in polygon
// ---------------------------------------------------------------------------

/// Ray-casting algorithm: does a horizontal ray from `p` cross the polygon edge an odd number of times?
pub fn point_in_polygon(p: Point2, poly: &Polygon) -> bool {
    let pts = &poly.points;
    let n = pts.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let yi_gt = pts[i].y > p.y;
        let yj_gt = pts[j].y > p.y;
        if yi_gt != yj_gt {
            let x_intersect = (pts[j].x - pts[i].x) * (p.y - pts[i].y) / (pts[j].y - pts[i].y) + pts[i].x;
            if p.x < x_intersect {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Winding number algorithm: returns winding number of polygon around `p`.
/// 0 = outside, nonzero = inside (with winding semantics).
pub fn winding_number(p: Point2, poly: &Polygon) -> i32 {
    let pts = &poly.points;
    let n = pts.len();
    let mut wn = 0i32;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        if a.y <= p.y {
            if b.y > p.y {
                if cross_2d(a, b, p) > 0.0 {
                    wn += 1;
                }
            }
        } else if b.y <= p.y {
            if cross_2d(a, b, p) < 0.0 {
                wn -= 1;
            }
        }
    }
    wn
}

fn cross_2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

// ---------------------------------------------------------------------------
// Segment vs polygon
// ---------------------------------------------------------------------------

/// Whether a line segment intersects a polygon.
pub fn segment_intersects_polygon(seg: LineSegment2, poly: &Polygon) -> bool {
    let pts = &poly.points;
    let n = pts.len();
    for i in 0..n {
        let edge = LineSegment2::new(pts[i], pts[(i + 1) % n]);
        if segments_intersect(seg, edge) {
            return true;
        }
    }
    point_in_polygon(seg.a, poly) || point_in_polygon(seg.b, poly)
}

/// Whether two line segments intersect (proper or improper).
pub fn segments_intersect(a: LineSegment2, b: LineSegment2) -> bool {
    let d1 = cross_2d(a.a, a.b, b.a);
    let d2 = cross_2d(a.a, a.b, b.b);
    let d3 = cross_2d(b.a, b.b, a.a);
    let d4 = cross_2d(b.a, b.b, a.b);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    // Collinear cases
    if d1.abs() < 1e-10 && on_segment(a, b.a) { return true; }
    if d2.abs() < 1e-10 && on_segment(a, b.b) { return true; }
    if d3.abs() < 1e-10 && on_segment(b, a.a) { return true; }
    if d4.abs() < 1e-10 && on_segment(b, a.b) { return true; }
    false
}

fn on_segment(seg: LineSegment2, p: Point2) -> bool {
    p.x >= seg.a.x.min(seg.b.x) - 1e-10
        && p.x <= seg.a.x.max(seg.b.x) + 1e-10
        && p.y >= seg.a.y.min(seg.b.y) - 1e-10
        && p.y <= seg.a.y.max(seg.b.y) + 1e-10
}

// ---------------------------------------------------------------------------
// Circle vs polygon
// ---------------------------------------------------------------------------

/// Whether a circle intersects a polygon (edge or interior).
pub fn circle_intersects_polygon(circ: Circle, poly: &Polygon) -> bool {
    if point_in_polygon(circ.center, poly) {
        return true;
    }
    let pts = &poly.points;
    let n = pts.len();
    for i in 0..n {
        let edge = LineSegment2::new(pts[i], pts[(i + 1) % n]);
        let closest = edge.closest_point(circ.center);
        if closest.distance_to(circ.center) <= circ.radius {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Polygon vs polygon (SAT)
// ---------------------------------------------------------------------------

/// Separating Axis Theorem: do two convex polygons overlap?
/// Panics if either polygon is concave (SAT is only valid for convex polygons).
pub fn polygons_intersect(poly_a: &Polygon, poly_b: &Polygon) -> bool {
    assert!(is_convex(poly_a), "polygons_intersect requires convex polygons (SAT)");
    assert!(is_convex(poly_b), "polygons_intersect requires convex polygons (SAT)");
    let axes_a = sat_axes(&poly_a.points);
    let axes_b = sat_axes(&poly_b.points);
    for axis in axes_a.iter().chain(axes_b.iter()) {
        if let Some((min_a, max_a)) = project_polygon(&poly_a.points, *axis) {
            if let Some((min_b, max_b)) = project_polygon(&poly_b.points, *axis) {
                if max_a < min_b || max_b < min_a {
                    return false;
                }
            }
        }
    }
    true
}

fn sat_axes(pts: &[Point2]) -> Vec<Point2> {
    let n = pts.len();
    let mut axes = Vec::with_capacity(n);
    for i in 0..n {
        let edge = Point2::new(
            pts[(i + 1) % n].x - pts[i].x,
            pts[(i + 1) % n].y - pts[i].y,
        );
        // Perpendicular (normal)
        axes.push(Point2::new(-edge.y, edge.x));
    }
    axes
}

fn project_polygon(pts: &[Point2], axis: Point2) -> Option<(f64, f64)> {
    if pts.is_empty() {
        return None;
    }
    let mut min = pts[0].x * axis.x + pts[0].y * axis.y;
    let mut max = min;
    for p in &pts[1..] {
        let proj = p.x * axis.x + p.y * axis.y;
        if proj < min { min = proj; }
        if proj > max { max = proj; }
    }
    Some((min, max))
}

// ---------------------------------------------------------------------------
// Convex hull
// ---------------------------------------------------------------------------

/// Andrew's monotone chain convex hull algorithm.
pub fn convex_hull(points: &[Point2]) -> Vec<Point2> {
    let mut pts = points.to_vec();
    if pts.len() <= 2 {
        return pts;
    }
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut lower: Vec<Point2> = Vec::new();
    for p in &pts {
        while lower.len() >= 2 {
            let n = lower.len();
            if cross_2d(lower[n - 2], lower[n - 1], *p) <= 0.0 {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(*p);
    }
    let mut upper: Vec<Point2> = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 {
            let n = upper.len();
            if cross_2d(upper[n - 2], upper[n - 1], *p) <= 0.0 {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

// ---------------------------------------------------------------------------
// Convexity test
// ---------------------------------------------------------------------------

/// Whether a simple polygon is convex.
pub fn is_convex(poly: &Polygon) -> bool {
    let pts = &poly.points;
    let n = pts.len();
    if n < 3 {
        return false;
    }
    let mut sign = 0i8;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        let c = pts[(i + 2) % n];
        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if cross.abs() < 1e-12 {
            continue;
        }
        let s = if cross > 0.0 { 1 } else { -1 };
        if sign != 0 && s != sign {
            return false;
        }
        sign = s;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::shapes2d::Polygon;

    fn pt(x: f64, y: f64) -> Point2 { Point2::new(x, y) }

    fn square() -> Polygon {
        Polygon::new(vec![pt(0.0, 0.0), pt(2.0, 0.0), pt(2.0, 2.0), pt(0.0, 2.0)])
    }

    #[test]
    fn point_in_polygon_test() {
        let sq = square();
        assert!(point_in_polygon(pt(1.0, 1.0), &sq));
        assert!(!point_in_polygon(pt(3.0, 3.0), &sq));
    }

    #[test]
    fn winding_number_test() {
        let sq = square();
        assert_ne!(winding_number(pt(1.0, 1.0), &sq), 0);
        assert_eq!(winding_number(pt(3.0, 3.0), &sq), 0);
    }

    #[test]
    fn segment_polygon_intersect() {
        let sq = square();
        let seg = LineSegment2::new(pt(-1.0, 1.0), pt(3.0, 1.0));
        assert!(segment_intersects_polygon(seg, &sq));
        let seg2 = LineSegment2::new(pt(3.0, 3.0), pt(5.0, 5.0));
        assert!(!segment_intersects_polygon(seg2, &sq));
    }

    #[test]
    fn circle_polygon_intersect() {
        let sq = square();
        let c = Circle::new(pt(1.0, 1.0), 0.5);
        assert!(circle_intersects_polygon(c, &sq));
        let c2 = Circle::new(pt(5.0, 5.0), 0.5);
        assert!(!circle_intersects_polygon(c2, &sq));
    }

    #[test]
    fn polygon_intersect_sat() {
        let a = square();
        let b = Polygon::new(vec![pt(1.0, 1.0), pt(3.0, 1.0), pt(3.0, 3.0), pt(1.0, 3.0)]);
        assert!(polygons_intersect(&a, &b));
        let c = Polygon::new(vec![pt(5.0, 5.0), pt(7.0, 5.0), pt(7.0, 7.0), pt(5.0, 7.0)]);
        assert!(!polygons_intersect(&a, &c));
    }

    #[test]
    fn convex_hull_test() {
        let pts = vec![pt(0.0, 0.0), pt(1.0, 1.0), pt(2.0, 0.0), pt(1.0, -1.0), pt(1.0, 0.5)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn convexity_test() {
        assert!(is_convex(&square()));
        let concave = Polygon::new(vec![pt(0.0, 0.0), pt(2.0, 0.0), pt(1.0, 0.5), pt(2.0, 2.0), pt(0.0, 2.0)]);
        assert!(!is_convex(&concave));
    }
}
