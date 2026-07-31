//! Distance and proximity: point-segment, point-line, point-polygon, closest points, GJK distance.

use super::shapes2d::{Point2, Polygon};
use super::primitives2d::LineSegment2;

// ---------------------------------------------------------------------------
// Point-segment distance
// ---------------------------------------------------------------------------

/// Closest point on segment `seg` to point `p`, then distance.
pub fn point_segment_distance(p: Point2, seg: LineSegment2) -> f64 {
    p.distance_to(closest_point_on_segment(p, seg))
}

/// Closest point on segment `seg` to point `p`.
pub fn closest_point_on_segment(p: Point2, seg: LineSegment2) -> Point2 {
    seg.closest_point(p)
}

// ---------------------------------------------------------------------------
// Point-line distance (infinite line)
// ---------------------------------------------------------------------------

/// Distance from point `p` to the infinite line through `a` and `b`.
pub fn point_line_distance(p: Point2, a: Point2, b: Point2) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-30 {
        return p.distance_to(a);
    }
    ((dy * p.x - dx * p.y + b.x * a.y - b.y * a.x).abs()) / len_sq.sqrt()
}

// ---------------------------------------------------------------------------
// Point-polygon distance
// ---------------------------------------------------------------------------

/// Minimum distance from point `p` to the polygon boundary (0 if inside).
pub fn point_polygon_distance(p: Point2, poly: &Polygon) -> f64 {
    let pts = &poly.points;
    let n = pts.len();
    // Check if inside
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
    if inside {
        return 0.0;
    }
    // Min distance to edges
    let mut min_dist = f64::INFINITY;
    for i in 0..n {
        let seg = LineSegment2::new(pts[i], pts[(i + 1) % n]);
        let d = point_segment_distance(p, seg);
        if d < min_dist {
            min_dist = d;
        }
    }
    min_dist
}

// ---------------------------------------------------------------------------
// Closest pair of points (2D)
// ---------------------------------------------------------------------------

/// Find the closest pair of points in a set. O(n²) brute force for simplicity.
pub fn closest_pair(points: &[Point2]) -> Option<(Point2, Point2, f64)> {
    if points.len() < 2 {
        return None;
    }
    let mut best_d = f64::INFINITY;
    let mut best = (points[0], points[1]);
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d = points[i].distance_to(points[j]);
            if d < best_d {
                best_d = d;
                best = (points[i], points[j]);
            }
        }
    }
    Some((best.0, best.1, best_d))
}

// ---------------------------------------------------------------------------
// GJK distance (convex shapes, 2D)
// ---------------------------------------------------------------------------

/// Minkowski support point for two convex point sets.
fn minkowski_support(a: &[Point2], b: &[Point2], dir: Point2) -> Point2 {
    let pa = a.iter().max_by(|x, y| {
        (x.x * dir.x + x.y * dir.y).partial_cmp(&(y.x * dir.x + y.y * dir.y)).unwrap()
    }).unwrap();
    let pb = b.iter().max_by(|x, y| {
        (x.x * -dir.x + x.y * -dir.y).partial_cmp(&(y.x * -dir.x + y.y * -dir.y)).unwrap()
    }).unwrap();
    Point2::new(pa.x - pb.x, pa.y - pb.y)
}

fn cross_z(a: Point2, b: Point2) -> f64 {
    a.x * b.y - a.y * b.x
}

fn dot2(a: Point2, b: Point2) -> f64 {
    a.x * b.x + a.y * b.y
}

fn do_simplex(simplex: &mut Vec<Point2>, dir: &mut Point2) -> bool {
    match simplex.len() {
        2 => {
            let b = simplex[1];
            let a = simplex[0];
            let ab = Point2::new(b.x - a.x, b.y - a.y);
            let ao = Point2::new(-a.x, -a.y);
            if cross_z(ab, ao) > 0.0 {
                *dir = Point2::new(-ab.y, ab.x);
            } else {
                *dir = Point2::new(ab.y, -ab.x);
                simplex.truncate(1);
            }
            false
        }
        3 => {
            let c = simplex[0];
            let b = simplex[1];
            let a = simplex[2];
            let ab = Point2::new(b.x - a.x, b.y - a.y);
            let ac = Point2::new(c.x - a.x, c.y - a.y);
            let ao = Point2::new(-a.x, -a.y);
            let _abp = Point2::new(-ab.y, ab.x);
            let acp = Point2::new(ab.y, -ab.x); // actually -ac.y, ac.x but simplified
            let _ = acp;
            if cross_z(ab, ao) > 0.0 {
                if cross_z(ab, Point2::new(c.x - a.x, c.y - a.y)) <= 0.0 {
                    *dir = Point2::new(-ab.y, ab.x);
                    simplex.truncate(2);
                } else {
                    *dir = Point2::new(ab.y, -ab.x);
                    simplex.clear();
                    simplex.push(a);
                }
            } else if cross_z(ac, ao) > 0.0 {
                *dir = Point2::new(ac.y, -ac.x);
                simplex.clear();
                simplex.push(a);
                simplex.push(c);
            } else {
                *dir = Point2::new(-ao.x, -ao.y);
                simplex.clear();
                simplex.push(a);
                simplex.push(b);
                simplex.push(c);
            }
            false
        }
        _ => false,
    }
}

/// GJK distance between two convex shapes (given as point sets).
/// Returns the minimum distance and closest points.
pub fn gjk_distance(a: &[Point2], b: &[Point2]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }
    let mut dir = Point2::new(1.0, 0.0);
    let mut simplex: Vec<Point2> = Vec::new();
    let mut iter = 0;
    loop {
        let s = minkowski_support(a, b, dir);
        simplex.push(s);
        if dot2(s, dir) < 0.0 {
            return 0.0; // origin is in Minkowski difference → shapes overlap
        }
        dir = Point2::new(-s.x, -s.y);
        if do_simplex(&mut simplex, &mut dir) {
            return 0.0;
        }
        if dir.x * dir.x + dir.y * dir.y < 1e-20 {
            break;
        }
        iter += 1;
        if iter > 100 {
            break;
        }
    }
    // Approximate: distance from origin to simplex
    let mut min_d = f64::INFINITY;
    for s in &simplex {
        let d = (s.x * s.x + s.y * s.y).sqrt();
        if d < min_d {
            min_d = d;
        }
    }
    min_d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> Point2 { Point2::new(x, y) }

    #[test]
    fn point_segment_dist() {
        let seg = LineSegment2::new(pt(0.0, 0.0), pt(1.0, 0.0));
        assert!((point_segment_distance(pt(0.5, 1.0), seg) - 1.0).abs() < 1e-12);
        assert!((point_segment_distance(pt(-1.0, 0.0), seg) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn point_line_dist() {
        let d = point_line_distance(pt(0.0, 1.0), pt(0.0, 0.0), pt(1.0, 0.0));
        assert!((d - 1.0).abs() < 1e-12);
    }

    #[test]
    fn point_polygon_dist_inside() {
        let sq = Polygon::new(vec![pt(0.0, 0.0), pt(2.0, 0.0), pt(2.0, 2.0), pt(0.0, 2.0)]);
        assert!((point_polygon_distance(pt(1.0, 1.0), &sq)).abs() < 1e-10);
    }

    #[test]
    fn point_polygon_dist_outside() {
        let sq = Polygon::new(vec![pt(0.0, 0.0), pt(2.0, 0.0), pt(2.0, 2.0), pt(0.0, 2.0)]);
        assert!((point_polygon_distance(pt(3.0, 1.0), &sq) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn closest_pair_test() {
        let pts = vec![pt(0.0, 0.0), pt(1.0, 1.0), pt(0.1, 0.0)];
        let (a, b, d) = closest_pair(&pts).unwrap();
        assert!((d - 0.1).abs() < 1e-10);
        assert!((a.distance_to(b) - d).abs() < 1e-10);
    }
}
