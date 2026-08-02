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
            let abp = Point2::new(-ab.y, ab.x);
            let acp = Point2::new(ac.y, -ac.x);
            if cross_z(ab, ao) > 0.0 {
                if cross_z(ab, ac) <= 0.0 {
                    *dir = abp;
                    simplex.truncate(2);
                } else {
                    *dir = Point2::new(ab.y, -ab.x);
                    simplex.clear();
                    simplex.push(a);
                }
            } else if cross_z(ac, ao) > 0.0 {
                *dir = acp;
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

fn closest_point_on_simplex(simplex: &[Point2]) -> Point2 {
    match simplex.len() {
        1 => simplex[0],
        2 => {
            let a = simplex[0];
            let b = simplex[1];
            let ab = Point2::new(b.x - a.x, b.y - a.y);
            let t = -dot2(a, ab) / dot2(ab, ab);
            let t = t.clamp(0.0, 1.0);
            Point2::new(a.x + t * ab.x, a.y + t * ab.y)
        }
        3 => {
            let a = simplex[0];
            let b = simplex[1];
            let c = simplex[2];
            let ab = Point2::new(b.x - a.x, b.y - a.y);
            let ac = Point2::new(c.x - a.x, c.y - a.y);
            let ao = Point2::new(-a.x, -a.y);
            let d1 = dot2(ab, ao);
            let d2 = dot2(ac, ao);
            if d1 <= 0.0 && d2 <= 0.0 {
                return a;
            }
            let bo = Point2::new(-b.x, -b.y);
            let d3 = dot2(ab, bo);
            let d4 = dot2(ac, bo);
            if d3 >= 0.0 && d4 <= d3 {
                return b;
            }
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                let v = d1 / (d1 - d3);
                return Point2::new(a.x + v * ab.x, a.y + v * ab.y);
            }
            let bo2 = Point2::new(-c.x, -c.y);
            let d5 = dot2(ab, bo2);
            let d6 = dot2(ac, bo2);
            let vb = d5 * d2 - d1 * d6;
            if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                let w = d2 / (d2 - d6);
                return Point2::new(a.x + w * ac.x, a.y + w * ac.y);
            }
            let va = d3 * d6 - d5 * d4;
            if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                let denom = (d4 - d3) + (d5 - d6);
                let u = (d4 - d3) / denom;
                let w = (d5 - d6) / denom;
                return Point2::new(
                    a.x + u * ab.x + w * ac.x,
                    a.y + u * ab.y + w * ac.y,
                );
            }
            let denom = va + vb + vc;
            let u = vb / denom;
            let w = vc / denom;
            Point2::new(
                a.x + u * ab.x + w * ac.x,
                a.y + u * ab.y + w * ac.y,
            )
        }
        _ => Point2::new(0.0, 0.0),
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
    // Distance from origin to closest point on simplex
    let cp = closest_point_on_simplex(&simplex);
    (cp.x * cp.x + cp.y * cp.y).sqrt()
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
