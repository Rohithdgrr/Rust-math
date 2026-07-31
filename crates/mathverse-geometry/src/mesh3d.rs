//! 3D extensions: triangle mesh, AABB tree, ray-triangle (Möller-Trumbore), plane-plane intersection, line-solid intersections.

use super::shapes3d::{Line3, Plane, Point3};
use super::spatial::AABB3;

// ---------------------------------------------------------------------------
// Triangle3
// ---------------------------------------------------------------------------

/// 3D triangle defined by three vertices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle3 {
    pub a: Point3,
    pub b: Point3,
    pub c: Point3,
}

impl Triangle3 {
    pub fn new(a: Point3, b: Point3, c: Point3) -> Self {
        Self { a, b, c }
    }
    pub fn normal(self) -> Point3 {
        let ab = Point3::new(self.b.x - self.a.x, self.b.y - self.a.y, self.b.z - self.a.z);
        let ac = Point3::new(self.c.x - self.a.x, self.c.y - self.a.y, self.c.z - self.a.z);
        cross(ab, ac)
    }
    pub fn area(self) -> f64 {
        norm(self.normal()) / 2.0
    }
    pub fn centroid(self) -> Point3 {
        Point3::new(
            (self.a.x + self.b.x + self.c.x) / 3.0,
            (self.a.y + self.b.y + self.c.y) / 3.0,
            (self.a.z + self.b.z + self.c.z) / 3.0,
        )
    }
}

// ---------------------------------------------------------------------------
// TriangleMesh
// ---------------------------------------------------------------------------

/// Simple triangle mesh (indexed).
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    pub vertices: Vec<Point3>,
    pub indices: Vec<(usize, usize, usize)>,
}

impl TriangleMesh {
    pub fn new(vertices: Vec<Point3>, indices: Vec<(usize, usize, usize)>) -> Self {
        Self { vertices, indices }
    }
    pub fn triangle(&self, i: usize) -> Triangle3 {
        let (a, b, c) = self.indices[i];
        Triangle3::new(self.vertices[a], self.vertices[b], self.vertices[c])
    }
    pub fn num_triangles(&self) -> usize {
        self.indices.len()
    }
    pub fn aabb(&self) -> Option<AABB3> {
        AABB3::from_points(&self.vertices)
    }
    /// Compute normals for each triangle.
    pub fn normals(&self) -> Vec<Point3> {
        (0..self.indices.len())
            .map(|i| self.triangle(i).normal())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AABB Tree
// ---------------------------------------------------------------------------

/// AABB tree node for fast ray intersection.
pub struct AABBTreeNode {
    pub bounds: AABB3,
    pub triangle_index: Option<usize>,
    pub left: Option<Box<AABBTreeNode>>,
    pub right: Option<Box<AABBTreeNode>>,
}

impl AABBTreeNode {
    pub fn leaf(bounds: AABB3, triangle_index: usize) -> Self {
        Self {
            bounds,
            triangle_index: Some(triangle_index),
            left: None,
            right: None,
        }
    }
    pub fn internal(bounds: AABB3, left: AABBTreeNode, right: AABBTreeNode) -> Self {
        Self {
            bounds,
            triangle_index: None,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }
    /// Find the closest triangle hit by a ray. Returns (triangle_index, t).
    pub fn ray_intersect(&self, mesh: &TriangleMesh, origin: Point3, dir: Point3) -> Option<(usize, f64)> {
        if !aabb_ray_intersect(self.bounds, origin, dir) {
            return None;
        }
        if let Some(idx) = self.triangle_index {
            let tri = mesh.triangle(idx);
            if let Some(t) = ray_triangle_intersect(origin, dir, tri) {
                return Some((idx, t));
            }
            return None;
        }
        let mut closest: Option<(usize, f64)> = None;
        if let Some(ref left) = self.left {
            if let Some(hit) = left.ray_intersect(mesh, origin, dir) {
                closest = Some(hit);
            }
        }
        if let Some(ref right) = self.right {
            if let Some(hit) = right.ray_intersect(mesh, origin, dir) {
                if closest.is_none() || hit.1 < closest.unwrap().1 {
                    closest = Some(hit);
                }
            }
        }
        closest
    }
}

/// Build a simple AABB tree from a mesh (median split on longest axis).
pub fn build_aabb_tree(mesh: &TriangleMesh) -> Option<AABBTreeNode> {
    if mesh.indices.is_empty() {
        return None;
    }
    let mut tris: Vec<(usize, AABB3)> = (0..mesh.indices.len())
        .map(|i| {
            let tri = mesh.triangle(i);
            let pts = [tri.a, tri.b, tri.c];
            (i, AABB3::from_points(&pts).unwrap())
        })
        .collect();
    build_tree_recursive(&mut tris)
}

fn build_tree_recursive(tris: &mut [(usize, AABB3)]) -> Option<AABBTreeNode> {
    if tris.is_empty() {
        return None;
    }
    if tris.len() == 1 {
        let (idx, bounds) = tris[0];
        return Some(AABBTreeNode::leaf(bounds, idx));
    }
    // Merge all bounds
    let bounds = tris.iter().map(|(_, b)| *b).reduce(|a, b| a.merge(b)).unwrap();
    // Split on longest axis
    let ext = [
        bounds.max.x - bounds.min.x,
        bounds.max.y - bounds.min.y,
        bounds.max.z - bounds.min.z,
    ];
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] { 0 }
        else if ext[1] >= ext[2] { 1 } else { 2 };
    tris.sort_by(|a, b| {
        let ca = match axis {
            0 => (a.1.min.x + a.1.max.x) / 2.0,
            1 => (a.1.min.y + a.1.max.y) / 2.0,
            _ => (a.1.min.z + a.1.max.z) / 2.0,
        };
        let cb = match axis {
            0 => (b.1.min.x + b.1.max.x) / 2.0,
            1 => (b.1.min.y + b.1.max.y) / 2.0,
            _ => (b.1.min.z + b.1.max.z) / 2.0,
        };
        ca.partial_cmp(&cb).unwrap()
    });
    let mid = tris.len() / 2;
    let right = build_tree_recursive(&mut tris[mid..])?;
    let left = build_tree_recursive(&mut tris[..mid])?;
    Some(AABBTreeNode::internal(bounds, left, right))
}

// ---------------------------------------------------------------------------
// Ray-triangle intersection (Möller-Trumbore)
// ---------------------------------------------------------------------------

/// Möller-Trumbore ray-triangle intersection. Returns `t` if hit (t > 0).
pub fn ray_triangle_intersect(origin: Point3, dir: Point3, tri: Triangle3) -> Option<f64> {
    let eps = 1e-7;
    let ab = Point3::new(tri.b.x - tri.a.x, tri.b.y - tri.a.y, tri.b.z - tri.a.z);
    let ac = Point3::new(tri.c.x - tri.a.x, tri.c.y - tri.a.y, tri.c.z - tri.a.z);
    let h = cross(dir, ac);
    let a = dot(ab, h);
    if a > -eps && a < eps {
        return None;
    }
    let f = 1.0 / a;
    let s = Point3::new(origin.x - tri.a.x, origin.y - tri.a.y, origin.z - tri.a.z);
    let u = f * dot(s, h);
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = cross(s, ab);
    let v = f * dot(dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * dot(ac, q);
    if t > eps {
        Some(t)
    } else {
        None
    }
}

/// AABB-ray intersection (Slab method). Returns `t_min` if hit.
pub fn aabb_ray_intersect(bb: AABB3, origin: Point3, dir: Point3) -> bool {
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;
    let orig = [origin.x, origin.y, origin.z];
    let dir = [dir.x, dir.y, dir.z];
    let min = [bb.min.x, bb.min.y, bb.min.z];
    let max = [bb.max.x, bb.max.y, bb.max.z];
    for i in 0..3 {
        if dir[i].abs() < 1e-30 {
            if orig[i] < min[i] || orig[i] > max[i] {
                return false;
            }
        } else {
            let inv = 1.0 / dir[i];
            let mut t1 = (min[i] - orig[i]) * inv;
            let mut t2 = (max[i] - orig[i]) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            if t1 > tmin { tmin = t1; }
            if t2 < tmax { tmax = t2; }
            if tmin > tmax {
                return false;
            }
        }
    }
    tmax >= 0.0
}

// ---------------------------------------------------------------------------
// Plane-plane intersection
// ---------------------------------------------------------------------------

/// Intersection of two planes: returns a line if they intersect, or None if parallel.
pub fn plane_plane_intersection(p1: Plane, p2: Plane) -> Option<Line3> {
    let n1 = p1.normal;
    let n2 = p2.normal;
    let dir = cross(n1, n2);
    if norm(dir) < 1e-10 {
        return None; // parallel
    }
    // Find a point on the intersection line
    // Solve: n1·p + d1 = 0, n2·p + d2 = 0, set one coord to 0
    let a = [
        [n1.y, n1.z, -p1.d],
        [n2.y, n2.z, -p2.d],
    ];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() > 1e-10 {
        let y = (a[0][2] * a[1][1] - a[0][1] * a[1][2]) / det;
        let z = (a[0][0] * a[1][2] - a[0][2] * a[1][0]) / det;
        return Some(Line3::new(Point3::new(0.0, y, z), dir));
    }
    // Try setting y = 0
    let a = [
        [n1.x, n1.z, -p1.d],
        [n2.x, n2.z, -p2.d],
    ];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() > 1e-10 {
        let x = (a[0][2] * a[1][1] - a[0][1] * a[1][2]) / det;
        let z = (a[0][0] * a[1][2] - a[0][2] * a[1][0]) / det;
        return Some(Line3::new(Point3::new(x, 0.0, z), dir));
    }
    // Try setting z = 0
    let a = [
        [n1.x, n1.y, -p1.d],
        [n2.x, n2.y, -p2.d],
    ];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() > 1e-10 {
        let x = (a[0][2] * a[1][1] - a[0][1] * a[1][2]) / det;
        let y = (a[0][0] * a[1][2] - a[0][2] * a[1][0]) / det;
        return Some(Line3::new(Point3::new(x, y, 0.0), dir));
    }
    None
}

// ---------------------------------------------------------------------------
// Line-solid intersections
// ---------------------------------------------------------------------------

/// Ray-sphere intersection. Returns closest `t > 0` or None.
pub fn ray_sphere_intersect(origin: Point3, dir: Point3, center: Point3, radius: f64) -> Option<f64> {
    let oc = Point3::new(origin.x - center.x, origin.y - center.y, origin.z - center.z);
    let a = dot(dir, dir);
    let b = 2.0 * dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    if t1 > 1e-7 {
        Some(t1)
    } else if t2 > 1e-7 {
        Some(t2)
    } else {
        None
    }
}

/// Ray-AABB intersection. Returns closest `t > 0` or None.
pub fn ray_aabb_intersect(origin: Point3, dir: Point3, bb: AABB3) -> Option<f64> {
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;
    let orig = [origin.x, origin.y, origin.z];
    let dir_arr = [dir.x, dir.y, dir.z];
    let min = [bb.min.x, bb.min.y, bb.min.z];
    let max = [bb.max.x, bb.max.y, bb.max.z];
    for i in 0..3 {
        if dir_arr[i].abs() < 1e-30 {
            if orig[i] < min[i] || orig[i] > max[i] {
                return None;
            }
        } else {
            let inv = 1.0 / dir_arr[i];
            let mut t1 = (min[i] - orig[i]) * inv;
            let mut t2 = (max[i] - orig[i]) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            if t1 > tmin { tmin = t1; }
            if t2 < tmax { tmax = t2; }
            if tmin > tmax {
                return None;
            }
        }
    }
    if tmax < 1e-7 {
        return None;
    }
    Some(if tmin > 1e-7 { tmin } else { tmax })
}

/// Ray-cylinder intersection (infinite cylinder along `axis` through `center`).
pub fn ray_cylinder_intersect(origin: Point3, dir: Point3, center: Point3, axis: Point3, radius: f64) -> Option<f64> {
    let oc = Point3::new(origin.x - center.x, origin.y - center.y, origin.z - center.z);
    let d_dot_ax = dot(dir, axis);
    let oc_dot_ax = dot(oc, axis);
    let proj_d = Point3::new(dir.x - d_dot_ax * axis.x, dir.y - d_dot_ax * axis.y, dir.z - d_dot_ax * axis.z);
    let proj_oc = Point3::new(oc.x - oc_dot_ax * axis.x, oc.y - oc_dot_ax * axis.y, oc.z - oc_dot_ax * axis.z);
    let a = dot(proj_d, proj_d);
    let b = 2.0 * dot(proj_oc, proj_d);
    let c = dot(proj_oc, proj_oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    if t1 > 1e-7 {
        Some(t1)
    } else if t2 > 1e-7 {
        Some(t2)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::shapes3d::{Point3, Plane};

    fn pt3(x: f64, y: f64, z: f64) -> Point3 { Point3::new(x, y, z) }

    #[test]
    fn triangle_normal() {
        let tri = Triangle3::new(pt3(0.0, 0.0, 0.0), pt3(1.0, 0.0, 0.0), pt3(0.0, 1.0, 0.0));
        let n = tri.normal();
        assert!((n.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ray_triangle_hit() {
        // Triangle in the xy-plane at z=0
        let tri = Triangle3::new(pt3(-1.0, -1.0, 0.0), pt3(1.0, -1.0, 0.0), pt3(0.0, 1.0, 0.0));
        let t = ray_triangle_intersect(pt3(0.0, 0.0, -5.0), pt3(0.0, 0.0, 1.0), tri);
        assert!(t.is_some());
        assert!((t.unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn ray_triangle_miss() {
        let tri = Triangle3::new(pt3(-1.0, -1.0, 0.0), pt3(1.0, -1.0, 0.0), pt3(0.0, 1.0, 0.0));
        let t = ray_triangle_intersect(pt3(5.0, 5.0, -5.0), pt3(0.0, 0.0, 1.0), tri);
        assert!(t.is_none());
    }

    #[test]
    fn ray_sphere_hit() {
        let t = ray_sphere_intersect(pt3(0.0, 0.0, -5.0), pt3(0.0, 0.0, 1.0), pt3(0.0, 0.0, 0.0), 1.0);
        assert!(t.is_some());
        assert!((t.unwrap() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn ray_sphere_miss() {
        let t = ray_sphere_intersect(pt3(5.0, 5.0, -5.0), pt3(0.0, 0.0, 1.0), pt3(0.0, 0.0, 0.0), 1.0);
        assert!(t.is_none());
    }

    #[test]
    fn plane_plane_intersect() {
        let p1 = Plane::new(pt3(0.0, 0.0, 1.0), 0.0); // z = 0
        let p2 = Plane::new(pt3(0.0, 1.0, 0.0), 0.0); // y = 0
        let line = plane_plane_intersection(p1, p2).unwrap();
        // Intersection is the x-axis
        assert!((line.dir.x.abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn aabb_tree_build() {
        let mesh = TriangleMesh::new(
            vec![pt3(0.0, 0.0, 0.0), pt3(1.0, 0.0, 0.0), pt3(0.0, 1.0, 0.0), pt3(0.0, 0.0, 1.0)],
            vec![(0, 1, 2), (0, 1, 3)],
        );
        let tree = build_aabb_tree(&mesh);
        assert!(tree.is_some());
    }
}
