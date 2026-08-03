//! Spatial structures: axis-aligned bounding box, oriented bounding box, quadtree, octree, frustum.

use super::shapes2d::Point2;
use super::shapes3d::Point3;

// ---------------------------------------------------------------------------
// AABB (2D)
// ---------------------------------------------------------------------------

/// 2D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Point2,
    pub max: Point2,
}

impl AABB {
    pub fn new(min: Point2, max: Point2) -> Self {
        Self { min, max }
    }
    pub fn from_points(pts: &[Point2]) -> Option<Self> {
        if pts.is_empty() {
            return None;
        }
        let mut minx = f64::INFINITY;
        let mut miny = f64::INFINITY;
        let mut maxx = f64::NEG_INFINITY;
        let mut maxy = f64::NEG_INFINITY;
        for p in pts {
            if p.x < minx { minx = p.x; }
            if p.y < miny { miny = p.y; }
            if p.x > maxx { maxx = p.x; }
            if p.y > maxy { maxy = p.y; }
        }
        Some(Self { min: Point2::new(minx, miny), max: Point2::new(maxx, maxy) })
    }
    pub fn width(self) -> f64 {
        self.max.x - self.min.x
    }
    pub fn height(self) -> f64 {
        self.max.y - self.min.y
    }
    pub fn center(self) -> Point2 {
        Point2::new((self.min.x + self.max.x) / 2.0, (self.min.y + self.max.y) / 2.0)
    }
    pub fn area(self) -> f64 {
        self.width() * self.height()
    }
    pub fn contains(self, p: Point2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
    pub fn intersects(self, other: AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }
    pub fn merge(self, other: AABB) -> AABB {
        AABB {
            min: Point2::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y)),
            max: Point2::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y)),
        }
    }
    pub fn expand(self, padding: f64) -> AABB {
        AABB {
            min: Point2::new(self.min.x - padding, self.min.y - padding),
            max: Point2::new(self.max.x + padding, self.max.y + padding),
        }
    }
}

// ---------------------------------------------------------------------------
// AABB3 (3D)
// ---------------------------------------------------------------------------

/// 3D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB3 {
    pub min: Point3,
    pub max: Point3,
}

impl AABB3 {
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }
    pub fn from_points(pts: &[Point3]) -> Option<Self> {
        if pts.is_empty() {
            return None;
        }
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in pts {
            if p.x < min.x { min.x = p.x; }
            if p.y < min.y { min.y = p.y; }
            if p.z < min.z { min.z = p.z; }
            if p.x > max.x { max.x = p.x; }
            if p.y > max.y { max.y = p.y; }
            if p.z > max.z { max.z = p.z; }
        }
        Some(Self { min, max })
    }
    pub fn center(self) -> Point3 {
        Point3::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }
    pub fn half_extents(self) -> Point3 {
        Point3::new(
            (self.max.x - self.min.x) / 2.0,
            (self.max.y - self.min.y) / 2.0,
            (self.max.z - self.min.z) / 2.0,
        )
    }
    pub fn volume(self) -> f64 {
        (self.max.x - self.min.x) * (self.max.y - self.min.y) * (self.max.z - self.min.z)
    }
    pub fn surface_area(self) -> f64 {
        let dx = self.max.x - self.min.x;
        let dy = self.max.y - self.min.y;
        let dz = self.max.z - self.min.z;
        2.0 * (dx * dy + dy * dz + dz * dx)
    }
    pub fn contains(self, p: Point3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }
    pub fn intersects(self, other: AABB3) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }
    pub fn merge(self, other: AABB3) -> AABB3 {
        AABB3 {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// OBB (2D)
// ---------------------------------------------------------------------------

/// 2D oriented bounding box: center, half-extents along local axes, rotation angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OBB {
    pub center: Point2,
    pub half_w: f64,
    pub half_h: f64,
    pub angle: f64,
}

impl OBB {
    pub fn new(center: Point2, half_w: f64, half_h: f64, angle: f64) -> Self {
        Self { center, half_w, half_h, angle }
    }
    /// Four corner vertices.
    pub fn corners(self) -> [Point2; 4] {
        let (s, c) = self.angle.sin_cos();
        let dx = Point2::new(c * self.half_w, s * self.half_w);
        let dy = Point2::new(-s * self.half_h, c * self.half_h);
        [
            self.center.translate(dx.x - dy.x, dx.y - dy.y),
            self.center.translate(dx.x + dy.x, dx.y + dy.y),
            self.center.translate(-dx.x + dy.x, -dx.y + dy.y),
            self.center.translate(-dx.x - dy.x, -dx.y - dy.y),
        ]
    }
    pub fn area(self) -> f64 {
        4.0 * self.half_w * self.half_h
    }
    /// Whether point `p` is inside this OBB (SAT with one axis).
    pub fn contains(self, p: Point2) -> bool {
        let (s, c) = self.angle.sin_cos();
        let dx = p.x - self.center.x;
        let dy = p.y - self.center.y;
        let local_x = c * dx + s * dy;
        let local_y = -s * dx + c * dy;
        local_x.abs() <= self.half_w && local_y.abs() <= self.half_h
    }
}

// ---------------------------------------------------------------------------
// Quadtree
// ---------------------------------------------------------------------------

/// Simple quadtree for 2D point queries.
pub struct Quadtree {
    bounds: AABB,
    points: Vec<Point2>,
    capacity: usize,
    divided: bool,
    ne: Option<Box<Quadtree>>,
    nw: Option<Box<Quadtree>>,
    se: Option<Box<Quadtree>>,
    sw: Option<Box<Quadtree>>,
}

impl Quadtree {
    pub fn new(bounds: AABB, capacity: usize) -> Self {
        Self {
            bounds,
            points: Vec::new(),
            capacity,
            divided: false,
            ne: None,
            nw: None,
            se: None,
            sw: None,
        }
    }
    pub fn insert(&mut self, p: Point2) -> bool {
        if !self.bounds.contains(p) {
            return false;
        }
        if self.points.len() < self.capacity && !self.divided {
            self.points.push(p);
            return true;
        }
        if !self.divided {
            self.subdivide();
        }
        // Safety/invariant: subdivide() always populates all four quadrants
        self.ne.as_mut().unwrap().insert(p)
            || self.nw.as_mut().unwrap().insert(p)
            || self.se.as_mut().unwrap().insert(p)
            || self.sw.as_mut().unwrap().insert(p)
    }
    /// Query all points within `range`.
    pub fn query(&self, range: AABB, result: &mut Vec<Point2>) {
        if !self.bounds.intersects(range) {
            return;
        }
        for p in &self.points {
            if range.contains(*p) {
                result.push(*p);
            }
        }
        if self.divided {
            // Safety/invariant: subdivide() always populates all four quadrants
            self.ne.as_ref().unwrap().query(range, result);
            self.nw.as_ref().unwrap().query(range, result);
            self.se.as_ref().unwrap().query(range, result);
            self.sw.as_ref().unwrap().query(range, result);
        }
    }
    fn subdivide(&mut self) {
        let c = self.bounds.center();
        let hx = self.bounds.width() / 2.0;
        let hy = self.bounds.height() / 2.0;
        let cap = self.capacity;
        self.ne = Some(Box::new(Quadtree::new(AABB::new(Point2::new(c.x, c.y), Point2::new(c.x + hx, c.y + hy)), cap)));
        self.nw = Some(Box::new(Quadtree::new(AABB::new(Point2::new(c.x - hx, c.y), Point2::new(c.x, c.y + hy)), cap)));
        self.se = Some(Box::new(Quadtree::new(AABB::new(Point2::new(c.x, c.y - hy), Point2::new(c.x + hx, c.y)), cap)));
        self.sw = Some(Box::new(Quadtree::new(AABB::new(Point2::new(c.x - hx, c.y - hy), Point2::new(c.x, c.y)), cap)));
        self.divided = true;
    }
}

// ---------------------------------------------------------------------------
// Octree
// ---------------------------------------------------------------------------

/// Simple octree for 3D point queries.
pub struct Octree {
    bounds: AABB3,
    points: Vec<Point3>,
    capacity: usize,
    divided: bool,
    children: Option<[Box<Octree>; 8]>,
}

impl Octree {
    pub fn new(bounds: AABB3, capacity: usize) -> Self {
        Self { bounds, points: Vec::new(), capacity, divided: false, children: None }
    }
    pub fn insert(&mut self, p: Point3) -> bool {
        if !self.bounds.contains(p) {
            return false;
        }
        if self.points.len() < self.capacity && !self.divided {
            self.points.push(p);
            return true;
        }
        if !self.divided {
            self.subdivide();
        }
        // Safety/invariant: subdivide() always populates all eight children
        for child in self.children.as_mut().unwrap().iter_mut() {
            if child.insert(p) {
                return true;
            }
        }
        false
    }
    pub fn query(&self, range: AABB3, result: &mut Vec<Point3>) {
        if !self.bounds.intersects(range) {
            return;
        }
        for p in &self.points {
            if range.contains(*p) {
                result.push(*p);
            }
        }
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.query(range, result);
            }
        }
    }
    fn subdivide(&mut self) {
        let c = self.bounds.center();
        let he = self.bounds.half_extents();
        let cap = self.capacity;
        let mut children: Vec<Box<Octree>> = Vec::with_capacity(8);
        for iz in 0..2 {
            for iy in 0..2 {
                for ix in 0..2 {
                    let min = Point3::new(
                        c.x + (ix as f64 - 0.5) * he.x,
                        c.y + (iy as f64 - 0.5) * he.y,
                        c.z + (iz as f64 - 0.5) * he.z,
                    );
                    let max = Point3::new(
                        min.x + he.x,
                        min.y + he.y,
                        min.z + he.z,
                    );
                    children.push(Box::new(Octree::new(AABB3::new(min, max), cap)));
                }
            }
        }
        let arr: [Box<Octree>; 8] = [
            children.remove(0), children.remove(0), children.remove(0), children.remove(0),
            children.remove(0), children.remove(0), children.remove(0), children.remove(0),
        ];
        self.children = Some(arr);
        self.divided = true;
    }
}

// ---------------------------------------------------------------------------
// Frustum (3D)
// ---------------------------------------------------------------------------

/// View frustum defined by 6 planes (near, far, left, right, top, bottom).
/// Each plane is (nx, ny, nz, d) where nx*x + ny*y + nz*z + d = 0.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub planes: [(f64, f64, f64, f64); 6],
}

impl Frustum {
    pub fn new(planes: [(f64, f64, f64, f64); 6]) -> Self {
        Self { planes }
    }
    /// Whether a point is inside (or on) all frustum planes.
    pub fn contains_point(self, p: Point3) -> bool {
        self.planes.iter().all(|(nx, ny, nz, d)| {
            nx * p.x + ny * p.y + nz * p.z + d >= -1e-10
        })
    }
    /// Whether an AABB intersects the frustum.
    pub fn intersects_aabb(self, bb: AABB3) -> bool {
        for (nx, ny, nz, d) in &self.planes {
            let px = if *nx >= 0.0 { bb.max.x } else { bb.min.x };
            let py = if *ny >= 0.0 { bb.max.y } else { bb.min.y };
            let pz = if *nz >= 0.0 { bb.max.z } else { bb.min.z };
            if nx * px + ny * py + nz * pz + d < -1e-10 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt2(x: f64, y: f64) -> Point2 { Point2::new(x, y) }

    #[test]
    fn aabb_basics() {
        let bb = AABB::new(pt2(1.0, 2.0), pt2(3.0, 4.0));
        assert!((bb.width() - 2.0).abs() < 1e-12);
        assert!((bb.height() - 2.0).abs() < 1e-12);
        assert_eq!(bb.center(), pt2(2.0, 3.0));
        assert!(bb.contains(pt2(2.0, 3.0)));
        assert!(!bb.contains(pt2(0.0, 0.0)));
        let other = AABB::new(pt2(2.5, 3.5), pt2(5.0, 5.0));
        assert!(bb.intersects(other));
        let disjoint = AABB::new(pt2(10.0, 10.0), pt2(20.0, 20.0));
        assert!(!bb.intersects(disjoint));
    }

    #[test]
    fn aabb_from_points() {
        let pts = vec![pt2(1.0, 5.0), pt2(3.0, 2.0), pt2(-1.0, 4.0)];
        let bb = AABB::from_points(&pts).unwrap();
        assert_eq!(bb.min, pt2(-1.0, 2.0));
        assert_eq!(bb.max, pt2(3.0, 5.0));
    }

    #[test]
    fn obb_contains() {
        let obb = OBB::new(pt2(0.0, 0.0), 2.0, 1.0, 0.0);
        assert!(obb.contains(pt2(0.0, 0.0)));
        assert!(obb.contains(pt2(1.9, 0.9)));
        assert!(!obb.contains(pt2(2.1, 0.0)));
    }

    #[test]
    fn quadtree_query() {
        let bounds = AABB::new(pt2(0.0, 0.0), pt2(10.0, 10.0));
        let mut qt = Quadtree::new(bounds, 4);
        for i in 0..20 {
            qt.insert(pt2(i as f64 + 0.5, i as f64 + 0.5));
        }
        let range = AABB::new(pt2(0.0, 0.0), pt2(5.0, 5.0));
        let mut result = Vec::new();
        qt.query(range, &mut result);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn frustum_contains() {
        // Simple frustum: all half-spaces with normals pointing inward
        let planes = [
            (0.0, 0.0, 1.0, 1.0),   // near: z >= -1
            (0.0, 0.0, -1.0, 10.0),  // far: z <= 10
            (1.0, 0.0, 0.0, 1.0),    // left: x >= -1
            (-1.0, 0.0, 0.0, 10.0),  // right: x <= 10
            (0.0, 1.0, 0.0, 1.0),    // bottom: y >= -1
            (0.0, -1.0, 0.0, 10.0),  // top: y <= 10
        ];
        let f = Frustum::new(planes);
        assert!(f.contains_point(Point3::new(5.0, 5.0, 5.0)));
        assert!(!f.contains_point(Point3::new(15.0, 5.0, 5.0)));
    }
}
