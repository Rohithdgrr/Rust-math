# MathVerse Geometry

[![Crates.io](https://img.shields.io/crates/v/mathverse-geometry.svg)](https://crates.io/crates/mathverse-geometry)
[![docs.rs](https://docs.rs/mathverse-geometry/badge.svg)](https://docs.rs/mathverse-geometry)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Full-featured geometry library — 2D/3D shapes, spatial structures, transforms, intersections, distances, and ray tracing.

---

## Features

- **2D shapes** — Point2, Circle, Triangle, Rectangle, Polygon, Ellipse with area/perimeter/centroid/contains
- **3D shapes** — Point3, Sphere, Cube, Cylinder, Cone, Plane, Line3 with volume/surface area
- **2D primitives** — LineSegment2, Ray2, Arc, Sector, CircularSegment, Polyline, BézierCurve
- **Spatial structures** — AABB (2D/3D), OBB, Quadtree, Octree, Frustum
- **Triangle mesh** — indexed TriangleMesh, AABB tree, ray tracing (Möller-Trumbore)
- **Intersections** — point-in-polygon (ray-cast + winding), segment-polygon, circle-polygon, polygon-polygon (SAT), convex hull
- **Distances** — point-segment, point-line, point-polygon, closest pair, GJK distance
- **Transforms** — `Transform2D` trait: translate, scale, rotate for all 2D shapes
- **Metrics** — angle between vectors, signed area, winding number, moment of inertia, Monte Carlo area

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `shapes2d` | 2D primitives with measures: `Point2`, `Circle`, `Triangle`, `Rectangle`, `Polygon`, `Ellipse` |
| `shapes3d` | 3D primitives with measures: `Point3`, `Sphere`, `Cube`, `Cylinder`, `Cone`, `Plane`, `Line3` |
| `primitives2d` | Extended 2D geometry: `LineSegment2`, `Ray2`, `Arc`, `Sector`, `CircularSegment`, `Polyline`, `BezierCurve` |
| `spatial` | Spatial indexing & bounding: `AABB`, `AABB3`, `OBB`, `Quadtree`, `Octree`, `Frustum` |
| `mesh3d` | Triangle mesh & ray tracing: `Triangle3`, `TriangleMesh`, `AABBTreeNode`, `ray_triangle_intersect`, `build_aabb_tree` |
| `intersection` | Collision detection: `point_in_polygon`, `winding_number`, `segments_intersect`, `polygons_intersect`, `convex_hull`, `is_convex` |
| `distance` | Distance & proximity: `point_segment_distance`, `point_line_distance`, `point_polygon_distance`, `closest_pair`, `gjk_distance` |
| `transforms` | 2D shape transforms: `Transform2D` trait — `translate`, `scale_xy`, `rotate`, `rotate_around` |
| `metrics` | Geometric metrics: `angle_between`, `signed_angle`, `signed_area`, `is_counterclockwise`, `moment_of_inertia_*`, `monte_carlo_area` |

---

## Installation

```toml
[dependencies]
mathverse-geometry = "0.1"
```

---

## Quick Start

```rust
use mathverse_geometry::{
    Circle, Triangle, Point2, Polygon,
    point_in_polygon, convex_hull,
    ray_triangle_intersect, Triangle3, Point3,
};

fn main() {
    // Circle area
    let c = Circle::new(Point2::new(0.0, 0.0), 5.0);
    println!("Circle area: {:.2}", c.area());
    // Circle area: 78.54

    // Triangle from vertices
    let t = Triangle::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0), Point2::new(0.0, 3.0));
    println!("Triangle area: {}, perimeter: {}", t.area(), t.perimeter());
    // Triangle area: 6, perimeter: 12

    // Point-in-polygon
    let sq = Polygon::new(vec![
        Point2::new(0.0, 0.0), Point2::new(2.0, 0.0),
        Point2::new(2.0, 2.0), Point2::new(0.0, 2.0),
    ]);
    println!("Inside square: {}", point_in_polygon(Point2::new(1.0, 1.0), &sq));
    // Inside square: true

    // Convex hull
    let pts = vec![Point2::new(0.0,0.0), Point2::new(1.0,1.0), Point2::new(2.0,0.0), Point2::new(1.0,-1.0)];
    let hull = convex_hull(&pts);
    println!("Convex hull: {} vertices", hull.len());
    // Convex hull: 4 vertices
}
```

---

## Module Documentation

### 2D Shapes (`shapes2d`)

| Shape | Area | Perimeter |
|---|---|---|
| Circle | `πr²` | `2πr` |
| Ellipse | `π·rx·ry` | Ramanujan approx: `π(3s - √((3rx+ry)(rx+3ry)))` |
| Triangle (shoelace) | `½|x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|` | sum of edge lengths |
| Rectangle | `w × h` | `2(w + h)` |
| Polygon | Shoelace formula | sum of edge lengths |

---

### 3D Shapes (`shapes3d`)

| Shape | Volume | Surface Area |
|---|---|---|
| Sphere | `(4/3)πr³` | `4πr²` |
| Cube | `s³` | `6s²` |
| Cylinder | `πr²h` | `2πr(r + h)` |
| Cone | `(1/3)πr²h` | `πr(r + √(r² + h²))` |

---

### 2D Primitives (`primitives2d`)

```rust
use mathverse_geometry::{LineSegment2, Point2, Arc, Sector, BezierCurve};
use mathverse_core::constants::PI;

let seg = LineSegment2::new(Point2::new(0.0, 0.0), Point2::new(3.0, 4.0));
assert!((seg.length() - 5.0).abs() < 1e-12);
assert_eq!(seg.midpoint(), Point2::new(1.5, 2.0));

let arc = Arc::new(Point2::new(0.0, 0.0), 2.0, 0.0, PI / 2.0);
assert!((arc.length() - PI).abs() < 1e-12);
```

---

### Spatial Structures (`spatial`)

```rust
use mathverse_geometry::{AABB, OBB, Quadtree, Point2, Frustum, Point3, AABB3};

// AABB
let bb = AABB::new(Point2::new(1.0, 2.0), Point2::new(5.0, 6.0));
assert_eq!(bb.width(), 4.0);
assert!(bb.contains(Point2::new(3.0, 4.0)));

// Quadtree spatial query
let bounds = AABB::new(Point2::new(0.0, 0.0), Point2::new(100.0, 100.0));
let mut qt = Quadtree::new(bounds, 4);
for i in 0..50 {
    qt.insert(Point2::new(i as f64, i as f64));
}
let range = AABB::new(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));
let mut result = Vec::new();
qt.query(range, &mut result);
assert_eq!(result.len(), 11); // 0..=10
```

---

### Triangle Mesh & Ray Tracing (`mesh3d`)

```rust
use mathverse_geometry::{
    Triangle3, Point3, TriangleMesh, build_aabb_tree, ray_triangle_intersect, ray_sphere_intersect
};

// Ray-triangle intersection (Möller-Trumbore)
let tri = Triangle3::new(
    Point3::new(-1.0, -1.0, 0.0),
    Point3::new(1.0, -1.0, 0.0),
    Point3::new(0.0, 1.0, 0.0),
);
let t = ray_triangle_intersect(
    Point3::new(0.0, 0.0, -5.0),
    Point3::new(0.0, 0.0, 1.0),
    tri,
);
assert!((t.unwrap() - 5.0).abs() < 1e-6);

// Build AABB tree from mesh
let mesh = TriangleMesh::new(
    vec![Point3::new(0.0,0.0,0.0), Point3::new(1.0,0.0,0.0),
         Point3::new(0.0,1.0,0.0), Point3::new(0.0,0.0,1.0)],
    vec![(0,1,2), (0,1,3)],
);
let tree = build_aabb_tree(&mesh).unwrap();
```

---

### Intersections (`intersection`)

| Function | Description |
|---|---|
| `point_in_polygon(p, poly)` | Ray-casting algorithm, odd crossings = inside |
| `winding_number(p, poly)` | Nonzero = inside (handles holes) |
| `segments_intersect(a, b)` | Segment-segment intersection test |
| `polygons_intersect(a, b)` | SAT (Separating Axis Theorem) for convex polygons |
| `convex_hull(points)` | Graham scan, O(n log n) |
| `is_convex(poly)` | Winding-number consistency check |

---

### Distances (`distance`)

| Function | Formula |
|---|---|
| `point_segment_distance` | `‖p - closest_on_segment‖` |
| `point_line_distance` | `‖(b-a) × (a-p)‖ / ‖b-a‖` |
| `point_polygon_distance` | 0 if inside, else min distance to edges |
| `closest_pair` | O(n²) brute-force closest pair |
| `gjk_distance` | GJK algorithm for convex shape distance |

---

### Transforms (`transforms`)

The `Transform2D` trait is implemented for: `Point2`, `LineSegment2`, `Circle`, `Rectangle`, `Triangle`, `Polygon`, `Ellipse`, `Arc`, `Sector`.

```rust
use mathverse_geometry::{Triangle, Point2, Transform2D};

let t = Triangle::new(Point2::new(0.0, 0.0), Point2::new(1.0, 0.0), Point2::new(0.0, 1.0));
let moved = t.translate(2.0, 3.0);
assert_eq!(moved.a, Point2::new(2.0, 3.0));

let rotated = t.rotate(std::f64::consts::FRAC_PI_2);
// Rotated 90° counterclockwise about origin
```

---

### Metrics (`metrics`)

| Function | Description |
|---|---|
| `angle_between(a, b)` | Angle between two vectors `[0, π]` |
| `signed_angle(a, b)` | Signed angle `(-π, π]` |
| `signed_area(poly)` | Positive = counterclockwise |
| `is_counterclockwise(poly)` | Winding direction test |
| `moment_of_inertia_origin(poly)` | `I₀` for uniform polygon |
| `moment_of_inertia_circle(r)` | `I = πr⁴/2` |
| `moment_of_inertia_rectangle(w,h)` | `I = wh(w²+h²)/12` |
| `monte_carlo_area(poly, bounds, n)` | Approximate area via sampling |

---

## Roadmap

- [ ] Rounding / offset polygons (Minkowski sum)
- [ ] 3D mesh simplification (vertex decimation)
- [ ] BSP tree construction
- [ ] Polygon triangulation (ear clipping)
- [ ] Curve intersection (Bézier-Bézier)
- [ ] `no_std` + `alloc` support
- [ ] Serde serialization for all shapes

---

## License

MIT — see [LICENSE](LICENSE).
