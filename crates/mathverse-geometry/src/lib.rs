#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

//! Geometry: 2D and 3D shapes, measures, transforms, distances, intersections.
//!
//! All values are `f64`. Constructors panic on invalid shapes
//! (e.g. negative radius); this is a documented programmer-error contract,
//! mirroring `nalgebra` style.
//!
//! # Errors
//! Use [`error::GeometryError`] for recoverable failures.

pub mod error;
pub mod shapes2d;
pub mod shapes3d;
pub mod primitives2d;
pub mod spatial;
pub mod intersection;
pub mod distance;
pub mod transforms;
pub mod metrics;
pub mod mesh3d;

// Re-export core types
pub use shapes2d::{Circle, Ellipse, Point2, Polygon, Rectangle, Triangle};
pub use shapes3d::{Cone, Cube, Cylinder, Line3, Plane, Point3, Sphere};

// Re-export 2D primitives
pub use primitives2d::{Arc, BezierCurve, CircularSegment, LineSegment2, Polyline, Ray2, Sector};

// Re-export spatial structures
pub use spatial::{AABB, AABB3, Frustum, OBB, Octree, Quadtree};

// Re-export triangle mesh and tree
pub use mesh3d::{AABBTreeNode, Triangle3, TriangleMesh};

// Re-export Transform2D trait
pub use transforms::Transform2D;
