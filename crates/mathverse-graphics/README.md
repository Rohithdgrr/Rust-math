# MathVerse Graphics

[![Crates.io](https://img.shields.io/crates/v/mathverse-graphics.svg)](https://crates.io/crates/mathverse-graphics)
[![docs.rs](https://docs.rs/mathverse-graphics/badge.svg)](https://docs.rs/mathverse-graphics)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

2D/3D graphics math: affine transforms, quaternions, camera matrices, and projection for the MathVerse ecosystem.

---

## Features

- **Affine transforms** — 3D translation, rotation (X/Y/Z), scale as 4×4 matrices
- **2D transforms** — 3×3 homogeneous affine matrix, 2D point rotation
- **Quaternions** — unit quaternion rotations, axis-angle, SLERP interpolation
- **Camera** — look-at matrix, perspective projection (right-handed, depth [0,1])

## Module Overview

| Module | Description |
|---|---|
| `lib` | Affine transform matrices, camera, projection, 2D transforms |
| `quat` | `Quat` type: quaternion rotations, composition, SLERP |

## Installation

```toml
[dependencies]
mathverse-graphics = "0.1"
```

## Quick Start

```rust
use mathverse_graphics::*;
use mathverse_graphics::quat::Quat;

fn main() {
    // 3D translation
    let t = translation(1.0, 2.0, 3.0);
    let (x, y, z) = apply(&t, 0.0, 0.0, 0.0);
    println!("Translated: ({x}, {y}, {z})");

    // Quaternion rotation: 90° around Y axis
    let q = Quat::from_axis_angle([0.0, 1.0, 0.0], std::f64::consts::FRAC_PI_2);
    let rotated = q.rotate([1.0, 0.0, 0.0]);
    println!("Rotated: ({:.3}, {:.3}, {:.3})", rotated[0], rotated[1], rotated[2]);

    // Perspective projection
    let proj = perspective(std::f64::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
    println!("Projection matrix created (4×4)");
}
```

Expected output:

```
Translated: (1, 2, 3)
Rotated: (0.000, 0.000, -1.000)
Projection matrix created (4×4)
```

## Per-Module Reference

### Transform Functions

| Function | Description |
|---|---|
| `translation(tx, ty, tz)` | 3D translation as 4×4 matrix |
| `rotation_x(a)` | Rotation about X axis (radians) |
| `rotation_y(a)` | Rotation about Y axis (radians) |
| `rotation_z(a)` | Rotation about Z axis (radians) |
| `scale(s)` | Uniform 3D scale |
| `transform2d(a, tx, ty, s)` | 2D affine transform (3×3 homogeneous) |
| `apply(m, x, y, z)` | Apply affine matrix to 3D point |
| `rotate2d(x, y, a)` | Rotate 2D point around origin |
| `look_at(eye, target, up)` | Camera look-at matrix |
| `perspective(fovy, aspect, near, far)` | Perspective projection matrix |

### `Quat` — Quaternion Rotations

| Method | Description |
|---|---|
| `Quat::new(w, x, y, z)` | Constructor |
| `Quat::identity()` | Identity rotation `(1, 0, 0, 0)` |
| `Quat::from_axis_angle(axis, angle)` | From axis-angle (radians) |
| `.norm()` | Euclidean norm |
| `.normalized()` | Unit quaternion |
| `.conjugate()` | Quaternion conjugate |
| `.mul(other)` | Hamilton product (rotation composition) |
| `.rotate(p)` | Rotate 3D vector |
| `.slerp(other, t)` | Spherical linear interpolation |

## Dependencies

- `mathverse-core`
- `mathverse-matrix`
- `mathverse-vector`

## Future Scope

- Euler angle conversions (yaw/pitch/roll ↔ quaternion)
- Rotation matrix ↔ quaternion conversion
- Bezier curves and surfaces
- Frustum culling helpers
- Homogeneous divide utilities

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE) for details.
