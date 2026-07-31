# mathverse-vision

Computer vision primitives in pure Rust: camera models, homography, epipolar geometry, feature detection, and optical flow — zero dependencies.

## Features

- Pinhole camera model with projection and unprojection
- Homography estimation via Direct Linear Transform (DLT)
- Fundamental matrix estimation for epipolar geometry
- Harris corner detection
- Lucas-Kanade optical flow
- Gaussian blur and 3×3 convolution on grayscale images

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `camera` | Pinhole camera model | `Camera::project`, `Camera::unproject` |
| `homography` | Planar homography estimation | `homography_dlt`, `Homography::apply` |
| `epipolar` | Fundamental matrix, epipolar constraints | `fundamental`, `Fundamental::line_in_second`, `sampson_distance` |
| `features` | Corner detection | `harris` |
| `flow` | Dense optical flow | `lucas_kanade` |
| `lib` | Image type with convolution | `Image::convolve3`, `Image::gaussian_blur` |

## Installation

```bash
cargo add mathverse-vision
```

Or add to `Cargo.toml`:

```toml
[dependencies]
mathverse-vision = { path = "../mathverse-vision" }
```

## Quick Start

```rust
use mathverse_vision::camera::Camera;

fn main() {
    let cam = Camera::new(800.0, 600.0, 320.0, 240.0);
    let (u, v) = cam.project(1.0, 2.0, 10.0);
    println!("Pixel: ({:.1}, {:.1})", u, v);
    // Pixel: (400.0, 360.0)
}
```

## Module Documentation

### Camera Model

Pinhole camera projecting 3D world points to 2D image coordinates.

```
Camera Projection:

    World (X,Y,Z)
         │
         ▼
    ┌─────────┐
    │  fx  0  cx │   K = intrinsic matrix
    │  0  fy  cy │
    │  0   0   1 │
    └─────────┘
         │
         ▼
    Image (u,v)

    u = fx · X/Z + cx
    v = fy · Y/Z + cy
```

```rust
use mathverse_vision::camera::Camera;

let cam = Camera::new(800.0, 600.0, 320.0, 240.0);

// Project 3D point to 2D
let (u, v) = cam.project(1.0, 2.0, 10.0);
// u = 800 * 1/10 + 320 = 400.0
// v = 600 * 2/10 + 240 = 360.0

// Unproject back (requires known depth)
let (x, y) = cam.unproject(u, v, 10.0);
// x = (400 - 320) * 10 / 800 = 1.0
// y = (360 - 240) * 10 / 600 = 2.0
```

**Formulas**:
- Projection: `u = fx · X/Z + cx`, `v = fy · Y/Z + cy`
- Unprojection: `X = (u - cx) · Z / fx`, `Y = (v - cy) · Z / fy`

### Homography

Estimates a 3×3 homography matrix mapping points between two planes using the Direct Linear Transform (DLT) algorithm.

```
Homography Transform:

  Source Plane          Destination Plane
  ┌──────────┐          ┌──────────┐
  │ (x,y)    │──── H ───│ (x',y')  │
  │          │          │          │
  └──────────┘          └──────────┘

  [x']   [h0 h1 h2] [x]
  [y'] = [h3 h4 h5] [y]
  [ w ]   [h6 h7 h8] [1]

  x' = (h0·x + h1·y + h2) / w
  y' = (h3·x + h4·y + h5) / w
```

```rust
use mathverse_vision::homography::homography_dlt;

// 4 point correspondences (minimum)
let src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
let dst = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];

let h = homography_dlt(&src, &dst).unwrap();
let (px, py) = h.apply(0.5, 0.5);
// (1.0, 1.0) — scaled 2x
```

**Use cases**: Document scanning (page flattening), panorama stitching, augmented reality planar tracking.

### Epipolar Geometry

Fundamental matrix encodes the epipolar constraint between two views of the same scene.

```
Epipolar Geometry:

  Camera 1              Camera 2
  ┌──────┐              ┌──────┐
  │   P₁ │              │  P₂  │
  └──┬───┘              └──┬───┘
     │                     │
     │    ┌──────────┐    │
     └────┤ Scene Pt ├───┘
          │    X     │
          └──────────┘

  x₂ᵀ F x₁ = 0  (epipolar constraint)

  F maps point in view 1 → epipolar line in view 2
```

```rust
use mathverse_vision::epipolar::fundamental;

// 8+ point correspondences
let a: Vec<(f64, f64)> = (0..8).map(|i| (i as f64 * 0.5, (i * i) as f64 * 0.1)).collect();
let b: Vec<(f64, f64)> = a.iter().map(|(x, y)| (*x, y + 1.0)).collect();

let f = fundamental(&a, &b).unwrap();

// Epipolar line for point in first image
let (a, b_coeff, c) = f.line_in_second(1.0, 2.0);
// Line: a·x + b·y + c = 0 in second image

// Sampson distance (geometric error)
let dist = f.sampson_distance(1.0, 2.0, 1.0, 3.0);
// Small distance = good correspondence
```

**Formula**: `x₂ᵀ F x₁ = 0` where F is the 3×3 fundamental matrix (rank 2).

### Harris Corner Detection

Detects corner features using the structure tensor.

```
Harris Response:

  R = det(M) - k · trace(M)²

  M = [ΣIx²   ΣIxIy]     (structure tensor)
      [ΣIxIy  ΣIy² ]

  R > threshold → corner
  R < 0         → edge
  |R| small     → flat region
```

```rust
use mathverse_vision::{Image, features::harris};

// Create a simple image with corners
let mut img = Image::new(24, 24);
for y in 0..24 {
    for x in 0..24 {
        img.set(x, y, if (x / 6 + y / 6) % 2 == 0 { 1.0 } else { 0.0 });
    }
}

let response = harris(&img, 1.0, 0.04);
// High response at corners, low at flat/edge regions
```

### Lucas-Kanade Optical Flow

Dense optical flow between two frames using spatial-temporal gradients.

```
Lucas-Kanade:

  For each pixel (x,y):
    ┌            ┐ ┌  u ┐   ┌ -It·Ix ┐
    │ ΣIx²  ΣIxIy│ │    │ = │        │
    │ ΣIxIy ΣIy² │ │  v ┘   │ -It·Iy  │
    └            ┘          └────────┘

  Solve 2×2 system per pixel window
  u = horizontal flow
  v = vertical flow
```

```rust
use mathverse_vision::{Image, flow::lucas_kanade};

let (w, h) = (32, 32);
let mut a = Image::new(w, h);
let mut b = Image::new(w, h);

// Blob at (8..24, 8..24) in frame a
for y in 8..24 { for x in 8..24 { a.set(x, y, 1.0); } }
// Same blob shifted right+down by 1 pixel
for y in 8..24 { for x in 8..24 { b.set(x + 1, y + 1, 1.0); } }

let (u, v) = lucas_kanade(&a, &b);
// u ≈ 1.0 (horizontal flow)
// v ≈ 1.0 (vertical flow)
```

### Image Processing

The `Image` type supports 3×3 convolution and Gaussian blur.

```rust
use mathverse_vision::Image;

let mut img = Image::new(64, 64);
// ... fill image data ...

// Gaussian blur
let blurred = img.gaussian_blur(2, 1.0); // radius=2, sigma=1.0

// Custom 3×3 convolution
let sharpen: [f64; 9] = [
    0.0, -1.0, 0.0,
   -1.0,  5.0, -1.0,
    0.0, -1.0, 0.0,
];
let sharp = img.convolve3(&sharpen);
```

## Future Scope

- [ ] SIFT / SURF / ORB feature descriptors
- [ ] RANSAC for robust estimation
- [ ] Essential matrix decomposition (recovery R, t)
- [ ] Stereo correspondence / disparity
- [ ] Color image support (RGB)
- [ ] Image pyramids for multi-scale processing
- [ ] Bundle adjustment

## License

MIT OR Apache-2.0
