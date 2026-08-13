# MathVerse Vision

[![Crates.io](https://img.shields.io/crates/v/mathverse-vision.svg)](https://crates.io/crates/mathverse-vision)
[![docs.rs](https://docs.rs/mathverse-vision/badge.svg)](https://docs.rs/mathverse-vision)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Computer vision primitives in pure Rust: camera models, homography, epipolar geometry, feature detection, and optical flow — with OpenCV-like live camera support.

---

## 🎥 Live Camera Features (NEW!)

**OpenCV-style live camera interface with real-time processing:**

- **Live Camera Capture** — Cross-platform camera support (Windows/Linux/macOS)
- **Window Display** — Real-time video visualization with `minifb`
- **Interactive Controls** — Keyboard shortcuts for features and modes
- **Real-time CV** — Edge detection, corners, motion, blur, thresholding
- **Drawing Overlays** — Lines, circles, rectangles, bounding boxes

### Quick Camera Example

```rust
use mathverse_vision::camera::SystemCamera;
use minifb::{Key, Window, WindowOptions};

let mut cap = SystemCamera::new(0)?;  // Like cv2.VideoCapture(0)
let mut window = Window::new("Camera", 640, 480, WindowOptions::default())?;

while window.is_open() && !window.is_key_down(Key::Escape) {
    let (ret, frame) = cap.read()?;  // Like cap.read()
    // Process and display frame...
}
```

### Run Examples

```bash
# Simple live camera window
cargo run --example simple_camera_window

# Live camera with real-time CV features
cargo run --example live_camera

# Full OpenCV-style demo with 10+ modes
cargo run --example opencv_features

# Headless batch pipeline (no camera needed): image I/O, filters,
# morphology, thresholding, contours, template matching, transforms...
cargo run --example opencv_pipeline
```

**opencv_features Controls:**
- `1-9, 0` — Switch modes (original, edges, corners, blur, etc.)
- `SPACE` — Freeze/resume
- `F` — Toggle FPS
- `ESC/Q` — Quit

---

## Features

- **Live Camera** — OpenCV-style camera capture with window display
- **Image I/O** — `imread` / `imwrite` for PNG, JPEG, BMP and PNM (like `cv2.imread`)
- **Drawing** — Lines, rectangles, circles, polygons, ellipses, text overlays (`putText`)
- **Color conversions** — RGB↔grayscale, RGB↔BGR, RGB↔HSV, jet colormaps (`cv2.cvtColor`)
- **Filters** — Box, median, bilateral, sharpen, Gaussian blur, arbitrary `filter2D`
- **Arithmetic & bitwise** — `add`, `subtract`, `multiply`, `divide`, `addWeighted`, AND/OR/XOR/NOT
- **Morphology** — Erode, dilate, opening, closing, gradient, top hat, black hat
- **Thresholding** — Binary, inverse, truncate, to-zero and Otsu's method (full `THRESH_*` family)
- **Edge Detection** — Canny, Sobel, Scharr, Laplacian operators
- **Transformations** — Resize, rotate, flip, transpose, crop, `warpAffine`, `warpPerspective`, `getRotationMatrix2D`, pyramids
- **Histogram** — Equalization and min-max normalization (`cv2.equalizeHist` / `normalize`)
- **Template Matching** — All six `cv2.matchTemplate` methods (SQDIFF, CCORR, CCOEFF + normalized)
- **Contours** — `findContours`, area, arc length, bounding rect, convex hull, Douglas–Peucker approximation
- **Connected Components** — 4/8-connectivity labeling
- **Hough Transform** — Line and circle detection
- **Corner Detection** — Harris, Shi–Tomasi, FAST
- **Pinhole camera model** — Projection and unprojection of 3D world points to 2D image coordinates
- **Homography estimation** — Direct Linear Transform (DLT) for planar homography
- **Epipolar geometry** — Fundamental matrix estimation, Sampson distance
- **Lucas-Kanade optical flow** — Dense optical flow between two frames

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `camera` | Live camera capture & pinhole model | `SystemCamera::new`, `Camera::project` |
| `io` | Image file I/O | `imread`, `imwrite`, `imread_color`, `imwrite_color` |
| `drawing` | Drawing primitives & text | `line`, `rectangle`, `circle`, `fill_poly`, `ellipse`, `put_text` |
| `color` | Color conversions | `rgb_to_gray`, `rgb_to_hsv`, `hsv_to_rgb`, `gray_to_jet` |
| `filters` | Spatial filtering | `filter2d`, `box_filter`, `median_blur`, `bilateral_filter`, `sharpen` |
| `arithmetic` | Arithmetic & bitwise ops | `add`, `subtract`, `add_weighted`, `bitwise_and` |
| `morphology` | Morphological operations | `erode`, `dilate`, `opening`, `closing`, `top_hat`, `black_hat` |
| `ops` | Image operations | `canny`, `sobel`, `laplacian`, `histogram_equalize`, `normalize_minmax` |
| `threshold` | Thresholding | `binary`, `adaptive`, `otsu`, `binary_inv`, `tozero` |
| `transform` | Geometric transforms | `resize`, `rotate`, `flip`, `crop`, `warp_affine`, `warp_perspective`, `pyr_down` |
| `template` | Template matching | `match_template` (all `TM_*` methods) |
| `contours` | Contour analysis | `find_contours`, `contour_area`, `convex_hull`, `approx_poly_dp`, `bounding_rect` |
| `connected_components` | Component labeling | `connected_components` |
| `hough` | Hough transforms | `hough_lines`, `hough_circles` |
| `features` | Corner detection | `harris`, `shi_tomasi`, `fast` |
| `homography` | Planar homography estimation | `homography_dlt`, `Homography::apply` |
| `epipolar` | Fundamental matrix, epipolar constraints | `fundamental`, `Fundamental::line_in_second` |
| `flow` | Dense optical flow | `lucas_kanade` |
| `utils` | Utility functions | `mean`, `std_dev`, `min_max` |
| `video` | Video I/O | `VideoWriter` |

## Installation

```toml
[dependencies]
mathverse-vision = { path = "crates/mathverse-vision" }
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

---

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

### Homography

Estimates a 3×3 homography matrix mapping points between two planes using the Direct Linear Transform (DLT) algorithm.

```rust
use mathverse_vision::homography::homography_dlt;

// 4 point correspondences (minimum)
let src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
let dst = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];

let h = homography_dlt(&src, &dst).unwrap();
let (px, py) = h.apply(0.5, 0.5);
// (1.0, 1.0) — scaled 2x
```

**Use cases:** Document scanning (page flattening), panorama stitching, augmented reality planar tracking.

### Epipolar Geometry

Fundamental matrix encodes the epipolar constraint between two views of the same scene.

```
Epipolar Geometry:

  x₂ᵀ F x₁ = 0  (epipolar constraint)

  F maps point in view 1 → epipolar line in view 2
```

```rust
use mathverse_vision::epipolar::fundamental;

let a: Vec<(f64, f64)> = (0..8).map(|i| (i as f64 * 0.5, (i * i) as f64 * 0.1)).collect();
let b: Vec<(f64, f64)> = a.iter().map(|(x, y)| (*x, y + 1.0)).collect();

let f = fundamental(&a, &b).unwrap();

// Epipolar line for point in first image
let (a, b_coeff, c) = f.line_in_second(1.0, 2.0);

// Sampson distance (geometric error)
let dist = f.sampson_distance(1.0, 2.0, 1.0, 3.0);
```

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

let mut img = Image::new(24, 24);
for y in 0..24 {
    for x in 0..24 {
        img.set(x, y, if (x / 6 + y / 6) % 2 == 0 { 1.0 } else { 0.0 });
    }
}

let response = harris(&img, 1.0, 0.04);
```

### Lucas-Kanade Optical Flow

Dense optical flow between two frames using spatial-temporal gradients.

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

---

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE).
