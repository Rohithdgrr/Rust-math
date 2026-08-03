# MathVerse Image

[![Crates.io](https://img.shields.io/crates/v/mathverse-image.svg)](https://crates.io/crates/mathverse-image)
[![docs.rs](https://docs.rs/mathverse-image/badge.svg)](https://docs.rs/mathverse-image)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Grayscale image processing: kernels, blur, edge detection, morphology, I/O, and pixel operations for the MathVerse ecosystem.

---

## Features

- **Grayscale images** — `GrayImage` type with row-major `f64` pixels in [0, 1]
- **Convolutions** — 3×3 kernel convolution, Gaussian blur, box blur, sharpen
- **Edge detection** — Sobel operator, full Canny pipeline (NMS, hysteresis)
- **Thresholding** — Binary threshold, adaptive local-mean threshold
- **Noise** — Gaussian noise, salt-and-pepper noise
- **Morphology** — Erode, dilate, open, close, binarize
- **Arithmetic** — Add, subtract, multiply, scale, offset, invert
- **Statistics** — Mean, std dev, min, max, histogram
- **Transforms** — Flip, rotate, resize (nearest-neighbor)
- **I/O** — Load/save PNG, JPEG, BMP via the `image` crate

## Module Overview

| Module | Description |
|--------|-------------|
| `lib` | `GrayImage` type, `box_blur`, `sharpen`, convolution, Sobel, histogram, transforms |
| `operations` | Thresholding, noise, arithmetic, statistics, contrast |
| `canny` | Full Canny edge detection pipeline |
| `morphology` | Binary morphology: erode, dilate, open, close |
| `io` | File and byte-level I/O |
| `error` | `ImageError` enum and `Result` alias |

## Installation

```toml
[dependencies]
mathverse-image = { path = "crates/mathverse-image" }
```

## Quick Start

```rust
use mathverse_image::*;

fn main() {
    // Create a 4×4 image
    let mut img = GrayImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.set(x, y, (x + y) as f64 / 6.0);
        }
    }

    // Gaussian blur
    let blurred = img.gaussian_blur(1, 1.0);
    println!("Blurred mean: {:.4}", blurred.mean());

    // Sobel edge detection
    let (magnitude, _direction) = img.sobel();
    println!("Max gradient: {:.4}", magnitude.max_value());

    // Threshold
    let binary = img.threshold(0.5);
    println!("Binary mean: {:.4}", binary.mean());

    // Histogram
    let hist = img.histogram();
    println!("Non-zero bins: {}", hist.iter().filter(|&&c| c > 0).count());
}
```

Expected output:

```
Blurred mean: 0.5000
Max gradient: 0.5000
Binary mean: 0.5000
Non-zero bins: 20
```

---

## Per-Module Reference

### `GrayImage` — Core Type

| Method | Description |
|--------|-------------|
| `GrayImage::new(w, h)` | Blank black image |
| `GrayImage::from_data(w, h, data)` | From raw `Vec<f64>`, validates dimensions |
| `.get(x, y)` | Get pixel value |
| `.set(x, y, v)` | Set pixel (clamped to [0, 1]) |
| `.width()` / `.height()` | Dimensions |
| `.data()` | Raw pixel slice |

### `lib` — Convolution & Filters

| Method | Description |
|--------|-------------|
| `.convolve3(kernel)` | 3×3 convolution, border-clamped |
| `.gaussian_blur(r, sigma)` | Separable Gaussian blur |
| `.sobel()` → `(GrayImage, Vec<f64>)` | Sobel gradient magnitude + direction |
| `.flip_h()` | Horizontal flip |
| `.rotate90()` | 90° clockwise rotation |
| `.resize(nw, nh)` | Nearest-neighbor resize |
| `.histogram()` → `[usize; 256]` | 256-bin histogram |

### `operations` — Pixel Operations

| Method | Description |
|--------|-------------|
| `.threshold(t)` | Binary threshold at `t` |
| `.adaptive_threshold(block_size, c)` | Local-mean adaptive threshold |
| `.add_gaussian_noise(mean, std_dev)` | Add Gaussian noise |
| `.add_salt_pepper_noise(density)` | Add salt-and-pepper noise |
| `.add(other)` | Element-wise add (clamped) |
| `.scale(factor)` | Scalar multiply |
| `.offset(value)` | Add constant |
| `.invert()` | 1.0 − pixel |
| `.gamma_correction(gamma)` | Gamma correction |
| `.mean()` | Mean pixel value |
| `.normalize()` | Normalize to [0, 1] |

### `canny` — Canny Edge Detection

| Function | Description |
|----------|-------------|
| `canny(img, sigma, low, high)` | Full pipeline: Gaussian → Sobel → NMS → double threshold → hysteresis |

### `morphology` — Binary Morphology

| Function | Description |
|----------|-------------|
| `binarize(img, t)` | Threshold to 0/1 |
| `erode(img)` | 3×3 cross erosion |
| `dilate(img)` | 3×3 cross dilation |
| `open(img)` | Erode then dilate |
| `close(img)` | Dilate then erode |

### `io` — Image I/O

| Function | Description |
|----------|-------------|
| `load(path)` | Load from PNG/JPEG/BMP |
| `save(img, path)` | Save to file |
| `load_from_bytes(bytes)` | Load from raw bytes |
| `save_to_bytes(img, format)` | Save to in-memory bytes |

---

## Dependencies

- `image 0.25`
- `thiserror 2`
- `rand 0.9`

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE).
