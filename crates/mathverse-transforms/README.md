# MathVerse Transforms

[![Crates.io](https://img.shields.io/crates/v/mathverse-transforms.svg)](https://crates.io/crates/mathverse-transforms)
[![docs.rs](https://docs.rs/mathverse-transforms/badge.svg)](https://docs.rs/mathverse-transforms)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Classical signal transforms in zero-dependency Rust — FFT, DCT, DST, Haar wavelets, Goertzel detection, Hough line/circle detection, and Radon transform for CT reconstruction.

---

## Features

- **Radix-2 FFT/IFFT** — Bit-reversal permutation, O(n log n)
- **DCT I-IV** and **DST I-IV** — Complete discrete cosine/sine transform family
- **Haar DWT** — Single and multi-level forward/inverse wavelet decomposition
- **Goertzel algorithm** — Single-frequency DFT bin detection in O(n)
- **Hough transform** — Line and circle accumulator voting
- **Radon transform** — Sinogram generation for tomographic reconstruction
- **Cross-correlation & convolution** — Via FFT acceleration
- **Power spectrum** — Extraction from real-valued signals

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `fft` | Fast Fourier Transform (radix-2) and inverse | `fft`, `ifft`, `dft`, `idft`, `fft_real`, `power_spectrum`, `cross_correlation`, `convolution` |
| `dct` | Discrete Cosine Transform (types I-IV) and inverse | `dct1`, `dct2`, `dct3`, `dct4`, `idct2` |
| `dst` | Discrete Sine Transform (types I-IV) | `dst1`, `dst2`, `dst3`, `dst4` |
| `wavelet` | Haar wavelet transform, multi-level decomposition | `haar_dwt`, `haar_idwt`, `haar_dwt_multi`, `haar_idwt_multi`, `haar_energy`, `haar_threshold` |
| `goertzel` | Single-frequency DFT bin detector | `goertzel`, `goertzel_magnitude`, `goertzel_batch` |
| `hough` | Hough line and circle detection | `hough_line_accumulator`, `hough_find_lines`, `hough_circle_accumulator` |
| `radon` | Radon transform for sinogram generation | `radon_transform`, `sinogram` |

## Installation

```toml
[dependencies]
mathverse-transforms = { path = "crates/mathverse-transforms" }
mathverse-complex = { path = "crates/mathverse-complex" }
```

## Quick Start

```rust
use mathverse_transforms::{fft, ifft, dct2, haar_dwt, goertzel_magnitude};
use mathverse_complex::Complex;

fn main() {
    // FFT: analyze a signal
    let signal: Vec<Complex> = (0..8)
        .map(|i| Complex::real((2.0 * std::f64::consts::PI * i as f64 / 8.0).sin()))
        .collect();
    let spectrum = fft(&signal).unwrap();
    let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
    println!("FFT magnitudes: {:?}", magnitudes);

    // Roundtrip: FFT → IFFT = original
    let recovered = ifft(&spectrum).unwrap();
    println!("Roundtrip error: {:.2e}",
        signal.iter().zip(&recovered)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0, f64::max));

    // DCT-II: energy compaction
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let coeffs = dct2(&data);
    println!("DCT-II coeffs: {:?}", coeffs);

    // Haar wavelet: multi-level decomposition
    let signal = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let wavelet = haar_dwt(&signal).unwrap();
    println!("Haar DWT: {:?}", wavelet);

    // Goertzel: detect frequency bin 3 in 8-point signal
    let mag = goertzel_magnitude(&signal, 3);
    println!("Goertzel bin 3 magnitude: {:.4}", mag);
}
```

---

## Module Documentation

### FFT Module (`fft`)

The Fast Fourier Transform computes the Discrete Fourier Transform in O(n log n) using the Cooley-Tukey radix-2 algorithm with bit-reversal permutation.

**Formula:**

```
X[k] = Σ_{n=0}^{N-1} x[n] · e^{-2πi·kn/N}

Inverse:
x[n] = (1/N) Σ_{k=0}^{N-1} X[k] · e^{2πi·kn/N}
```

**Key functions:**

| Function | Signature | Description |
|---|---|---|
| `fft` | `(&[Complex]) -> MathResult<Vec<Complex>>` | Radix-2 FFT. Input length must be power of 2. |
| `ifft` | `(&[Complex]) -> MathResult<Vec<Complex>>` | Inverse FFT via conjugation trick. |
| `dft` | `(&[Complex]) -> Vec<Complex>` | Naive O(n²) DFT. Useful for verification. |
| `fft_real` | `(&[f64]) -> Vec<Complex>` | FFT of real-valued signal (wraps to Complex). |
| `power_spectrum` | `(&[f64]) -> Vec<f64>` | \|X[k]\|² for real input. |
| `cross_correlation` | `(&[f64], &[f64]) -> Vec<f64>` | Cross-correlation via FFT. |
| `convolution` | `(&[f64], &[f64]) -> Vec<f64>` | Linear convolution via FFT. |

---

### DCT Module (`dct`)

Discrete Cosine Transforms concentrate signal energy in low-frequency coefficients — the backbone of JPEG, MP3, and HEVC compression.

**Formula (DCT-II):**

```
X[k] = c_k · Σ_{i=0}^{N-1} x[i] · cos(π(i+½)k/N)
    c_0 = √(1/N),  c_k = √(2/N) for k > 0
```

---

### Wavelet Module (`wavelet`)

Haar wavelet decomposition splits a signal into approximation (averages) and detail (differences) coefficients, enabling multi-resolution analysis.

**Formulas:**

```
Forward (single level):
    a[i] = (x[2i] + x[2i+1]) / √2     ← approximation
    d[i] = (x[2i] - x[2i+1]) / √2     ← detail

Inverse:
    x[2i]   = (a[i] + d[i]) / √2
    x[2i+1] = (a[i] - d[i]) / √2
```

---

### Goertzel Module (`goertzel`)

The Goertzel algorithm detects a single frequency bin in O(n) — far more efficient than a full FFT when you only need a few frequency components.

**Use cases:** DTMF detection, radar/sonar tone detection, power line harmonics, musical note detection.

---

### Hough Module (`hough`)

The Hough transform detects parametric shapes (lines, circles) by voting in parameter space.

**Line formula:** `ρ = x·cos(θ) + y·sin(θ)`

**Circle formula:** `(x - cx)² + (y - cy)² = r²`

---

### Radon Module (`radon`)

The Radon transform computes line integrals through an image at various angles — the mathematical foundation of CT reconstruction.

**Formula:**

```
R(θ, t) = ∫∫ f(x,y) · δ(x·cosθ + y·sinθ - t) dx dy
```

---

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE).
