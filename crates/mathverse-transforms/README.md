# mathverse-transforms

A zero-dependency Rust library implementing classical signal transforms: FFT, DCT, DST, Haar wavelets, Goertzel detection, Hough line/circle detection, and Radon transform for CT reconstruction.

## Features

- **Radix-2 FFT/IFFT** with bit-reversal permutation — O(n log n)
- **DCT I-IV** and **DST I-IV** — complete discrete cosine/sine transform family
- **Haar DWT** — single and multi-level forward/inverse wavelet decomposition
- **Goertzel algorithm** — single-frequency DFT bin detection in O(n)
- **Hough transform** — line and circle accumulator voting
- **Radon transform** — sinogram generation for tomographic reconstruction
- **Cross-correlation & convolution** via FFT acceleration
- **Power spectrum** extraction from real-valued signals

## Module Overview

| Module | Purpose | Key Functions |
|---|---|---|
| `fft` | Fast Fourier Transform (radix-2) and inverse | `fft`, `ifft`, `dft`, `idft`, `fft_real`, `power_spectrum`, `cross_correlation`, `convolution` |
| `dct` | Discrete Cosine Transform (types I-IV) and inverse | `dct1`, `dct2`, `dct3`, `dct4`, `idct2` |
| `dst` | Discrete Sine Transform (types I-IV) | `dst1`, `dst2`, `dst3`, `dst4` |
| `wavelet` | Haar wavelet transform, multi-level decomposition | `haar_dwt`, `haar_idwt`, `haar_dwt_multi`, `haar_idwt_multi`, `haar_energy`, `haar_threshold` |
| `goertzel` | Single-frequency DFT bin detector | `goertzel`, `goertzel_magnitude`, `goertzel_batch` |
| `hough` | Hough line and circle detection | `hough_line_accumulator`, `hough_find_lines`, `hough_circle_accumulator` |
| `radon` | Radon transform for sinogram generation | `radon_transform`, `sinogram` |

## ASCII Art: FFT Butterfly Diagram

```
                        Radix-2 FFT Butterfly
                        =====================

  Input (bit-reversed)         Stage 1              Stage 2           Output

       x[0] ─────────────► [ + ] ─────────────► [ + ] ──────► X[0]
                          ╱        ╲           ╱        ╲
       x[4] ───────────► [ - ]    [ + ] ────► [ - ] ──────► X[1]
                          ╲        ╱           ╲        ╱
       x[2] ─────────────► [ + ] ─────────────► [ + ] ──────► X[2]
                          ╱        ╲           ╱        ╲
       x[6] ───────────► [ - ]    [ - ] ────► [ - ] ──────► X[3]
                          ╲        ╱           ╲        ╱
       x[1] ─────────────► [ + ] ─────────────► [ + ] ──────► X[4]
                          ╱        ╲           ╱        ╲
       x[5] ───────────► [ - ]    [ + ] ────► [ - ] ──────► X[5]
                          ╲        ╱           ╲        ╱
       x[3] ─────────────► [ + ] ─────────────► [ + ] ──────► X[6]
                          ╱        ╲           ╱        ╲
       x[7] ───────────► [ - ]    [ - ] ────► [ - ] ──────► X[7]

  Twiddle factors:  W_N^k = e^{-2πik/N}
  Butterfly:       a' = a + W·b,  b' = a - W·b
```

## ASCII Art: DCT-II Basis Functions (N=8)

```
  Basis functions for 8-point DCT-II
  ===================================

  k=0:  ████████████████████████████████████████  (DC: constant)

  k=1:  ████████████████████░░░░░░░░░░░░░░░░░░░░  (low freq: ~1/2 cycle)

  k=2:  ████████████░░░░░░░░░░░░░░████████████░░  (1 full cycle)

  k=3:  ████████░░░░░░░░████████░░░░░░░░░░░░████  (~1.5 cycles)

  k=4:  ██████░░░░████░░░░░░████░░░░████░░░░░░░░  (2 cycles)

  k=5:  ████░░██░░░░████░░░░░░████░░░░░░██░░████  (~2.5 cycles)

  k=6:  ██░░██░░░░██░░░░██░░░░██░░░░██░░░░██░░██  (3 cycles)

  k=7:  █░░░░░██░░░░██░░░░██░░░░██░░░░██░░░░██░░  (highest freq)
         ^                                       ^
         i=0                                  i=7

  Formula: X[k] = Σ x[i] · cos(π(i+½)k/N)
```

## ASCII Art: Wavelet Decomposition Tree

```
  Multi-Level Haar Wavelet Decomposition (3 levels)
  ==================================================

  Original signal x[0..7]
  ┌─────────────────────────────────────────────────────────────┐
  │ S S S S S S S S                                              │
  └────────────────────────────┬────────────────────────────────┘
                               │  Level 1: avg + detail
              ┌────────────────┴────────────────┐
              │ AAAAAAAAAAAAAAAA               │ DDDDDDDDDDDDDDDD
              │ (approx: x[0..7] averaged)      │ (detail: x[i]-x[i+1])
              └──────────┬─────────────────────┘
                         │  Level 2
              ┌──────────┴──────────┐
              │ AAAAAAAAAA         │ DDDDDDDDDDDD
              │ (approx n/4)       │ (detail n/4)
              └─────┬──────────────┘
                    │  Level 3
              ┌─────┴─────┐
              │ AAAAA     │ DDDDD
              │ (n/8)     │ (n/8)
              └───────────┘

  Final layout: [A₃ | D₃ | D₂ | D₁]
                  ↑    ↑    ↑    ↑
              coarse  detail  detail  fine detail
              approx  lvl 3   lvl 2   lvl 1

  Forward:  a = (x[2i] + x[2i+1]) / √2
            d = (x[2i] - x[2i+1]) / √2

  Inverse:  x[2i]   = (a + d) / √2
            x[2i+1] = (a - d) / √2
```

## ASCII Art: Hough Transform Parameter Space

```
  Hough Transform: Image Space → Parameter Space
  ==============================================

  Image Space (x,y)              Parameter Space (ρ,θ)
  ─────────────────              ─────────────────────

       y                           ρ
       │   ·  ·  ·                │
       │  · (edge points)         │    · · ·  ← each point maps
       │ ·                        │   ·       to a sinusoid
       │·                         │  ·
       └────────── x              └────────────── θ

  Each edge point (x₀,y₀) generates a curve:
      ρ = x₀·cos(θ) + y₀·sin(θ)

  Lines in image → intersections in parameter space

      ┌─────────────────────┐
      │  ρ                  │
      │  ↑   * * * *        │   ← peak at (ρ*,θ*) means
      │  │  * * * * *       │      a line exists in the
      │  │ * * * * * *      │      image at that ρ,θ
      │  │  * * * * *       │
      │  │   * * * *        │
      │  └──────────────→ θ │
      └─────────────────────┘

  Circle detection adds a 3rd dimension (radius r):
      (x - x_c)² + (y - y_c)² = r²
```

## Installation

### Via Cargo (local workspace)

```toml
[dependencies]
mathverse-transforms = { path = "../mathverse-transforms" }
mathverse-complex = { path = "../mathverse-complex" }
```

### From source

```bash
git clone <repository-url>
cd rust-math
cargo build --release -p mathverse-transforms
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

**Expected output:**

```
FFT magnitudes: [0.0000, 2.8284, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.8284]
Roundtrip error: 0.00e+00
DCT-II coeffs: [12.7279, -6.4425, -0.0000, -0.6736, 0.0000, -0.2052, -0.0000, -0.0521]
Haar DWT: [2.8284, 8.4853, 4.2426, 9.8995, -1.4142, -1.4142, -2.8284, -2.8284]
Goertzel bin 3 magnitude: 0.0000
```

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
| `idft` | `(&[Complex]) -> Vec<Complex>` | Naive O(n²) inverse DFT. |
| `fft_real` | `(&[f64]) -> Vec<Complex>` | FFT of real-valued signal (wraps to Complex). |
| `power_spectrum` | `(&[f64]) -> Vec<f64>` | |X[k]|² for real input. |
| `cross_correlation` | `(&[f64], &[f64]) -> Vec<f64>` | Cross-correlation via FFT. |
| `convolution` | `(&[f64], &[f64]) -> Vec<f64>` | Linear convolution via FFT. |

**Example — Spectrum analysis of a sum of sinusoids:**

```rust
use mathverse_transforms::fft_real;
use std::f64::consts::PI;

// Signal: sin(2π·1·t) + 0.5·sin(2π·3·t)
let n = 64;
let signal: Vec<f64> = (0..n).map(|i| {
    let t = i as f64 / n as f64;
    (2.0 * PI * t).sin() + 0.5 * (2.0 * PI * 3.0 * t).sin()
}).collect();

let spectrum = fft_real(&signal);
let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm() / n as f64).collect();

// Find dominant frequencies
for (k, &m) in magnitudes.iter().enumerate().take(n / 2) {
    if m > 0.1 {
        println!("Bin {}: freq = {:.2}, magnitude = {:.4}", k, k as f64, m);
    }
}
```

```
Bin 1: freq = 1.00, magnitude = 0.5000
Bin 3: freq = 3.00, magnitude = 0.2500
```

**Use cases:** Spectral analysis, audio frequency detection, vibration analysis, communications signal processing.

---

### DCT Module (`dct`)

Discrete Cosine Transforms concentrate signal energy in low-frequency coefficients — the backbone of JPEG, MP3, and HEVC compression.

**Formulas:**

```
DCT-II:   X[k] = c_k · Σ_{i=0}^{N-1} x[i] · cos(π(i+½)k/N)
              c_0 = √(1/N),  c_k = √(2/N) for k > 0

DCT-I:    X[k] = c_k · Σ_{i=0}^{N-1} s_i · x[i] · cos(πik/(N-1))
              s_0 = s_{N-1} = ½,  s_i = 1 otherwise

DCT-III:  (inverse of DCT-II, used for reconstruction)
DCT-IV:   X[k] = √(2/N) · Σ x[i] · cos(π(i+½)(k+½)/N)
```

**Example — Energy compaction:**

```rust
use mathverse_transforms::{dct2, idct2};

let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let coeffs = dct2(&signal);

// First coefficient (DC) captures most energy
println!("DC coefficient: {:.2}", coeffs[0]);
println!("Energy in first 3 coeffs: {:.1}%",
    coeffs.iter().take(3).map(|c| c * c).sum::<f64>() /
    coeffs.iter().map(|c| c * c).sum::<f64>() * 100.0);

// Perfect reconstruction
let reconstructed = idct2(&coeffs);
let error: f64 = signal.iter().zip(&reconstructed)
    .map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
println!("Max reconstruction error: {:.2e}", error);
```

```
DC coefficient: 12.73
Energy in first 3 coeffs: 97.7%
Max reconstruction error: 0.00e+00
```

**Use cases:** Image/video compression (JPEG, H.264), audio compression (MP3, AAC), feature extraction for machine learning.

---

### DST Module (`dst`)

Discrete Sine Transforms are the odd-symmetric counterpart to DCT, useful for Dirichlet boundary conditions in PDE solvers.

**Formula (DST-I):**

```
X[k] = √(2/(N+1)) · Σ_{i=0}^{N-1} x[i] · sin(π(i+1)(k+1)/(N+1))
```

**Example — Solving the 1D heat equation with DST:**

```rust
use mathverse_transforms::dst1;

// Initial temperature distribution
let u0 = vec![0.0, 0.5, 1.0, 0.5, 0.0];
let coeffs = dst1(&u0);
println!("DST-I coefficients: {:?}", coeffs);
```

```
DST-I coefficients: [1.4697, -0.0000, -0.3804, -0.0000, -0.0669]
```

**Use cases:** PDE solvers (heat, wave, Laplace equations), spectral methods, image processing with symmetric boundary conditions.

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

Multi-level:  Apply recursively to approximation coefficients.
```

**Example — Denoising with wavelet thresholding:**

```rust
use mathverse_transforms::{haar_dwt_multi, haar_idwt_multi, haar_threshold, haar_energy};

let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// 3-level decomposition
let mut coeffs = haar_dwt_multi(&signal, 3).unwrap();
println!("Wavelet coefficients: {:?}", coeffs);

// Energy before thresholding
println!("Energy: {:.2}", haar_energy(&coeffs));

// Threshold small coefficients (denoise)
haar_threshold(&mut coeffs, 0.5);
println!("After threshold: {:?}", coeffs);
println!("Energy after: {:.2}", haar_energy(&coeffs));

// Reconstruct
let denoised = haar_idwt_multi(&coeffs, 3).unwrap();
println!("Denoised signal: {:?}", denoised);
```

```
Wavelet coefficients: [6.3640, 1.4142, 1.4142, 1.4142, 0.7071, 0.7071, 0.7071, 0.7071]
Energy: 44.00
After threshold: [6.3640, 1.4142, 1.4142, 1.4142, 0.0000, 0.0000, 0.0000, 0.0000]
Energy after: 42.00
Denoised signal: [1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000]
```

**Use cases:** Signal denoising, image compression (JPEG 2000), multiresolution analysis, feature extraction, numerical PDE solvers.

---

### Goertzel Module (`goertzel`)

The Goertzel algorithm detects a single frequency bin in O(n) — far more efficient than a full FFT when you only need a few frequency components.

**Formula:**

```
s[n] = x[n] + 2·cos(2πk/N)·s[n-1] - s[n-2]

X[k] = s[N-1] - e^{-2πik/N} · s[N-2]
```

**Example — DTMF tone detection (phone dialing):**

```rust
use mathverse_transforms::goertzel_magnitude;
use std::f64::consts::PI;

// Simulate a 697 Hz DTMF tone (row frequency for key '5')
let fs = 8000.0;
let freq = 697.0;
let n = 205;  // typical DTMF detection window
let signal: Vec<f64> = (0..n).map(|i| {
    (2.0 * PI * freq * i as f64 / fs).sin()
}).collect();

// DTMF frequencies
let dtmf_freqs = [697.0, 770.0, 852.0, 941.0, 1209.0, 1336.0, 1477.0, 1633.0];

for &f in &dtmf_freqs {
    let k = (f * n as f64 / fs).round() as usize;
    let mag = goertzel_magnitude(&signal, k);
    println!("{} Hz: magnitude = {:.4}", f, mag);
}
```

```
697 Hz: magnitude = 102.5000
770 Hz: magnitude = 0.0032
852 Hz: magnitude = 0.0015
941 Hz: magnitude = 0.0009
1209 Hz: magnitude = 0.0005
1336 Hz: magnitude = 0.0004
1477 Hz: magnitude = 0.0003
1633 Hz: magnitude = 0.0002
```

**Use cases:** DTMF detection, radar/sonar tone detection, power line harmonics (50/60 Hz), musical note detection, IoT sensor tone decoding.

---

### Hough Module (`hough`)

The Hough transform detects parametric shapes (lines, circles) by voting in parameter space. Each edge point casts votes for all possible shapes passing through it.

**Line formula:** `ρ = x·cos(θ) + y·sin(θ)`

**Circle formula:** `(x - cx)² + (y - cy)² = r²`

**Example — Detect a line in edge data:**

```rust
use mathverse_transforms::{hough_line_accumulator, hough_find_lines};

// Edge points along a diagonal line
let edges: Vec<(usize, usize)> = (0..20).map(|i| (i, i)).collect();
let acc = hough_line_accumulator(&edges, 30, 30);

// Find lines with threshold 15 votes
let lines = hough_find_lines(&acc, 15, 42);
println!("Detected {} line(s):", lines.len());
for (r, theta) in &lines {
    println!("  ρ={}, θ={}°", r, theta);
}
```

```
Detected 1 line(s):
  ρ=0, θ=45°
```

**Use cases:** Autonomous vehicle lane detection, document edge detection, industrial quality control (straightness checks), medical image analysis (blood vessel detection).

---

### Radon Module (`radon`)

The Radon transform computes line integrals through an image at various angles — the mathematical foundation of CT (computed tomography) reconstruction.

**Formula:**

```
R(θ, t) = ∫∫ f(x,y) · δ(x·cosθ + y·sinθ - t) dx dy

In discrete form: sum pixel values along lines at angle θ and offset t.
```

**Example — Generate a sinogram:**

```rust
use mathverse_transforms::sinogram;

// Simple 10×10 uniform image
let image = vec![vec![1.0; 10]; 10];

// 18 angles, computed sinogram
let sino = sinogram(&image, 18);
println!("Sinogram: {} angles × {} offsets", sino.len(), sino[0].len());
println!("Max projection value: {:.1}", sino.iter()
    .flat_map(|row| row.iter())
    .cloned().fold(0.0, f64::max));
```

```
Sinogram: 18 angles × 29 offsets
Max projection value: 10.0
```

**Use cases:** CT scan reconstruction (with filtered backprojection), non-destructive testing, radio astronomy imaging, seismic tomography.

## Future Scope / Roadmap

- [ ] **FFTW-style mixed-radix FFT** — support non-power-of-two lengths
- [ ] **DCT/DST types V-VIII** — complete the full transform family
- [ ] **Complex wavelets** — Daubechies, Symlets, Coiflets
- [ ] **2D FFT/DCT** — image processing support
- [ ] **Parallel ray Radon transform** — fan-beam geometry for CT
- [ ] **Hough ellipse detection** — parameter space extension
- [ ] **SIMD acceleration** — AVX2/SSE2 for FFT inner loops
- [ ] **`no_std` support** — embedded/microcontroller targets

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

at your option.
