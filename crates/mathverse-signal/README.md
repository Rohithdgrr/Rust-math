# MathVerse Signal

[![Crates.io](https://img.shields.io/crates/v/mathverse-signal.svg)](https://crates.io/crates/mathverse-signal)
[![docs.rs](https://docs.rs/mathverse-signal/badge.svg)](https://docs.rs/mathverse-signal)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Pure-Rust signal processing library providing FIR/IIR filters, convolution, windowing, spectral estimation, peak detection, and modulation — zero external dependencies.

---

## Features

- **Linear convolution & correlation** — O(n²) time-domain, plus FFT-accelerated path
- **FIR filter design** — Lowpass, highpass, bandpass with Hamming windowing
- **IIR biquad filters** — Lowpass, highpass, bandpass, bandstop with cascade/parallel topology
- **Window functions** — Hamming, Hanning, Blackman, Bartlett, Kaiser, Gaussian, flat-top
- **Spectral estimation** — Periodogram, Welch PSD, autocorrelation
- **Feature detection** — Peaks, envelope, onset, zero-crossing rate, spectral centroid/rolloff
- **Modulation** — AM, FM, FSK, BPSK modulation and demodulation
- **Utility** — dB conversions, RMS, dynamic range, moving average, median filter

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `convolution` | Linear convolution, correlation, FIR filtering, peak detection | `convolve`, `correlate`, `fir`, `fir_lowpass`, `fir_highpass`, `fir_bandpass`, `moving_average`, `median_filter`, `find_peaks` |
| `filter_design` | Filter analysis, bilinear transform, impulse/step response | `bilinear_transform`, `impulse_response`, `step_response`, `group_delay` |
| `iir` | Second-order IIR (biquad) filters with standard configurations | `Biquad::lowpass`, `Biquad::highpass`, `Biquad::bandpass`, `Biquad::bandstop`, `cascade`, `parallel` |
| `windowing` | Window functions for spectral analysis and filter design | `window_hamming`, `window_hanning`, `window_blackman`, `window_bartlett`, `window_kaiser`, `window_gaussian`, `window_flat_top`, `apply_window` |
| `spectrum` | Power spectral density estimation and autocorrelation | `periodogram`, `welch_psd`, `autocorrelation`, `energy`, `parseval` |
| `detection` | Envelope, onset, zero-crossing, spectral features | `envelope`, `onset_detection`, `zero_crossing_rate`, `spectral_centroid`, `spectral_rolloff`, `dynamic_range` |
| `modulation` | AM/FM modulation/demodulation, digital modulation | `amplitude_modulate`, `frequency_modulate`, `am_demodulate`, `fsk_modulate`, `bpsk_modulate`, `db_to_linear`, `linear_to_db` |

## Installation

```toml
[dependencies]
mathverse-signal = { path = "crates/mathverse-signal" }
```

## Quick Start

```rust
use mathverse_signal::*;

fn main() {
    // Design and apply a lowpass FIR filter
    let taps = fir_lowpass(0.1, 31);  // cutoff at 0.1×Nyquist, 31 taps
    let signal: Vec<f64> = (0..100).map(|i| {
        (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 3.0).sin()
    }).collect();
    let filtered = fir(&signal, &taps);

    // Compute power spectrum
    let psd = welch_psd(&signal, 32, 16);
    println!("PSD bins: {}", psd.len());

    // IIR biquad: highpass at 1000 Hz, 48kHz sample rate
    let bq = Biquad::highpass(48000.0, 1000.0);
    let output = bq.process(&signal);

    // AM modulation
    let carrier = vec![1.0; 100];
    let msg: Vec<f64> = (0..100).map(|i| {
        (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()
    }).collect();
    let am = amplitude_modulate(&carrier, &msg, 0.8);
    println!("AM signal length: {}", am.len());
}
```

---

## Module Documentation

### Convolution Module (`convolution`)

Implements time-domain linear convolution, FIR filtering, and standard signal analysis utilities.

**Formula:**

```
Convolution:  y[n] = Σ_{k=0}^{M-1} x[n-k] · h[k]
Correlation:  r[n] = Σ_{k=0}^{M-1} x[n+k] · h[k]
```

### IIR Module (`iir`)

Second-order biquad filters — the building blocks of all IIR filter implementations. Supports standard audio filter types with cascade and parallel topologies.

**Transfer function:**

```
H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²) / (1 + a₁z⁻¹ + a₂z⁻²)
```

### Windowing Module (`windowing`)

Window functions shape finite-length signals to control spectral leakage in DFT analysis and FIR filter design.

### Spectrum Module (`spectrum`)

Power spectral density estimation and autocorrelation for stationary signal analysis.

### Detection Module (`detection`)

Feature extraction from signals: envelope following, onset detection, zero-crossing rate, and spectral features.

### Modulation Module (`modulation`)

Analog and digital modulation/demodulation for communication systems.

**Formulas:**

```
AM:       y(t) = c(t) · (1 + m · m(t))
FM:       y(t) = sin(ωc·t + β · ∫m(τ)dτ)
BPSK:     y(t) = sin(2πfc·t + φ),  φ ∈ {0, π}
```

---

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE).
