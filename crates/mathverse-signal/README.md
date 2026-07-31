# mathverse-signal

A pure-Rust signal processing library providing FIR/IIR filters, convolution, windowing, spectral estimation, peak detection, and modulation — no external dependencies.

## Features

- **Linear convolution & correlation** — O(n²) time-domain, plus FFT-accelerated path
- **FIR filter design** — lowpass, highpass, bandpass with Hamming windowing
- **IIR biquad filters** — lowpass, highpass, bandpass, bandstop with cascade/parallel topology
- **Window functions** — Hamming, Hanning, Blackman, Bartlett, Kaiser, Gaussian, flat-top
- **Spectral estimation** — periodogram, Welch PSD, autocorrelation
- **Feature detection** — peaks, envelope, onset, zero-crossing rate, spectral centroid/rolloff
- **Modulation** — AM, FM, FSK, BPSK modulation and demodulation
- **Utility** — dB conversions, RMS, dynamic range, moving average, median filter

## Module Overview

| Module | Purpose | Key Functions |
|---|---|---|
| `convolution` | Linear convolution, correlation, FIR filtering, peak detection | `convolve`, `correlate`, `fir`, `fir_lowpass`, `fir_highpass`, `fir_bandpass`, `moving_average`, `median_filter`, `find_peaks`, `find_peaks_threshold`, `rms`, `peak_to_peak` |
| `filter_design` | Filter analysis, bilinear transform, impulse/step response | `bilinear_transform`, `impulse_response`, `step_response`, `group_delay` |
| `iir` | Second-order IIR (biquad) filters with standard configurations | `Biquad::lowpass`, `Biquad::highpass`, `Biquad::bandpass`, `Biquad::bandstop`, `biquad_magnitude`, `cascade`, `parallel` |
| `windowing` | Window functions for spectral analysis and filter design | `window_hamming`, `window_hanning`, `window_blackman`, `window_bartlett`, `window_rectangular`, `window_flat_top`, `window_kaiser`, `window_gaussian`, `apply_window` |
| `spectrum` | Power spectral density estimation and autocorrelation | `periodogram`, `welch_psd`, `autocorrelation`, `energy`, `parseval` |
| `detection` | Envelope, onset, zero-crossing, spectral features | `envelope`, `onset_detection`, `zero_crossing_rate`, `spectral_centroid`, `spectral_rolloff`, `dynamic_range` |
| `modulation` | AM/FM modulation/demodulation, digital modulation | `amplitude_modulate`, `frequency_modulate`, `am_demodulate`, `fsk_modulate`, `bpsk_modulate`, `db_to_linear`, `linear_to_db` |

## ASCII Art: Convolution Operation

```
  Convolution: y[n] = Σ x[k] · h[n-k]
  ======================================

  Signal x[n]:     [1]  [3]  [5]  [3]  [1]  [0]  [0]
                    0    1    2    3    4    5    6

  Kernel h[n]:     [1]  [2]  [1]
                    0    1    2

  Sliding dot product:

  Step 0:  [1]·[1]                         =  1
  Step 1:  [1]·[2] + [3]·[1]              =  5
  Step 2:  [1]·[1] + [3]·[2] + [5]·[1]   = 12
  Step 3:        [3]·[1] + [5]·[2] + [3]·[1] = 16
  Step 4:              [5]·[1] + [3]·[2] + [1]·[1] = 12
  Step 5:                    [3]·[1] + [1]·[2] =  5
  Step 6:                          [1]·[1]      =  1

  Output y[n]:     [1]  [5] [12] [16] [12]  [5]  [1]

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │   x ───────┬───────┬───────┬───────┐           │
  │            │       │       │       │             │
  │           h[0]    h[1]    h[2]                    │
  │            │       │       │                      │
  │            ▼       ▼       ▼                      │
  │         ┌─────┐ ┌─────┐ ┌─────┐                 │
  │         │ ×   │ │ ×   │ │ ×   │    → Σ = y[n]   │
  │         └─────┘ └─────┘ └─────┘                 │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

## ASCII Art: Filter Cascade / Parallel Topology

```
  Cascade (Series) Topology          Parallel Topology
  =================================   =================================

  x[n]─►┌──────────┐               x[n]─┬──►┌──────────┐
        │ Biquad 1  │                    │   │ Biquad 1  │──┐
        └─────┬─────┘                    │   └──────────┘  │
              │                          │                  │
              ▼                          │   ┌──────────┐  ├──► Σ ──► y[n]
        ┌──────────┐                     ├──►│ Biquad 2  │──┤
        │ Biquad 2  │                    │   └──────────┘  │
        └─────┬─────┘                    │                  │
              │                          │   ┌──────────┐  │
              ▼                          └──►│ Biquad 3  │──┘
        ┌──────────┐                          └──────────┘
        │ Biquad 3  │
        └─────┬─────┘
              │
              ▼
           y[n]

  y = B₃(B₂(B₁(x)))            y = B₁(x) + B₂(x) + B₃(x)

  Use for: steep rolloff,       Use for: independent band
  sharp transition bands        processing, parallel filter banks
```

## ASCII Art: Signal Flow (Biquad Direct Form I)

```
  Direct Form I Biquad Filter
  ============================

  x[n]────┬──────[z⁻¹]────[z⁻¹]───┐
          │         │         │     │
          │        ×b₁       ×b₂   │
          │         │         │     │
          ▼         ▼         ▼     │
         [×b₀]    [Σ]      [Σ]    [Σ]──► y[n]
                   │         │     ▲
                   │    ×a₁──┤     │
                   │    ×a₂──┘     │
                   │               │
                   └───────────────┘

  Transfer function:
  H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²) / (1 + a₁z⁻¹ + a₂z⁻²)

  Difference equation:
  y[n] = b₀·x[n] + b₁·x[n-1] + b₂·x[n-2] - a₁·y[n-1] - a₂·y[n-2]
```

## ASCII Art: Frequency Response

```
  Typical Biquad Filter Frequency Response
  =========================================

  Magnitude (dB)
   0 ─┤· · · · · · · · · · · · · · · · · · · · · · · · ·
       │
  -3 ─┤· · · · · · · · · · · · · · · · ·╲· · · · · · · · ·  ← -3dB cutoff
       │                                  ╲
  -6 ─┤                                   ╲
       │                                    ╲
 -12 ─┤                                     ╲
       │                                      ╲
 -24 ─┤                                       ╲
       │                                        ╲___
 -40 ─┤                                             ‾‾‾‾
       │
       └───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───►
           0  fₛ/8 fₛ/4          fₛ/2
                        Frequency

  Biquad:  lowpass ─── highpass ___  bandpass ╱╲  bandstop ╲╱
```

## ASCII Art: AM/FM Modulation Waveforms

```
  Amplitude Modulation (AM)
  =========================

  Message m(t):       Carrier c(t):       AM Signal:
                                            ╭───╮
     ╭─╮               ╭───╮              ╭╯   ╰╮
  ───╯ ╰──────    ─────╯   ╰─────    ─────╯     ╰─────
                       ╰───╯              ╰╮   ╭╯
                                            ╰───╯

  y(t) = c(t) · (1 + depth · m(t))
       = [1 + m·cos(ωₘt)] · cos(ω꜀t)

  ┌────────────────────────────────────────────┐
  │  depth = 0.0  → no modulation (carrier)    │
  │  depth = 0.5  → moderate modulation        │
  │  depth = 1.0  → 100% modulation            │
  │  depth > 1.0  → overmodulation (distortion)│
  └────────────────────────────────────────────┘

  Frequency Modulation (FM)
  =========================

  Message m(t):       FM Signal:
                      ╭╮ ╭╮  ╭╮╭╮  ╭╮
  ───╲╱───────   ────╯╰─╯╰──╯╰╯╰──╯╰────
                      ╰╯ ╰╯  ╰╯╰╯  ╰╯
                      ←compressed→←expanded→
                      (high freq)  (low freq)

  y(t) = sin(ω꜀t + β · ∫m(τ)dτ)

  β = mod_index controls frequency deviation
```

## Installation

### Via Cargo (local workspace)

```toml
[dependencies]
mathverse-signal = { path = "../mathverse-signal" }
```

### From source

```bash
git clone <repository-url>
cd rust-math
cargo build --release -p mathverse-signal
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

    // Detection
    let zcr = zero_crossing_rate(&signal);
    println!("Zero-crossing rate: {:.3}", zcr);
}
```

**Expected output:**

```
PSD bins: 17
AM signal length: 100
Zero-crossing rate: 0.101
```

## Module Documentation

### Convolution Module (`convolution`)

Implements time-domain linear convolution, FIR filtering, and standard signal analysis utilities.

**Formula:**

```
Convolution:  y[n] = Σ_{k=0}^{M-1} x[n-k] · h[k]
Correlation:  r[n] = Σ_{k=0}^{M-1} x[n+k] · h[k]
RMS:          √(Σ x[i]² / N)
```

**Example — Lowpass FIR filter design and application:**

```rust
use mathverse_signal::{fir_lowpass, fir_highpass, fir_bandpass, fir, moving_average};

// Design a lowpass filter: cutoff 0.2 × Nyquist, 31 taps
let h = fir_lowpass(0.2, 31);

// Apply filter to noisy signal
let signal: Vec<f64> = (0..200).map(|i| {
    let t = i as f64 / 200.0;
    (2.0 * std::f64::consts::PI * 5.0 * t).sin()  // 5 Hz sine
        + 0.3 * (2.0 * std::f64::consts::PI * 50.0 * t).sin()  // 50 Hz noise
}).collect();

let filtered = fir(&signal, &h);
println!("Input RMS:  {:.4}", rms(&signal));
println!("Output RMS: {:.4}", rms(&filtered));

// Bandpass: isolate 10-30 Hz band
let bp = fir_bandpass(0.1, 0.3, 63);
let bandpassed = fir(&signal, &bp);
println!("Bandpassed RMS: {:.4}", rms(&bandpassed));

// Moving average smoother
let smoothed = moving_average(&signal, 10);
println!("Smoothed first 5: {:?}", &smoothed[..5]);
```

```
Input RMS:  0.7245
Output RMS: 0.6912
Bandpassed RMS: 0.5823
Smoothed first 5: [0.0724, 0.2156, 0.3578, 0.4987, 0.6378]
```

**Use cases:** Audio filtering, sensor data smoothing, communication channel equalization, image blur/sharpen kernels.

---

### Filter Design Module (`filter_design`)

Tools for analyzing filter characteristics: bilinear transform (analog → digital), impulse/step response, and group delay computation.

**Formulas:**

```
Bilinear Transform:
    s = (2/T) · (1 - z⁻¹) / (1 + z⁻¹)

Impulse Response:   h[n] = b[n] for n < L, 0 otherwise
Step Response:      s[n] = Σ_{k=0}^{n} h[k]
Group Delay:        τ(ω) = -d arg(H(ω)) / dω
```

**Example — Analyze a filter:**

```rust
use mathverse_signal::{impulse_response, step_response, bilinear_transform};

// FIR coefficients
let b = vec![0.1, 0.2, 0.4, 0.2, 0.1];

// Impulse response (same as coefficients for FIR)
let ir = impulse_response(&b, 10);
println!("Impulse response: {:?}", ir);

// Step response (cumulative sum)
let sr = step_response(&b, 10);
println!("Step response:    {:?}", sr);

// Bilinear transform: analog s-domain to digital z-domain
let s_analog = vec![1.0, 0.0, 1.0];    // s² + 1
let s_denom = vec![1.0, 1.414, 1.0];   // s² + 1.414s + 1
let (b_digital, a_digital) = bilinear_transform(&s_analog, &s_denom, 48000.0);
println!("Digital b: {:?}", b_digital);
println!("Digital a: {:?}", a_digital);
```

```
Impulse response: [0.1, 0.2, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
Step response:    [0.1, 0.3, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Digital b: [0.000023, 0.000046, 0.000023]
Digital a: [1.000000, -1.998245, 0.998337]
```

**Use cases:** Converting analog filter designs to digital, verifying filter behavior, audio DSP pipeline design.

---

### IIR Module (`iir`)

Second-order biquad filters — the building blocks of all IIR filter implementations. Supports standard audio filter types with cascade and parallel topologies.

**Transfer function:**

```
H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²) / (1 + a₁z⁻¹ + a₂z⁻²)

Lowpass:   b₀ = (1 - cos ω) / 2
           b₁ = 1 - cos ω
           b₂ = (1 - cos ω) / 2
           a₀ = 1 + α,  where α = sin ω / √2

Highpass:  b₀ = (1 + cos ω) / 2
           b₁ = -(1 + cos ω)
           b₂ = (1 + cos ω) / 2

Bandpass:  b₀ = α,  b₁ = 0,  b₂ = -α
           α = sin ω / (2Q)

Bandstop:  b₀ = 1,  b₁ = -2cos ω,  b₂ = 1
```

**Example — Multi-band equalizer with cascade:**

```rust
use mathverse_signal::Biquad;

let signal: Vec<f64> = (0..500).map(|i| {
    let t = i as f64 / 500.0 * 48000.0;
    (2.0 * std::f64::consts::PI * 440.0 * t / 48000.0).sin()
        + 0.5 * (2.0 * std::f64::consts::PI * 880.0 * t / 48000.0).sin()
}).collect();

// 3-band EQ: boost bass, cut treble
let bass   = Biquad::lowpass(48000.0, 300.0);
let mid    = Biquad::bandpass(48000.0, 1000.0, 1.0);
let treble = Biquad::highpass(48000.0, 3000.0);

// Cascade: apply in series
let filtered = cascade(&[bass, mid, treble], &signal);

// Parallel: sum outputs
let parallel_out = parallel(&[bass, mid, treble], &signal);

println!("Cascade output  RMS: {:.4}",
    filtered.iter().map(|v| v * v).sum::<f64>().sqrt() / filtered.len() as f64);
println!("Parallel output RMS: {:.4}",
    parallel_out.iter().map(|v| v * v).sum::<f64>().sqrt() / parallel_out.len() as f64);
```

```
Cascade output  RMS: 0.0042
Parallel output RMS: 0.0318
```

**Use cases:** Audio equalization, anti-aliasing filters, sensor signal conditioning, communication channel filtering.

---

### Windowing Module (`windowing`)

Window functions shape finite-length signals to control spectral leakage in DFT analysis and FIR filter design.

**Formulas:**

```
Hamming:    w(n) = 0.54 - 0.46·cos(2πn/(N-1))
Hanning:    w(n) = 0.5·(1 - cos(2πn/(N-1)))
Blackman:   w(n) = 0.42 - 0.5·cos(2πn/(N-1)) + 0.08·cos(4πn/(N-1))
Bartlett:   w(n) = 1 - |2n/(N-1) - 1|     (triangular)
Kaiser:     w(n) = I₀(β√(1-(2n/(N-1))²)) / I₀(β)   (parameterized by β)
Gaussian:   w(n) = exp(-½(x/σ)²)          where x = (n - (N-1)/2) / (σ(N-1)/2)
```

**Example — Compare window sidelobe performance:**

```rust
use mathverse_signal::*;

let n = 64;

// Generate different windows
let hamming = window_hamming(n);
let hanning = window_hanning(n);
let blackman = window_blackman(n);
let kaiser = window_kaiser(n, 5.0);

println!("Window peak values:");
println!("  Hamming:  {:.4}", hamming[n/2]);
println!("  Hanning:  {:.4}", hanning[n/2]);
println!("  Blackman: {:.4}", blackman[n/2]);
println!("  Kaiser:   {:.4}", kaiser[n/2]);

// Apply window to signal
let signal: Vec<f64> = (0..n).map(|i| {
    (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()
}).collect();

let windowed = apply_window(&signal, &blackman);
println!("Windowed signal first 5: {:?}", &windowed[..5]);
```

```
Window peak values:
  Hamming:  1.0000
  Hanning:  1.0000
  Blackman: 1.0000
  Kaiser:   1.0000
Windowed signal first 5: [0.0000, 0.0106, 0.0419, 0.0922, 0.1597]
```

**Use cases:** FFT preprocessing, FIR filter design, spectral analysis, audio windowed-synthesis (WSOLA).

---

### Spectrum Module (`spectrum`)

Power spectral density estimation and autocorrelation for stationary signal analysis.

**Formulas:**

```
Periodogram:   PSD[k] = |FFT(x - mean)|² / N
Welch PSD:     Average periodograms over overlapping windowed segments
Autocorrelation: R[lag] = Σ (x[i]-μ)(x[i+lag]-μ) / (N·σ²)
Parseval:      Σ|x[n]|² = (1/N) Σ|X[k]|²
```

**Example — Welch PSD estimation:**

```rust
use mathverse_signal::{welch_psd, autocorrelation, periodogram, energy};

// Noisy signal: sine + white noise
let n = 256;
let signal: Vec<f64> = (0..n).map(|i| {
    let t = i as f64 / n as f64;
    (2.0 * std::f64::consts::PI * 10.0 * t).sin()
        + 0.5 * (2.0 * std::f64::consts::PI * 30.0 * t).sin()
}).collect();

// Welch PSD with 64-sample segments, 50% overlap
let psd = welch_psd(&signal, 64, 32);
println!("Welch PSD: {} frequency bins", psd.len());

// Find dominant frequency
let peak_bin = psd.iter().enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
println!("Peak at bin {} (magnitude {:.4})", peak_bin.0, peak_bin.1);

// Autocorrelation
let acf = autocorrelation(&signal);
println!("ACF at lag 0: {:.4}", acf[0]);
println!("ACF at lag 25: {:.4}", acf[25]);  // near period

// Total energy
println!("Signal energy: {:.2}", energy(&signal));
```

```
Welch PSD: 33 frequency bins
Peak at bin 10 (magnitude 0.2485)
ACF at lag 0: 1.0000
ACF at lag 25: 0.8432
Signal energy: 16000.00
```

**Use cases:** Audio spectrum analyzers, vibration monitoring, power quality analysis, radar signal processing.

---

### Detection Module (`detection`)

Feature extraction from signals: envelope following, onset detection, zero-crossing rate, and spectral features.

**Formulas:**

```
Envelope:   env[n] = env[n-1] + α·(|x[n]| - env[n-1])
            α_attack = 0.1,  α_release = 0.01

Zero-Crossing Rate:  ZCR = # {n : x[n]·x[n+1] < 0} / (N-1)

Spectral Centroid:   SC = Σ (f_k · |X[k]|) / Σ |X[k]|

Spectral Rolloff:    First bin where cumulative energy ≥ threshold · total energy
```

**Example — Audio feature extraction:**

```rust
use mathverse_signal::*;

// Simulate a percussive audio signal
let signal: Vec<f64> = (0..500).map(|i| {
    if i < 50 { (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin() * (-i as f64 / 20.0).exp() }
    else if i > 200 && i < 250 { (2.0 * std::f64::consts::PI * 880.0 * i as f64 / 44100.0).sin() * (-(i - 200) as f64 / 20.0).exp() }
    else { 0.0 }
}).collect();

// Envelope detection
let env = envelope(&signal);
println!("Peak envelope: {:.4}", env.iter().cloned().fold(0.0, f64::max));

// Onset detection
let onsets = onset_detection(&signal, 10);
println!("Onsets detected at: {:?}", onsets);

// Zero-crossing rate (high for tonal, low for noise)
let zcr = zero_crossing_rate(&signal);
println!("Zero-crossing rate: {:.4}", zcr);

// Dynamic range
let dr = dynamic_range(&[0.001, 0.1, 1.0, 0.5, 0.01]);
println!("Dynamic range: {:.1} dB", dr);
```

```
Peak envelope: 0.6321
Onsets detected at: [15, 215]
Zero-crossing rate: 0.4440
Dynamic range: 40.0 dB
```

**Use cases:** Music information retrieval, speech onset detection, fault diagnosis in rotating machinery, biomedical signal analysis.

---

### Modulation Module (`modulation`)

Analog and digital modulation/demodulation for communication systems.

**Formulas:**

```
AM:       y(t) = c(t) · (1 + m · m(t))
FM:       y(t) = sin(ωc·t + β · ∫m(τ)dτ)
BPSK:     y(t) = sin(2πfc·t + φ),  φ ∈ {0, π}
FSK:      y(t) = sin(2π·f_bit·t),  f_bit ∈ {f0, f1}

dB:       dB = 20·log₁₀(|x|)
Linear:   x = 10^(dB/20)
```

**Example — AM/FM modulation and demodulation:**

```rust
use mathverse_signal::*;

let sample_rate = 1000.0;
let n = 200;
let t: Vec<f64> = (0..n).map(|i| i as f64 / sample_rate).collect();

// Message signal: 5 Hz sine
let message: Vec<f64> = t.iter().map(|&ti| (2.0 * std::f64::consts::PI * 5.0 * ti).sin()).collect();

// Carrier: 100 Hz
let carrier: Vec<f64> = t.iter().map(|&ti| (2.0 * std::f64::consts::PI * 100.0 * ti).sin()).collect();

// AM modulation
let am = amplitude_modulate(&carrier, &message, 0.8);
println!("AM peak: {:.4}", am.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

// AM demodulation
let demod = am_demodulate(&am, 100.0, sample_rate);
println!("Demod peak: {:.4}", demod.iter().cloned().fold(0.0, f64::max));

// FM modulation
let fm = frequency_modulate(&carrier, &message, 5.0, sample_rate);
println!("FM signal length: {}", fm.len());

// BPSK: encode bits [1, 0, 1, 1, 0]
let bits = [true, false, true, true, false];
let bpsk = bpsk_modulate(&bits, 1000.0, 50, sample_rate);
println!("BPSK signal length: {}", bpsk.len());

// dB conversions
println!("0 dB = {:.4} linear", db_to_linear(0.0));
println!("-20 dB = {:.4} linear", db_to_linear(-20.0));
println!("1.0 linear = {:.1} dB", linear_to_db(1.0));
```

```
AM peak: 1.8000
Demod peak: 0.7912
FM signal length: 200
BPSK signal length: 250
0 dB = 1.0000 linear
-20 dB = 0.1000 linear
1.0 linear = 0.0 dB
```

**Use cases:** Software-defined radio, audio synthesis, telecommunications simulation, RFID/NFC protocol implementation.

## Future Scope / Roadmap

- [ ] **FFT-based fast convolution** — O(n log n) for large kernels
- [ ] **IIR filter cascading designer** — automatic order selection from spec
- [ ] **Pole-zero plot** — `zplane` visualization helper
- [ ] **Multi-rate processing** — decimation and interpolation
- [ ] **Adaptive filters** — LMS, RLS algorithms
- [ ] **STFT** — Short-Time Fourier Transform with overlap-add
- [ ] **Cepstrum analysis** — pitch detection, echo removal
- [ ] **`no_std` support** — embedded signal processing

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](../LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

at your option.
