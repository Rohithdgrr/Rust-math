//! # mathverse-transforms
//!
//! Signal transforms for the MathVerse ecosystem.
//!
//! Provides:
//! - **FFT**: radix-2 Cooley-Tukey, power-of-2 forward/inverse
//! - **DCT**: Discrete Cosine Transform (Type II) and inverse
//! - **DST**: Discrete Sine Transform (Type I) and inverse
//! - **Wavelet**: Haar wavelet transform and inverse
//! - **Goertzel**: single-frequency tone detection
//! - **Hough**: line detection via Hough transform
//! - **Radon**: Radon transform for tomographic reconstruction
//!
//! All transforms operate on `Vec<f64>` and return `Vec<f64>`.
//!
//! # Conventions
//!
//! - FFT/DFT use the forward sign `exp(-2πi·k·n/N)` and are **un-normalized**;
//!   the inverse [`fft::ifft`] applies the `1/N` scale.
//! - DCT/DST variants are **orthonormalized** (scaled by `√(1/N)` / `√(2/N)`
//!   per coefficient), so forward-then-inverse is an exact round trip.
//! - The Haar wavelet is orthonormal (factors of `1/√2`), preserving energy
//!   across the transform.

pub mod fft;
pub mod dct;
pub mod dst;
pub mod wavelet;
pub mod goertzel;
pub mod hough;
pub mod radon;

pub use fft::*;
pub use dct::*;
pub use dst::*;
pub use wavelet::*;
pub use goertzel::*;
pub use hough::*;
pub use radon::*;
