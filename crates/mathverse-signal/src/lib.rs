//! # mathverse-signal
//!
//! Signal processing for the MathVerse ecosystem.
//!
//! Provides:
//! - **Convolution**: 1D convolution, correlation
//! - **Filter design**: Butterworth low-pass/high-pass/band-pass filters
//! - **IIR filters**: second-order sections (biquad) implementation
//! - **Windowing**: rectangular, Hanning, Hamming, Blackman, Blackman-Harris, Kaiser
//! - **Spectrum**: power spectral density, frequency response
//! - **Detection**: peak finding, threshold detection
//! - **Modulation**: AM, FM, PM modulation and demodulation
//!
//! All signals are represented as `&[f64]` slices.

pub mod convolution;
pub mod filter_design;
pub mod iir;
pub mod windowing;
pub mod spectrum;
pub mod detection;
pub mod modulation;

pub use convolution::*;
pub use filter_design::*;
pub use iir::*;
pub use windowing::*;
pub use spectrum::*;
pub use detection::*;
pub use modulation::*;
