//! Fast Fourier Transform: spectral peak detection + round-trip identity.
//!
//! Run with: `cargo run -p mathverse-complex --example fft`

use mathverse_complex::{fft, ifft, Complex};

fn main() {
    let n = 1024usize;
    let freq = 8.0; // cycles per buffer

    // A real sine wave at `freq` cycles over the buffer.
    let signal: Vec<Complex> = (0..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            Complex::new((2.0 * std::f64::consts::PI * freq * t).sin(), 0.0)
        })
        .collect();

    let spectrum = fft(&signal);

    // Find the peak bin. A real sine at frequency f has symmetric peaks at
    // bins f and n - f.
    let peak = (0..n / 2)
        .max_by(|&a, &b| spectrum[a].norm().total_cmp(&spectrum[b].norm()))
        .unwrap();
    println!(
        "peak bin = {peak}  (expected ≈ {freq}), magnitude = {:.3}",
        spectrum[peak].norm()
    );

    // Round-trip: ifft(fft(x)) ≈ x
    let back = ifft(&spectrum);
    let max_err = (0..n)
        .map(|k| (back[k] - signal[k]).norm())
        .fold(0.0f64, f64::max);
    println!("max |ifft(fft(x)) - x| = {max_err:.3e}");
    assert!(max_err < 1e-10);
}
