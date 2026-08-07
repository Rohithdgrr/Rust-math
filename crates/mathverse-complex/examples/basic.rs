//! A quick tour of the `Complex` core type.
//!
//! Run with: `cargo run -p mathverse-complex --example basic`

use mathverse_complex::{Complex, C32};

fn main() {
    // Construction
    let z = Complex::new(3.0, 4.0);
    let w = Complex::polar(2.0, std::f64::consts::FRAC_PI_2); // 0 + 2i
    println!("z = {z}, |z| = {}, arg(z) = {}", z.norm(), z.arg());
    println!("w = {w}, conj(w) = {}", w.conjugate());

    // Arithmetic
    let sum = z + w;
    let prod = z * w;
    let quot = z / w;
    println!("z + w = {sum}");
    println!("z * w = {prod}");
    println!("z / w = {quot}");

    // Transcendental round-trip: exp(ln z) == z
    let roundtrip = z.ln().exp();
    println!("exp(ln z) = {roundtrip}  (should be ≈ {z})");
    assert!((roundtrip - z).norm() < 1e-12);

    // Roots and powers
    let s = z.sqrt();
    println!("sqrt(z) = {s}, and s² = {}", s * s);
    println!("z^3 = {}", z.pow(Complex::real(3.0)));

    // f32 works with the same code via the C32 alias
    let z32: C32 = Complex::new(1.0f32, -1.0);
    println!("f32: |{z32}| = {}, arg = {} rad", z32.norm(), z32.arg());

    // numpy/cmath parity names
    println!("phase(z) = {} (same as arg)", z.phase());
    let (r, theta) = z.to_polar();
    println!(
        "polar(z) = ({r}, {theta}); rect back = {}",
        Complex::rect(r, theta)
    );
    assert!(z.is_close(&Complex::rect(r, theta), 1e-12, 1e-12));
}
