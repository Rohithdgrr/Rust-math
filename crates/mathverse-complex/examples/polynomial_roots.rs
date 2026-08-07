//! Find all complex roots of polynomials with the Durand–Kerner method.
//!
//! Run with: `cargo run -p mathverse-complex --example polynomial_roots`

use mathverse_complex::{eval_polynomial, polynomial_roots, Complex};

/// Coefficients are lowest-order first: c[0] + c[1]·z + c[2]·z² + …
fn roots_of(name: &str, coeffs: &[Complex]) {
    let roots = polynomial_roots(coeffs, 1000, 1e-12);
    println!("{name}:");
    for (i, r) in roots.iter().enumerate() {
        // Verify by re-evaluating the polynomial at the root.
        let residual = eval_polynomial(coeffs, *r).norm();
        println!("  root {i}: {r:.6}  (|p(root)| = {residual:.2e})");
    }
}

fn main() {
    // z² + 1 = 0  →  ±i
    roots_of("z² + 1", &[Complex::one(), Complex::zero(), Complex::one()]);

    // z² - 2z + 2 = 0  →  1 ± i
    roots_of(
        "z² - 2z + 2",
        &[Complex::real(2.0), Complex::real(-2.0), Complex::one()],
    );

    // z³ - 1 = 0  →  1, -1/2 ± i·√3/2
    roots_of(
        "z³ - 1",
        &[
            Complex::real(-1.0),
            Complex::zero(),
            Complex::zero(),
            Complex::one(),
        ],
    );
}
