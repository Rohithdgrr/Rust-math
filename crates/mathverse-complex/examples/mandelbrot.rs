//! Render the Mandelbrot set to the terminal using the crate's iteration and
//! smooth-coloring helpers.
//!
//! Run with: `cargo run -p mathverse-complex --example mandelbrot`
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::needless_range_loop
)]

use mathverse_complex::{mandelbrot_iterate, Complex};

const WIDTH: usize = 80;
const HEIGHT: usize = 24;
const MAX_ITER: usize = 100;

const RAMP: &[u8] = b" .:-=+*#%@";

fn main() {
    for row in 0..HEIGHT {
        let mut line = String::new();
        for col in 0..WIDTH {
            // Map pixel to the region [-2.1, 0.7] x [-1.2, 1.2]
            let re = -2.1 + col as f64 * 2.8 / WIDTH as f64;
            let im = -1.2 + row as f64 * 2.4 / HEIGHT as f64;
            let c = Complex::new(re, im);
            let iters = mandelbrot_iterate(c, MAX_ITER, 2.0);
            let idx = if iters == MAX_ITER {
                RAMP.len() - 1 // inside the set
            } else {
                iters * (RAMP.len() - 1) / MAX_ITER
            };
            line.push(RAMP[idx] as char);
        }
        println!("{line}");
    }
}
