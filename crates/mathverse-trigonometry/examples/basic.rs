//! End-to-end tour of `mathverse-trigonometry`.
//!
//! Run with `cargo run --example basic`.

#![allow(clippy::pedantic, clippy::nursery)]

use std::f64::consts::PI;

use mathverse_trigonometry::{
    angle_difference, cos, cosd, cospi, haversine_distance_deg, sin, sin_cos_deg, sin_double, sind,
    sinpi, sum_sin, tan_half, unwrap_angles,
};

fn main() {
    // 1. Core circular functions (radians) and their degree variants.
    let (s, c) = (sin(0.5), cos(0.5));
    println!("sin(0.5) = {s:.4}, cos(0.5) = {c:.4}");
    println!("sin_deg(30) = {}, cos_deg(60) = {}", sind(30.0), cosd(60.0));
    println!(
        "sinpi(0.5) = {} (exact), cospi(1.0) = {}",
        sinpi(0.5),
        cospi(1.0)
    );
    println!("sin_cos_deg(90) = {:?}", sin_cos_deg(90.0));

    // 2. Identities: double-angle sanity check.
    let x = 1.2;
    println!(
        "sin(2x) = {:.6}, 2 sin x cos x = {:.6}",
        sin_double(x),
        2.0 * sin(x) * cos(x)
    );

    // 3. Spherical distance (latitude/longitude in degrees).
    let d = haversine_distance_deg(52.5, 13.4, 41.9, 12.5, 6_371_000.0);
    println!("Berlin -> Rome approx {d:.0} m");

    // 4. Batched sine sum (DSP-ish).
    let xs = [0.1f64, 0.2, 0.3, 0.4, 0.5];
    println!("sum(sin xs) = {:.6}", sum_sin(&xs));

    // 5. Angle utilities.
    let mut phases = [-PI, -0.5, 0.5, 3.0];
    unwrap_angles(&mut phases);
    println!("unwrap_angles -> {phases:?}");
    println!(
        "angle_difference(2π, 0) = {:.3}, tan_half(1.0) = {:.4}",
        angle_difference(PI * 2.0, 0.0),
        tan_half(1.0)
    );
}
