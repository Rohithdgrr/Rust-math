//! Basic quantity usage example.
//!
//! This example demonstrates creating and working with quantities.

use mathverse_units::{Quantity, dimensions::LengthDim, si::Meter};

fn main() {
    // Create a length quantity
    let length: Quantity<LengthDim, Meter> = Quantity::new(5.0);
    println!("Length: {} m", length.value());

    // Arithmetic operations
    let q1: Quantity<LengthDim, Meter> = Quantity::new(5.0);
    let q2: Quantity<LengthDim, Meter> = Quantity::new(3.0);

    let sum = q1 + q2;
    println!("Sum: {} m", sum.value());

    let diff = q1 - q2;
    println!("Difference: {} m", diff.value());

    let scaled = q1 * 2.0;
    println!("Scaled: {} m", scaled.value());

    let divided = q1 / 2.0;
    println!("Divided: {} m", divided.value());
}
