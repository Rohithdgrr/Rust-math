//! CVD (Color Vision Deficiency) simulation example.
//!
//! Demonstrates how to simulate color blindness on plot colors.

use mathverse_plot::cvd::{simulate_cvd, simulate_palette, CvdType, cvd_comparison_svg};
use mathverse_plot::style::Color;

fn main() {
    // Create a sample palette
    let palette = vec![
        Color::RED,
        Color::GREEN,
        Color::BLUE,
        Color::YELLOW,
        Color::rgb(255, 128, 0),  // Orange
        Color::rgb(128, 0, 255),  // Purple
    ];

    println!("Original palette:");
    for (i, color) in palette.iter().enumerate() {
        println!("  {}: {}", i + 1, color.to_hex());
    }

    // Simulate for different CVD types
    let cvd_types = vec![
        CvdType::Protanopia,
        CvdType::Deuteranopia,
        CvdType::Tritanopia,
    ];

    for cvd_type in &cvd_types {
        println!("\n{} simulation:", cvd_type.name());
        let simulated = simulate_palette(&palette, *cvd_type);
        for (i, color) in simulated.iter().enumerate() {
            println!("  {}: {} -> {}", i + 1, palette[i].to_hex(), color.to_hex());
        }
    }

    // Generate comparison SVG
    let svg = cvd_comparison_svg(&palette, CvdType::Deuteranopia, 600, 200);
    println!("\nGenerated CVD comparison SVG: {} bytes", svg.len());
    println!("SVG preview (first 200 chars):");
    println!("{}", &svg[..200.min(svg.len())]);

    // Simulate a single color
    let original = Color::rgb(255, 0, 0);
    let simulated = simulate_cvd(original, CvdType::Protanopia);
    println!("\nSingle color simulation:");
    println!("  Original: {}", original.to_hex());
    println!("  Protanopia: {}", simulated.to_hex());

    println!("\nCVD simulation example complete!");
}
