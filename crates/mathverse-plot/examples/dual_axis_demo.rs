//! Dual y-axis and broken axes example.
//!
//! Demonstrates how to use dual y-axes and broken axes for plots.

use mathverse_plot::dual_axis::{AxisBreak, AxisBuilder, BrokenAxis, BreakStyle, DualYAxis};
use mathverse_plot::axis_config::AxisConfig;
use mathverse_plot::style::Color;

fn main() {
    // Create a dual y-axis configuration
    let dual = DualYAxis::new("Temperature (°C)", "Precipitation (mm)")
        .with_primary(AxisConfig::new().with_label("Temperature (°C)"))
        .with_secondary(AxisConfig::new().with_label("Precipitation (mm)"))
        .with_separator(Color::rgb(200, 200, 200));

    println!("Dual Y-Axis configuration:");
    println!("  Primary label: {}", dual.primary.label);
    println!("  Secondary label: {}", dual.secondary.label);

    // Map values between axes
    let temp_config = AxisConfig::new().with_limits(0.0, 40.0);
    let precip_config = AxisConfig::new().with_limits(0.0, 200.0);
    let dual_with_limits = DualYAxis::new("Temp", "Precip")
        .with_primary(temp_config)
        .with_secondary(precip_config);

    let temp_20 = 20.0;
    let mapped_precip = dual_with_limits.map_to_secondary(temp_20);
    println!("\nMapping temperature to precipitation:");
    println!("  {}°C -> {:.1} mm", temp_20, mapped_precip);

    // Create a broken axis
    let broken = AxisBuilder::broken_y(10.0, 20.0);

    println!("\nBroken axis:");
    println!("  Break range: {} - {}", broken.breaks[0].start, broken.breaks[0].end);
    println!("  Gap size: {} px", broken.breaks[0].gap_px);
    println!("  Style: {:?}", broken.breaks[0].style);

    // Test if values are in break
    let test_values = vec![5.0, 15.0, 25.0];
    for val in &test_values {
        println!("  Value {}: {} in break", val, if broken.in_break(*val) { "is" } else { "is not" });
    }

    // Render break marks
    let break_svg = broken.render_breaks_svg(100.0, 50.0, 150.0);
    println!("\nBreak SVG ({} bytes):", break_svg.len());

    // Create a more complex broken axis with multiple breaks
    let multi_break = AxisBuilder::broken_x(10.0, 20.0)
        .with_break(AxisBreak::new(50.0, 60.0).with_gap_px(20.0));

    println!("\nMultiple breaks:");
    println!("  Number of breaks: {}", multi_break.breaks.len());
    println!("  Total gap: {} px", multi_break.total_gap_px());

    // Test different break styles
    let styles = vec![
        BreakStyle::Zigzag,
        BreakStyle::ParallelSlashes,
        BreakStyle::Gap,
        BreakStyle::SquareBracket,
    ];

    println!("\nBreak styles:");
    for style in &styles {
        let break_ = AxisBreak::new(0.0, 1.0).with_style(*style);
        let svg = break_.render_svg(50.0, 0.0, 100.0);
        println!("  {:?}: {} bytes", style, svg.len());
    }

    // Use preset builders
    let temp_precip = AxisBuilder::temperature_precipitation();
    println!("\nPreset: Temperature-Precipitation");
    println!("  Primary: {}", temp_precip.primary.label);
    println!("  Secondary: {}", temp_precip.secondary.label);

    let count_pct = AxisBuilder::count_percentage();
    println!("\nPreset: Count-Percentage");
    println!("  Primary: {}", count_pct.primary.label);
    println!("  Secondary: {}", count_pct.secondary.label);

    println!("\nDual axis and broken axes example complete!");
}
