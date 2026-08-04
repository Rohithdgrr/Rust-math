//! Statistical annotations example.
//!
//! Demonstrates how to add significance brackets and p-values to plots.

use mathverse_plot::stats_annotate::{StatAnnotation, StatAnnotations, StatTest};

fn main() {
    // Create a simple bracket annotation
    let bracket = StatAnnotation::bracket(
        "Control", 0.0,
        "Treatment", 1.0,
        1.5,
        StatTest::TTest { p_value: 0.003 },
    );

    println!("Statistical annotation:");
    println!("  Left: {} at x={}", bracket.left_label, bracket.left_x);
    println!("  Right: {} at x={}", bracket.right_label, bracket.right_x);
    println!("  Test: {}", bracket.test.name());
    println!("  P-value: {}", bracket.test.format_p());
    println!("  Significance: {}", bracket.test.stars());
    println!("  Display text: {}", bracket.display_text());

    // Render the bracket as SVG
    let svg = bracket.render_svg(0.0, 1.0);
    println!("\nSVG output:");
    println!("{}", svg);

    // Create a collection of annotations
    let annotations = StatAnnotations::new()
        .with_alpha(0.05)
        .bracket("A", 0.0, "B", 1.0, 2.0, StatTest::TTest { p_value: 0.01 })
        .bracket("B", 1.0, "C", 2.0, 2.5, StatTest::MannWhitney { p_value: 0.04 })
        .bracket("A", 0.0, "C", 2.0, 3.0, StatTest::TTest { p_value: 0.1 });

    println!("\nAnnotation collection:");
    println!("  Total annotations: {}", annotations.len());
    println!("  Significant (p < 0.05): {}", annotations.significant_only().len());

    // Render all annotations
    let all_svg = annotations.render_svg(0.0, 1.0);
    println!("\nAll annotations SVG ({} bytes):", all_svg.len());

    // Test different statistical tests
    let tests = vec![
        StatTest::TTest { p_value: 0.001 },
        StatTest::PairedTTest { p_value: 0.03 },
        StatTest::MannWhitney { p_value: 0.08 },
        StatTest::Anova { f_statistic: 5.2, p_value: 0.005 },
        StatTest::ChiSquared { p_value: 0.0001 },
    ];

    println!("\nStatistical tests:");
    for test in &tests {
        println!("  {}: p={}, stars={}", test.name(), test.format_p(), test.stars());
    }

    println!("\nStatistical annotations example complete!");
}
