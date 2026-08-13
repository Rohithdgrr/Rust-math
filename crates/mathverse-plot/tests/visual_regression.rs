//! Visual regression tests + lock-in tests for the P0 correctness fixes.
//!
//! Canonical SVG outputs are hashed with FNV-1a (stable, no external deps).
//! Any rendering change that alters the SVG breaks these hashes on purpose;
//! update them only when the change is intentional and reviewed.

use mathverse_plot::boxplot::BoxStats;
use mathverse_plot::common::{DataPoint, DataSeries, PlotConfig};
use mathverse_plot::style::Color;
use mathverse_plot::svg::SvgPlot;

/// FNV-1a 64-bit — deterministic across platforms and Rust versions.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn line_plot_svg(title: &str, x_label: &str, y_label: &str, name: &str) -> String {
    let mut cfg = PlotConfig::new()
        .with_title(title.to_string())
        .with_x_label(x_label.to_string())
        .with_y_label(y_label.to_string())
        .with_dimensions(320, 240)
        .with_tick_count(4);
    cfg.show_legend = true;
    let mut sp = SvgPlot::new(cfg);
    let pts: Vec<DataPoint> = (0..=20)
        .map(|i| {
            let x = i as f64 / 10.0;
            DataPoint::new(x, (x * std::f64::consts::PI).sin())
        })
        .collect();
    sp.add_series(DataSeries::new(name.to_string(), pts));
    sp.generate()
}

// ---------------------------------------------------------------------------
// P0 lock-in
// ---------------------------------------------------------------------------

#[test]
fn xml_injection_is_escaped() {
    let svg = line_plot_svg(
        "<script>alert(1)</script> & title",
        "x < 1 && x > 0",
        "y \"quoted\" 'single'",
        "s<&>s",
    );
    // Raw script tags / attribute-openers must never appear verbatim.
    assert!(!svg.contains("<script>"), "raw <script> leaked into SVG");
    assert!(!svg.contains("<script"), "raw script tag leaked into SVG");
    // The escaped forms must be present.
    assert!(svg.contains("&lt;script&gt;"), "title not escaped");
    assert!(svg.contains("&amp;"), "ampersand not escaped");
    assert!(svg.contains("&quot;"), "double quote not escaped");
    assert!(svg.contains("&apos;"), "single quote not escaped");
    assert!(svg.contains("&lt;"), "less-than not escaped");
}

#[test]
fn named_color_resolves_to_hex() {
    assert_eq!(Color::RED.to_hex(), "#ff0000");
    assert_eq!(Color::GREEN.to_hex(), "#008000");
    assert_eq!(Color::BLUE.to_hex(), "#0000ff");
    assert_eq!(Color::BLACK.to_hex(), "#000000");
    assert_eq!(Color::WHITE.to_hex(), "#ffffff");
    // Unknown names fall back to a deterministic valid hex (never a bare name).
    let unknown = Color::Named("chartreuse");
    let hex = unknown.to_hex();
    assert!(hex.starts_with('#'), "unknown name leaked: {hex}");
    assert_eq!(hex.len(), 7, "unknown name hex malformed: {hex}");
    // Round-trip through rgb.
    let (r, g, b) = unknown.to_rgb();
    assert_eq!(Color::rgb(r, g, b).to_hex(), hex);
}

#[test]
fn quartiles_match_reference() {
    // Method-2 (linear interpolation) reference values for [1..=9].
    let stats = BoxStats::compute(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    assert_eq!(stats.q1, 3.0);
    assert_eq!(stats.median, 5.0);
    assert_eq!(stats.q3, 7.0);
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 9.0);
    assert!(stats.outliers.is_empty());

    // Even-length sample: [1..=8] via linear interpolation (numpy default)
    // -> q1=2.75, median=4.5, q3=6.25.
    let stats = BoxStats::compute(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(stats.q1, 2.75);
    assert_eq!(stats.median, 4.5);
    assert_eq!(stats.q3, 6.25);
}

// ---------------------------------------------------------------------------
// Visual regression hashes
// ---------------------------------------------------------------------------

#[test]
fn line_plot_regression() {
    let svg = line_plot_svg("regression", "x", "y", "series");
    let hash = fnv1a(&svg);
    assert_eq!(
        hash,
        0xf7e4d60fb3fa1002,
        "line plot SVG changed (hash {hash:#x}) — update after intentional change"
    );
}

#[test]
fn styled_boxplot_regression() {
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 50.0],
        vec![3.0, 4.0, 5.0, 6.0, 7.0],
    ];
    let labels = vec!["A".to_string(), "B".to_string()];
    let cfg = mathverse_plot::BoxPlotConfig::new().with_notch(true);
    let svg = mathverse_plot::render_styled_boxplot(&data, &labels, &cfg).unwrap();
    let hash = fnv1a(&svg);
    assert_eq!(
        hash,
        0x16a002b0b6c63555,
        "styled boxplot SVG changed (hash {hash:#x}) — update after intentional change"
    );
}

#[test]
fn contourf_regression() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..12 {
        for j in 0..12 {
            xs.push(i as f64 / 11.0);
            ys.push(j as f64 / 11.0);
        }
    }
    let zs: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| ((x * 4.0).sin() * (y * 4.0).cos()))
        .collect();
    let grid: Vec<Vec<f64>> = zs.chunks(12).map(|row| row.to_vec()).collect();
    let cfg = mathverse_plot::ContourConfig::new();
    let svg = mathverse_plot::contour::render_contour(&grid, (0.0, 1.0), (0.0, 1.0), &cfg).unwrap();
    let hash = fnv1a(&svg);
    assert_eq!(
        hash,
        0x272543f2ea048bcb,
        "contour SVG changed (hash {hash:#x}) — update after intentional change"
    );
}
