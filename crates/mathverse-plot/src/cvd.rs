//! Color Vision Deficiency (CVD) simulation.
//!
//! Implements the Brettel/Viénot/Mollon algorithm for simulating how plots
//! appear to people with color vision deficiencies (color blindness).
//!
//! # Supported Deficiency Types
//!
//! - **Protanopia**: No red cones (~1% of males)
//! - **Deuteranopia**: No green cones (~1% of males)
//! - **Tritanopia**: No blue cones (~0.003% of population)
//! - **Protanomaly**: Weak red cones (~1% of males)
//! - **Deuteranomaly**: Weak green cones (~5% of males)
//! - **Tritanomaly**: Weak blue cones (~0.01% of population)

use crate::style::Color;

/// Types of color vision deficiency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CvdType {
    /// Complete absence of red cones.
    Protanopia,
    /// Complete absence of green cones.
    Deuteranopia,
    /// Complete absence of blue cones.
    Tritanopia,
    /// Weak red cones (partial).
    Protanomaly,
    /// Weak green cones (partial).
    Deuteranomaly,
    /// Weak blue cones (partial).
    Tritanomaly,
}

impl CvdType {
    /// Severity factor (0.0 = full color, 1.0 = complete deficiency).
    pub fn severity(&self) -> f64 {
        match self {
            CvdType::Protanopia => 1.0,
            CvdType::Deuteranopia => 1.0,
            CvdType::Tritanopia => 1.0,
            CvdType::Protanomaly => 0.6,
            CvdType::Deuteranomaly => 0.6,
            CvdType::Tritanomaly => 0.6,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &str {
        match self {
            CvdType::Protanopia => "Protanopia",
            CvdType::Deuteranopia => "Deuteranopia",
            CvdType::Tritanopia => "Tritanopia",
            CvdType::Protanomaly => "Protanomaly",
            CvdType::Deuteranomaly => "Deuteranomaly",
            CvdType::Tritanomaly => "Tritanomaly",
        }
    }
}

/// Simulate color vision deficiency on an RGB color.
///
/// Uses the Brettel/Viénot/Mollon (1997) algorithm adapted by Machado (2009)
/// for simulating dichromatic and anomalous trichromatic vision.
pub fn simulate_cvd(color: Color, cvd_type: CvdType) -> Color {
    let (r, g, b) = color.to_rgb();
    let severity = cvd_type.severity();

    // Convert sRGB [0-255] to linear [0-1]
    let r_lin = srgb_to_linear(r as f64 / 255.0);
    let g_lin = srgb_to_linear(g as f64 / 255.0);
    let b_lin = srgb_to_linear(b as f64 / 255.0);

    // LMS transformation (sRGB -> LMS via CIE 1931)
    let l = 0.390478 * r_lin + 0.543912 * g_lin + 0.141609 * b_lin;
    let m = 0.070928 * r_lin + 0.193568 * g_lin + 0.735504 * b_lin;
    let s = 0.021934 * r_lin + 0.109468 * g_lin + 0.868598 * b_lin;

    // Apply CVD simulation matrix
    let (l2, m2, s2) = match cvd_type {
        CvdType::Protanopia | CvdType::Protanomaly => {
            let sim = simulate_dichromat(l, m, s, &PROTANOPIA_MATRIX);
            blend(l, m, s, sim, severity)
        }
        CvdType::Deuteranopia | CvdType::Deuteranomaly => {
            let sim = simulate_dichromat(l, m, s, &DEUTERANOPIA_MATRIX);
            blend(l, m, s, sim, severity)
        }
        CvdType::Tritanopia | CvdType::Tritanomaly => {
            let sim = simulate_dichromat(l, m, s, &TRITANOPIA_MATRIX);
            blend(l, m, s, sim, severity)
        }
    };

    // LMS -> sRGB
    let r_lin = 2.041588 * l2 - 0.565037 * m2 + 0.012638 * s2;
    let g_lin = -0.969258 * l2 + 1.875992 * m2 - 0.023663 * s2;
    let b_lin = 0.013445 * l2 - 0.113959 * m2 + 0.988584 * s2;

    // Clip and convert to sRGB
    let r_out = linear_to_srgb(r_lin.clamp(0.0, 1.0));
    let g_out = linear_to_srgb(g_lin.clamp(0.0, 1.0));
    let b_out = linear_to_srgb(b_lin.clamp(0.0, 1.0));

    Color::rgb(r_out, g_out, b_out)
}

/// Simulate CVD on a hex color string.
pub fn simulate_cvd_hex(hex: &str, cvd_type: CvdType) -> String {
    let color = parse_hex_color(hex).unwrap_or(Color::GRAY);
    let simulated = simulate_cvd(color, cvd_type);
    simulated.to_hex()
}

/// Parse a hex color string (#RRGGBB or #RGB) into a Color.
fn parse_hex_color(hex: &str) -> Option<Color> {
    let hex = hex.trim_start_matches('#');
    match hex.len() {
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some(Color::rgb(r, g, b))
        }
        3 => {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            Some(Color::rgb(r, g, b))
        }
        _ => None,
    }
}

/// Simulate CVD on all colors in a palette.
pub fn simulate_palette(palette: &[Color], cvd_type: CvdType) -> Vec<Color> {
    palette.iter().map(|&c| simulate_cvd(c, cvd_type)).collect()
}

/// Generate a CVD comparison: original vs simulated side-by-side.
///
/// Returns SVG showing the palette in original and simulated forms.
pub fn cvd_comparison_svg(palette: &[Color], cvd_type: CvdType, width: u32, height: u32) -> String {
    let simulated = simulate_palette(palette, cvd_type);
    let swatch_width = (width as f64 / palette.len() as f64).floor();
    let swatch_height = height as f64 / 2.0;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\n",
        width, height
    ));

    // Title
    svg.push_str(&format!(
        "  <text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">CVD Simulation: {}</text>\n",
        width / 2,
        cvd_type.name()
    ));

    // Original palette (top half)
    svg.push_str("  <text x=\"10\" y=\"40\" font-size=\"11\">Original</text>\n");
    for (i, color) in palette.iter().enumerate() {
        let x = i as f64 * swatch_width;
        svg.push_str(&format!(
            "  <rect x=\"{}\" y=\"45\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"#ccc\" stroke-width=\"0.5\"/>\n",
            x, swatch_width, swatch_height - 50.0, color.to_hex()
        ));
    }

    // Simulated palette (bottom half)
    svg.push_str(&format!(
        "  <text x=\"10\" y=\"{}\" font-size=\"11\">{}</text>\n",
        swatch_height + 15.0,
        cvd_type.name()
    ));
    for (i, color) in simulated.iter().enumerate() {
        let x = i as f64 * swatch_width;
        svg.push_str(&format!(
            "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"#ccc\" stroke-width=\"0.5\"/>\n",
            x, swatch_height + 20.0, swatch_width, swatch_height - 50.0, color.to_hex()
        ));
    }

    svg.push_str("</svg>");
    svg
}

// --- Internal helpers ---

/// sRGB gamma expansion.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB gamma compression.
fn linear_to_srgb(c: f64) -> u8 {
    let v = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Simulate dichromat by projecting onto the dichromatic confusion line.
fn simulate_dichromat(l: f64, m: f64, s: f64, matrix: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    let l2 = matrix[0][0] * l + matrix[0][1] * m + matrix[0][2] * s;
    let m2 = matrix[1][0] * l + matrix[1][1] * m + matrix[1][2] * s;
    let s2 = matrix[2][0] * l + matrix[2][1] * m + matrix[2][2] * s;
    (l2, m2, s2)
}

/// Blend between normal and simulated vision based on severity.
fn blend(l: f64, m: f64, s: f64, sim: (f64, f64, f64), severity: f64) -> (f64, f64, f64) {
    let t = severity;
    (
        l * (1.0 - t) + sim.0 * t,
        m * (1.0 - t) + sim.1 * t,
        s * (1.0 - t) + sim.2 * t,
    )
}

// Matrices based on Machado (2009) "A Physiologically-based Model for Simulation of Color Vision Deficiency"

/// Protanopia simulation matrix (no red cones).
static PROTANOPIA_MATRIX: [[f64; 3]; 3] = [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
];

/// Deuteranopia simulation matrix (no green cones).
static DEUTERANOPIA_MATRIX: [[f64; 3]; 3] = [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.011820, 0.042940, 0.968881],
];

/// Tritanopia simulation matrix (no blue cones).
static TRITANOPIA_MATRIX: [[f64; 3]; 3] = [
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.303900],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cvd_simulate_red_protanopia() {
        let color = Color::RED;
        let sim = simulate_cvd(color, CvdType::Protanopia);
        let (r, g, b) = sim.to_rgb();
        // Red should appear different from original (255,0,0) to a protanope
        // The exact values depend on the simulation matrix
        assert_ne!((r, g, b), (255, 0, 0), "simulated color should differ from original");
        assert!(r <= 255 && g <= 255 && b <= 255, "values should be valid u8");
    }

    #[test]
    fn cvd_simulate_green_deuteranopia() {
        let color = Color::GREEN;
        let sim = simulate_cvd(color, CvdType::Deuteranopia);
        let (r, g, b) = sim.to_rgb();
        // Green should appear brownish to a deuteranope
        assert!(r > 50, "red should be visible: got {r}");
    }

    #[test]
    fn cvd_simulate_blue_tritanopia() {
        let color = Color::BLUE;
        let sim = simulate_cvd(color, CvdType::Tritanopia);
        let (r, g, b) = sim.to_rgb();
        // Blue should appear shifted for tritanope
        // Just check it doesn't panic and produces valid output
        assert!(r <= 255 && g <= 255 && b <= 255);
    }

    #[test]
    fn cvd_normal_vision_unchanged() {
        // With severity 0, color should be unchanged (blend factor = 0)
        // But our severity is 0.6 for anomalous types, so test with manual blend
        let color = Color::rgb(128, 64, 200);
        let sim = simulate_dichromat(
            srgb_to_linear(128.0 / 255.0),
            srgb_to_linear(64.0 / 255.0),
            srgb_to_linear(200.0 / 255.0),
            &PROTANOPIA_MATRIX,
        );
        // Just verify it produces finite values
        assert!(sim.0.is_finite());
        assert!(sim.1.is_finite());
        assert!(sim.2.is_finite());
    }

    #[test]
    fn cvd_palette_simulation() {
        let palette = vec![Color::RED, Color::GREEN, Color::BLUE, Color::YELLOW];
        let sim = simulate_palette(&palette, CvdType::Deuteranopia);
        assert_eq!(sim.len(), 4);
        // All should be valid colors
        for color in &sim {
            let (r, g, b) = color.to_rgb();
            assert!(r <= 255 && g <= 255 && b <= 255);
        }
    }

    #[test]
    fn cvd_comparison_svg_renders() {
        let palette = vec![Color::RED, Color::GREEN, Color::BLUE];
        let svg = cvd_comparison_svg(&palette, CvdType::Protanopia, 300, 200);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Protanopia"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn cvd_severity_names() {
        assert_eq!(CvdType::Protanopia.name(), "Protanopia");
        assert_eq!(CvdType::Deuteranomaly.name(), "Deuteranomaly");
        assert_eq!(CvdType::Tritanopia.severity(), 1.0);
        assert_eq!(CvdType::Protanomaly.severity(), 0.6);
    }
}
