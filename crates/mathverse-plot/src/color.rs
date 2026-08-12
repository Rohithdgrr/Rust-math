//! Color spaces, colormaps, and value normalization for heatmaps.

use crate::style::Color;

/// Matplotlib Viridis stops (sRGB), sampled at the same positions as
/// matplotlib's reference table.
const VIRIDIS: [(f64, (u8, u8, u8)); 5] = [
    (0.00, (0x44, 0x01, 0x54)),
    (0.25, (0x3b, 0x52, 0x8b)),
    (0.50, (0x21, 0x91, 0x8c)),
    (0.75, (0x5e, 0xc9, 0x62)),
    (1.00, (0xfd, 0xe7, 0x25)),
];

/// Matplotlib Plasma stops (sRGB).
const PLASMA: [(f64, (u8, u8, u8)); 6] = [
    (0.00, (0x0d, 0x08, 0x87)),
    (0.20, (0x6a, 0x00, 0xa8)),
    (0.40, (0xb1, 0x2a, 0x90)),
    (0.60, (0xe1, 0x64, 0x62)),
    (0.80, (0xfc, 0xa6, 0x36)),
    (1.00, (0xf0, 0xf9, 0x21)),
];

/// Matplotlib Inferno stops (sRGB).
const INFERNO: [(f64, (u8, u8, u8)); 6] = [
    (0.00, (0x00, 0x00, 0x04)),
    (0.20, (0x42, 0x0a, 0x68)),
    (0.40, (0x93, 0x26, 0x67)),
    (0.60, (0xcc, 0x54, 0x39)),
    (0.80, (0xf8, 0x9a, 0x1b)),
    (1.00, (0xfc, 0xff, 0xa4)),
];

/// Matplotlib Magma stops (sRGB).
const MAGMA: [(f64, (u8, u8, u8)); 6] = [
    (0.00, (0x00, 0x00, 0x04)),
    (0.20, (0x2c, 0x10, 0x5a)),
    (0.40, (0x71, 0x1f, 0x7e)),
    (0.60, (0xb6, 0x3f, 0x73)),
    (0.80, (0xef, 0x83, 0x43)),
    (1.00, (0xfc, 0xff, 0xa4)),
];

/// Matplotlib Cividis stops (sRGB, colorblind-friendly).
const CIVIDIS: [(f64, (u8, u8, u8)); 6] = [
    (0.00, (0x00, 0x20, 0x4d)),
    (0.20, (0x00, 0x4c, 0x6d)),
    (0.40, (0x36, 0x74, 0x7b)),
    (0.60, (0x7a, 0x9c, 0x6e)),
    (0.80, (0xc2, 0xc2, 0x3e)),
    (1.00, (0xfd, 0xe9, 0x4f)),
];

/// Sample a colormap at position `t in [0, 1]`, interpolating linearly
/// between stops in sRGB. Values outside `[0, 1]` clamp to the endpoints.
#[must_use]
pub fn color_map(t: f64, stops: &[(f64, (u8, u8, u8))]) -> Color {
    debug_assert!(stops.len() >= 2);
    let t = t.clamp(0.0, 1.0);
    let i = stops
        .partition_point(|(s, _)| *s < t)
        .saturating_sub(1)
        .min(stops.len() - 2);
    let (t0, c0) = stops[i];
    let (t1, c1) = stops[i + 1];
    let f = if t1 == t0 { 0.0 } else { (t - t0) / (t1 - t0) };
    let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * f).round() as u8;
    Color::rgb(lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2))
}

/// Generate a high-resolution colormap with 256 stops.
/// This produces smoother gradients for heatmaps and colorbars.
#[must_use]
pub fn generate_colormap_256(stops: &[(f64, (u8, u8, u8))]) -> Vec<Color> {
    (0..256)
        .map(|i| {
            let t = i as f64 / 255.0;
            color_map(t, stops)
        })
        .collect()
}

/// Viridis colormap (perceptually uniform, matplotlib-compatible).
#[must_use]
pub fn viridis(t: f64) -> Color {
    color_map(t, &VIRIDIS)
}

/// Viridis colormap with 256 stops for smooth gradients.
#[must_use]
pub fn viridis_256() -> Vec<Color> {
    generate_colormap_256(&VIRIDIS)
}

/// Plasma colormap (matplotlib-compatible).
#[must_use]
pub fn plasma(t: f64) -> Color {
    color_map(t, &PLASMA)
}

/// Plasma colormap with 256 stops.
#[must_use]
pub fn plasma_256() -> Vec<Color> {
    generate_colormap_256(&PLASMA)
}

/// Inferno colormap (matplotlib-compatible).
#[must_use]
pub fn inferno(t: f64) -> Color {
    color_map(t, &INFERNO)
}

/// Inferno colormap with 256 stops.
#[must_use]
pub fn inferno_256() -> Vec<Color> {
    generate_colormap_256(&INFERNO)
}

/// Magma colormap (matplotlib-compatible).
#[must_use]
pub fn magma(t: f64) -> Color {
    color_map(t, &MAGMA)
}

/// Magma colormap with 256 stops.
#[must_use]
pub fn magma_256() -> Vec<Color> {
    generate_colormap_256(&MAGMA)
}

/// Cividis colormap (colorblind-friendly, matplotlib-compatible).
#[must_use]
pub fn cividis(t: f64) -> Color {
    color_map(t, &CIVIDIS)
}

/// Cividis colormap with 256 stops.
#[must_use]
pub fn cividis_256() -> Vec<Color> {
    generate_colormap_256(&CIVIDIS)
}

/// Get a colormap by name.
#[must_use]
pub fn colormap_by_name(name: &str) -> fn(f64) -> Color {
    match name.to_lowercase().as_str() {
        "viridis" => viridis,
        "plasma" => plasma,
        "inferno" => inferno,
        "magma" => magma,
        "cividis" => cividis,
        _ => viridis, // Default
    }
}

/// Get a high-resolution colormap by name.
#[must_use]
pub fn colormap_256_by_name(name: &str) -> Vec<Color> {
    match name.to_lowercase().as_str() {
        "viridis" => viridis_256(),
        "plasma" => plasma_256(),
        "inferno" => inferno_256(),
        "magma" => magma_256(),
        "cividis" => cividis_256(),
        _ => viridis_256(), // Default
    }
}

/// Value normalization for coloring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// `(v - min) / (max - min)`.
    #[default]
    Linear,
    /// Log-spaced; falls back to linear on non-positive data.
    Log,
    /// Empirical rank `index / (n - 1)` of each value.
    Quantile,
}

/// Map values into `[0, 1]` for colormap sampling. Degenerate input
/// (constant data) maps to the midpoint `0.5`.
#[must_use]
pub fn normalize(values: &[f64], method: Normalization) -> Vec<f64> {
    match method {
        Normalization::Linear => {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = max - min;
            if span <= 0.0 {
                vec![0.5; values.len()]
            } else {
                values.iter().map(|v| (v - min) / span).collect()
            }
        }
        Normalization::Log => {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            if min <= 0.0 {
                normalize(values, Normalization::Linear)
            } else {
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lo = min.ln();
                let span = max.ln() - lo;
                if span <= 0.0 {
                    vec![0.5; values.len()]
                } else {
                    values.iter().map(|v| (v.ln() - lo) / span).collect()
                }
            }
        }
        Normalization::Quantile => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let n = sorted.len();
            if n <= 1 {
                return vec![0.5; n];
            }
            values
                .iter()
                .map(|v| sorted.partition_point(|&x| x < *v) as f64 / (n - 1) as f64)
                .collect()
        }
    }
}

/// Color each value via `map` after `normalize`. Equivalent to
/// `values.iter().map(|v| map(normalize(...)[i])).collect()`.
#[must_use]
pub fn color_by_value(values: &[f64], method: Normalization, map: fn(f64) -> Color) -> Vec<Color> {
    normalize(values, method).into_iter().map(map).collect()
}

/// A linearly-segmented colormap built from an explicit list of stops — the
/// analogue of matplotlib's `LinearSegmentedColormap.from_list`. Sample it
/// with [`LinearSegmentedColormap::map`] or `resample_256`; where the crate
/// expects a bare [`Colormap`](crate::heatmap::Colormap) function, pair it
/// with a builtin like [`viridis`].
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSegmentedColormap {
    stops: Vec<(f64, (u8, u8, u8))>,
}

impl LinearSegmentedColormap {
    /// Build from `(position, color)` pairs. Positions must be strictly
    /// increasing and within `[0, 1]`.
    #[must_use]
    pub fn from_list(colors: &[(f64, Color)]) -> Self {
        let mut stops: Vec<(f64, (u8, u8, u8))> = colors
            .iter()
            .filter(|(t, _)| t.is_finite())
            .map(|(t, c)| (t.clamp(0.0, 1.0), c.to_rgb()))
            .collect();
        stops.sort_by(|a, b| a.0.total_cmp(&b.0));
        stops.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
        if stops.len() < 2 {
            // Fall back to a black->white ramp so sampling never panics.
            stops = vec![(0.0, (0, 0, 0)), (1.0, (255, 255, 255))];
        }
        Self { stops }
    }

    /// Build from raw `(position, (r, g, b))` stops.
    #[must_use]
    pub fn from_list_rgb(stops: &[(f64, (u8, u8, u8))]) -> Self {
        let colors: Vec<(f64, Color)> = stops
            .iter()
            .map(|(t, rgb)| (*t, Color::rgb(rgb.0, rgb.1, rgb.2)))
            .collect();
        Self::from_list(&colors)
    }

    /// Sample the colormap at `t in [0, 1]` (clamped).
    #[must_use]
    pub fn map(&self, t: f64) -> Color {
        color_map(t, &self.stops)
    }

    /// Resample to a 256-entry lookup table for smooth gradients.
    #[must_use]
    pub fn resample_256(&self) -> Vec<Color> {
        generate_colormap_256(&self.stops)
    }

    /// Get the underlying stops.
    #[must_use]
    pub fn stops(&self) -> &[(f64, (u8, u8, u8))] {
        &self.stops
    }
}

/// Two-slope (diverging) normalization around a center value — matplotlib's
/// `TwoSlopeNorm`. Values between `vmin` and `vcenter` map to `[0, 0.5]`,
/// values between `vcenter` and `vmax` to `[0.5, 1]`.
#[must_use]
pub fn normalize_two_slope(values: &[f64], vmin: f64, vcenter: f64, vmax: f64) -> Vec<f64> {
    if !(vmin < vcenter && vcenter < vmax) {
        return normalize(values, Normalization::Linear);
    }
    values
        .iter()
        .map(|&v| {
            if v <= vcenter {
                0.5 * (v - vmin) / (vcenter - vmin)
            } else {
                0.5 + 0.5 * (v - vcenter) / (vmax - vcenter)
            }
        })
        .map(|t| t.clamp(0.0, 1.0))
        .collect()
}

/// Boundary-based normalization — matplotlib's `BoundaryNorm`. Values are
/// mapped by the index of the boundary bin they fall into, so every bin gets
/// an equal share of the colormap.
#[must_use]
pub fn normalize_boundary(values: &[f64], boundaries: &[f64]) -> Vec<f64> {
    let mut bounds: Vec<f64> = boundaries.iter().copied().filter(|b| b.is_finite()).collect();
    bounds.sort_by(|a, b| a.total_cmp(b));
    bounds.dedup();
    if bounds.len() < 2 {
        return vec![0.5; values.len()];
    }
    let n_bins = bounds.len() - 1;
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return 0.5;
            }
            let mut bin = bounds.partition_point(|&b| b <= v);
            // partition_point gives the first bound > v; bin index is that minus 1.
            bin = bin.saturating_sub(1).min(n_bins);
            if v < bounds[0] {
                bin = 0;
            } else if v > *bounds.last().unwrap() {
                bin = n_bins;
            }
            if bin == n_bins {
                // Clamp the top so exactly the last bin's color is used.
                1.0 - 0.5 / n_bins as f64
            } else {
                (bin as f64 + 0.5) / n_bins as f64
            }
        })
        .collect()
}

/// RGB to HSL; returns `(hue in [0, 360), saturation, lightness)`.
#[must_use]
pub fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let (r, g, b) = (
        f64::from(r) / 255.0,
        f64::from(g) / 255.0,
        f64::from(b) / 255.0,
    );
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    if max == min {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h * 60.0, s, l)
}

/// HSL to RGB, rounding to nearest byte.
#[must_use]
pub fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h.rem_euclid(360.0) / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match hp as usize {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    let conv = |v: f64| ((v + m) * 255.0).round() as u8;
    (conv(r1), conv(g1), conv(b1))
}

/// RGB to HSV; returns `(hue in [0, 360), saturation, value)`.
#[must_use]
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let (r, g, b) = (
        f64::from(r) / 255.0,
        f64::from(g) / 255.0,
        f64::from(b) / 255.0,
    );
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let v = max;
    if max == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let d = max - min;
    let s = d / max;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h * 60.0, s, v)
}

/// HSV to RGB, rounding to nearest byte.
#[must_use]
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let hp = h.rem_euclid(360.0) / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match hp as usize {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    let conv = |v: f64| ((v + m) * 255.0).round() as u8;
    (conv(r1), conv(g1), conv(b1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_reference_stops() {
        assert_eq!(viridis(0.0).to_hex(), "#440154");
        assert_eq!(viridis(0.5).to_hex(), "#21918c");
        assert_eq!(viridis(1.0).to_hex(), "#fde725");
    }

    #[test]
    fn plasma_reference_stops() {
        assert_eq!(plasma(0.0).to_hex(), "#0d0887");
        assert_eq!(plasma(1.0).to_hex(), "#f0f921");
    }

    #[test]
    fn viridis_interpolates_midpoint() {
        assert_eq!(viridis(0.125).to_hex(), "#402a70");
    }

    #[test]
    fn viridis_clamps() {
        assert_eq!(viridis(-1.0), viridis(0.0));
        assert_eq!(viridis(2.0), viridis(1.0));
    }

    #[test]
    fn normalize_linear() {
        assert_eq!(
            normalize(&[0.0, 5.0, 10.0], Normalization::Linear),
            vec![0.0, 0.5, 1.0]
        );
        assert_eq!(
            normalize(&[3.0, 3.0], Normalization::Linear),
            vec![0.5, 0.5]
        );
    }

    #[test]
    fn normalize_log() {
        assert_eq!(
            normalize(&[1.0, 10.0, 100.0], Normalization::Log),
            vec![0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn normalize_log_falls_back_linear_on_nonpositive() {
        assert_eq!(
            normalize(&[-1.0, 0.0, 1.0], Normalization::Log),
            vec![0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn normalize_quantile_ranks() {
        assert_eq!(
            normalize(&[4.0, 1.0, 3.0, 2.0], Normalization::Quantile),
            vec![1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0]
        );
    }

    #[test]
    fn hsl_roundtrip_primaries() {
        assert_eq!(hsl_to_rgb(0.0, 1.0, 0.5), (255, 0, 0));
        assert_eq!(hsl_to_rgb(120.0, 1.0, 0.25), (0, 128, 0));
        assert_eq!(hsl_to_rgb(240.0, 1.0, 0.5), (0, 0, 255));
        let (h, s, l) = rgb_to_hsl(255, 0, 0);
        assert!((h - 0.0).abs() < 1e-9 && (s - 1.0).abs() < 1e-9 && (l - 0.5).abs() < 1e-9);
    }

    #[test]
    fn hsv_roundtrip_primaries() {
        assert_eq!(hsv_to_rgb(0.0, 1.0, 1.0), (255, 0, 0));
        assert_eq!(hsv_to_rgb(120.0, 1.0, 1.0), (0, 255, 0));
        assert_eq!(hsv_to_rgb(60.0, 1.0, 1.0), (255, 255, 0));
        let (h, s, v) = rgb_to_hsv(255, 255, 0);
        assert!((h - 60.0).abs() < 1e-9 && (s - 1.0).abs() < 1e-9 && (v - 1.0).abs() < 1e-9);
    }

    #[test]
    fn color_by_value_uses_normalization() {
        let colors = color_by_value(&[0.0, 10.0], Normalization::Linear, viridis);
        assert_eq!(colors[0], viridis(0.0));
        assert_eq!(colors[1], viridis(1.0));
    }

    #[test]
    fn from_list_red_white_blue() {
        let cm = LinearSegmentedColormap::from_list(&[
            (0.0, Color::RED),
            (0.5, Color::WHITE),
            (1.0, Color::BLUE),
        ]);
        assert_eq!(cm.map(0.0).to_rgb(), Color::RED.to_rgb());
        assert_eq!(cm.map(1.0).to_rgb(), Color::BLUE.to_rgb());
        let mid = cm.map(0.5);
        assert_eq!(mid.to_rgb(), (255, 255, 255));
        // Clamped sampling never panics.
        assert_eq!(cm.map(-1.0).to_rgb(), Color::RED.to_rgb());
        assert_eq!(cm.map(2.0).to_rgb(), Color::BLUE.to_rgb());
        assert_eq!(cm.resample_256().len(), 256);
    }

    #[test]
    fn from_list_handles_unsorted_and_duplicates() {
        let cm = LinearSegmentedColormap::from_list(&[
            (1.0, Color::BLACK),
            (0.0, Color::WHITE),
            (0.5, Color::GRAY),
            (0.5, Color::RED),
        ]);
        assert_eq!(cm.map(0.0).to_rgb(), Color::WHITE.to_rgb());
        assert_eq!(cm.map(1.0).to_rgb(), Color::BLACK.to_rgb());
    }

    #[test]
    fn from_list_falls_back_on_degenerate_input() {
        let cm = LinearSegmentedColormap::from_list(&[]);
        assert_eq!(cm.stops().len(), 2);
        let single = LinearSegmentedColormap::from_list(&[(0.0, Color::RED)]);
        assert_eq!(single.map(0.5).to_rgb(), (128, 128, 128));
    }

    #[test]
    fn two_slope_normalization_symmetric_around_center() {
        let out = normalize_two_slope(&[0.0, 5.0, 10.0], 0.0, 5.0, 10.0);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.5);
        assert_eq!(out[2], 1.0);
        let asym = normalize_two_slope(&[-5.0, 0.0, 10.0], -5.0, 0.0, 10.0);
        assert_eq!(asym[0], 0.0);
        assert_eq!(asym[1], 0.5);
        assert_eq!(asym[2], 1.0);
        // Invalid ranges fall back to linear.
        let fallback = normalize_two_slope(&[0.0, 5.0, 10.0], 0.0, 10.0, 5.0);
        assert_eq!(fallback, normalize(&[0.0, 5.0, 10.0], Normalization::Linear));
    }

    #[test]
    fn boundary_normalization_even_bins() {
        let out = normalize_boundary(&[0.0, 2.0, 4.0, 6.0], &[0.0, 2.0, 4.0, 6.0]);
        // Three bins share [0, 1) evenly: centers at 1/6, 3/6, 5/6.
        assert!((out[0] - 1.0 / 6.0).abs() < 1e-9);
        assert!((out[1] - 3.0 / 6.0).abs() < 1e-9);
        assert!((out[2] - 5.0 / 6.0).abs() < 1e-9);
        // Out-of-range clamps to the edge bins (2 bins here -> centers at
        // 0.25 and 0.75).
        let lo = normalize_boundary(&[-100.0], &[0.0, 2.0, 4.0]);
        assert_eq!(lo[0], 0.25);
        let hi = normalize_boundary(&[100.0], &[0.0, 2.0, 4.0]);
        assert!((hi[0] - (1.0 - 0.5 / 2.0)).abs() < 1e-9);
    }

    #[test]
    fn boundary_normalization_degenerate() {
        let out = normalize_boundary(&[1.0, 2.0], &[5.0]);
        assert_eq!(out, vec![0.5, 0.5]);
    }
}
