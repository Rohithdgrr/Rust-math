//! Colour-space conversion utilities.

/// Convert an RGB triple (0–255 each) to a normalised grayscale `f64` in `[0, 1]`.
///
/// Uses the Rec. 601 luma weights: `Y = 0.299R + 0.587G + 0.114B`.
pub fn rgb_to_gray(r: u8, g: u8, b: u8) -> f64 {
    (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) / 255.0
}

/// Convert an RGB triple with normalised `f64` components in `[0, 1]` to
/// grayscale.
pub fn rgb_to_gray_norm(r: f64, g: f64, b: f64) -> f64 {
    (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)
}

/// Convert RGB to HSV (Hue, Saturation, Value).
///
/// * `h` is in degrees, in `[0, 360)`.
/// * `s` and `v` are in `[0, 1]`.
///
/// Returns `(0.0, 0.0, 0.0)` for `max == 0` (black input).
pub fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let h = (h + 360.0) % 360.0; // wrap negative into [0, 360)
    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h, s, max)
}

/// Convert HSV to RGB.
///
/// * `h` — hue in degrees, `[0, 360)`
/// * `s` — saturation, `[0, 1]`
/// * `v` — value, `[0, 1]`
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    (r1 + m, g1 + m, b1 + m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_gray() {
        assert_relative_eq!(rgb_to_gray(255, 0, 0), 0.299, epsilon = 1e-3);
        assert_relative_eq!(rgb_to_gray(0, 255, 0), 0.587, epsilon = 1e-3);
        assert_relative_eq!(rgb_to_gray(0, 0, 255), 0.114, epsilon = 1e-3);
    }

    #[test]
    fn test_hsv_roundtrip() {
        let (r, g, b) = (0.3, 0.6, 0.9);
        let (h, s, v) = rgb_to_hsv(r, g, b);
        let (r2, g2, b2) = hsv_to_rgb(h, s, v);
        assert_relative_eq!(r, r2, epsilon = 1e-10);
        assert_relative_eq!(g, g2, epsilon = 1e-10);
        assert_relative_eq!(b, b2, epsilon = 1e-10);
    }

    #[test]
    fn test_black_hsv() {
        let (h, s, v) = rgb_to_hsv(0.0, 0.0, 0.0);
        assert_relative_eq!(v, 0.0, epsilon = 1e-10);
        assert_relative_eq!(s, 0.0, epsilon = 1e-10);
    }
}
