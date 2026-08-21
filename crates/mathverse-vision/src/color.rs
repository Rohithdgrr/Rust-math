//! Color space conversions and operations.

use crate::Image;

/// Converts a grayscale image to a "pseudo-color" image using a simple colormap.
///
/// The conversion maps pixel values to RGB using a jet colormap-like scheme:
/// - 0.0 → blue (0, 0, 1)
/// - 0.5 → cyan/green (0, 1, 1)
/// - 1.0 → red (1, 0, 0)
/// - Intermediate values interpolate between these colors.
///
/// # Returns
///
/// An `Image` with width `w * 3` and height `h`, with data vector containing
/// interleaved RGB pixels: `[r, g, b, r, g, b, ...]` totaling `3 * w * h` elements.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, color::gray_to_jet};
///
/// let img = Image::new(4, 4);
/// let color_img = gray_to_jet(&img);
/// assert_eq!(color_img.data.len(), 3 * 4 * 4); // RGB interleaved
/// ```
pub fn gray_to_jet(img: &Image) -> Image {
    let n = img.w * img.h;
    let mut data = vec![0.0; n * 3];
    
    for (i, &val) in img.data.iter().enumerate() {
        // Clamp value to [0, 1]
        let v = val.clamp(0.0, 1.0);
        
        // Simple jet-like colormap
        let r = if v < 0.5 {
            0.0
        } else if v < 0.75 {
            (v - 0.5) * 4.0
        } else {
            1.0
        };
        
        let g = if v < 0.25 {
            v * 4.0
        } else if v < 0.75 {
            1.0
        } else {
            1.0 - (v - 0.75) * 4.0
        };
        
        let b = if v < 0.25 {
            1.0
        } else if v < 0.5 {
            1.0 - (v - 0.25) * 4.0
        } else {
            0.0
        };
        
        // Interleave RGB into the output image data
        let out_idx = i * 3;
        data[out_idx] = r;     // R
        data[out_idx + 1] = g; // G
        data[out_idx + 2] = b; // B
    }
    
    Image::from_data(img.w * 3, img.h, data)
}

/// Converts a pseudo-color RGB image back to grayscale.
///
/// The input image should have width = actual_width * 3 with interleaved RGB data.
/// The grayscale value is computed as `0.299*R + 0.587*G + 0.114*B`.
///
/// # Returns
///
/// A single-channel grayscale `Image`.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, color::{gray_to_jet, jet_to_gray}};
///
/// let img = Image::new(4, 4);
/// let color_img = gray_to_jet(&img);
/// let gray_img = jet_to_gray(&color_img);
/// assert_eq!(gray_img.w, 4);
/// assert_eq!(gray_img.h, 4);
/// ```
pub fn jet_to_gray(img: &Image) -> Image {
    let actual_w = img.w / 3;
    let n = actual_w * img.h;
    let mut data = vec![0.0; n];
    
    for i in 0..n {
        // RGB interleaved: indices i*3, i*3+1, i*3+2
        let r = img.data[i * 3];
        let g = img.data[i * 3 + 1];
        let b = img.data[i * 3 + 2];
        // Standard luminance weights
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        data[i] = gray;
    }
    
    Image::from_data(actual_w, img.h, data)
}

/// Converts an interleaved RGB image (width `3·w`) to grayscale using
/// standard luminance weights `0.299R + 0.587G + 0.114B`.
///
/// Equivalent to `cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)`.
///
/// # Panics
///
/// Panics if the width is not a multiple of 3.
pub fn rgb_to_gray(rgb: &Image) -> Image {
    assert!(rgb.w % 3 == 0, "rgb_to_gray: width must be a multiple of 3");
    let (w, h) = (rgb.w / 3, rgb.h);
    let mut data = vec![0.0; w * h];
    for i in 0..w * h {
        let r = rgb.data[i * 3];
        let g = rgb.data[i * 3 + 1];
        let b = rgb.data[i * 3 + 2];
        data[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    Image::from_data(w, h, data)
}

/// Converts a grayscale image to interleaved RGB (width `3·w`, `r = g = b`).
///
/// Equivalent to `cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)`.
pub fn gray_to_rgb(gray: &Image) -> Image {
    let mut data = Vec::with_capacity(gray.data.len() * 3);
    for &v in &gray.data {
        data.push(v);
        data.push(v);
        data.push(v);
    }
    Image::from_data(gray.w * 3, gray.h, data)
}

/// Converts a grayscale image to interleaved BGR (width `3·w`, `b = g = r`).
///
/// Equivalent to `cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)`.
pub fn gray_to_bgr(gray: &Image) -> Image {
    gray_to_rgb(gray)
}

/// Converts an interleaved RGB image to BGR (swaps red and blue channels).
///
/// Equivalent to `cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)`.
///
/// # Panics
///
/// Panics if the width is not a multiple of 3.
pub fn rgb_to_bgr(rgb: &Image) -> Image {
    assert!(rgb.w % 3 == 0, "rgb_to_bgr: width must be a multiple of 3");
    let mut out = rgb.clone();
    for i in 0..out.data.len() / 3 {
        out.data.swap(i * 3, i * 3 + 2);
    }
    out
}

/// Converts an interleaved RGB image to HSV.
///
/// The output is interleaved `[h, s, v, …]` with hue in degrees `[0, 360)`
/// and saturation/value in `[0, 1]`. Equivalent to `cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)`.
///
/// # Panics
///
/// Panics if the width is not a multiple of 3.
pub fn rgb_to_hsv(rgb: &Image) -> Image {
    assert!(rgb.w % 3 == 0, "rgb_to_hsv: width must be a multiple of 3");
    let mut data = vec![0.0; rgb.data.len()];
    for i in 0..rgb.data.len() / 3 {
        let (r, g, b) = (rgb.data[i * 3], rgb.data[i * 3 + 1], rgb.data[i * 3 + 2]);
        let (h, s, v) = rgb_to_hsv_pixel(r, g, b);
        data[i * 3] = h;
        data[i * 3 + 1] = s;
        data[i * 3 + 2] = v;
    }
    Image::from_data(rgb.w, rgb.h, data)
}

/// Converts an interleaved HSV image to RGB.
///
/// Expects hue in degrees `[0, 360)`, saturation and value in `[0, 1]`.
/// Equivalent to `cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)`.
///
/// # Panics
///
/// Panics if the width is not a multiple of 3.
pub fn hsv_to_rgb(hsv: &Image) -> Image {
    assert!(hsv.w % 3 == 0, "hsv_to_rgb: width must be a multiple of 3");
    let mut data = vec![0.0; hsv.data.len()];
    for i in 0..hsv.data.len() / 3 {
        let (h, s, v) = (hsv.data[i * 3], hsv.data[i * 3 + 1], hsv.data[i * 3 + 2]);
        let (r, g, b) = hsv_to_rgb_pixel(h, s, v);
        data[i * 3] = r;
        data[i * 3 + 1] = g;
        data[i * 3 + 2] = b;
    }
    Image::from_data(hsv.w, hsv.h, data)
}

fn rgb_to_hsv_pixel(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (r, g, b) = (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0));
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    (if h < 0.0 { h + 360.0 } else { h }, s, v)
}

fn hsv_to_rgb_pixel(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let h = ((h % 360.0) + 360.0) % 360.0;
    let (s, v) = (s.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match (h / 60.0) as usize {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r1 + m, g1 + m, b1 + m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_to_jet_preserves_range() {
        let mut img = Image::new(2, 2);
        img.data[0] = 0.0; // Black pixel
        let color = gray_to_jet(&img);
        // Black pixel (0.0) should map to blue (r=0, g=0, b=1)
        assert!((color.data[0] - 0.0).abs() < 0.1); // R
        assert!((color.data[1] - 0.0).abs() < 0.1); // G
        assert!((color.data[2] - 1.0).abs() < 0.1); // B
    }

    #[test]
    fn jet_to_gray_roundtrip() {
        let mut img = Image::new(4, 4);
        // Set some pixel values in [0, 1] range
        for i in 0..img.data.len() {
            img.data[i] = i as f64 / (img.data.len() - 1) as f64;
        }
        let color = gray_to_jet(&img);
        let gray = jet_to_gray(&color);
        // Should have correct dimensions
        assert_eq!(gray.w, 4);
        assert_eq!(gray.h, 4);
    }

    #[test]
    fn gray_rgb_roundtrip() {
        let mut img = Image::new(4, 4);
        for i in 0..img.data.len() {
            img.data[i] = i as f64 / 15.0;
        }
        let rgb = gray_to_rgb(&img);
        assert_eq!(rgb.w, 12);
        let back = rgb_to_gray(&rgb);
        for i in 0..img.data.len() {
            assert!((back.data[i] - img.data[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn bgr_swaps_channels() {
        let mut rgb = Image::new(6, 1);
        rgb.data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // red, green pixels
        let bgr = rgb_to_bgr(&rgb);
        // First pixel was pure red (1,0,0) -> BGR (0,0,1).
        assert!((bgr.data[0] - 0.0).abs() < 1e-12);
        assert!((bgr.data[1] - 0.0).abs() < 1e-12);
        assert!((bgr.data[2] - 1.0).abs() < 1e-12);
        // Second pixel was pure green (0,1,0) -> BGR (0,1,0) unchanged.
        assert!((bgr.data[3] - 0.0).abs() < 1e-12);
        assert!((bgr.data[4] - 1.0).abs() < 1e-12);
        assert!((bgr.data[5] - 0.0).abs() < 1e-12);
        let back = rgb_to_bgr(&bgr);
        for i in 0..rgb.data.len() {
            assert!((back.data[i] - rgb.data[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn hsv_roundtrip() {
        let colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
        ];
        for &(r, g, b) in &colors {
            let (h, s, v) = rgb_to_hsv_pixel(r, g, b);
            assert!((0.0..360.0).contains(&h), "hue {h}");
            assert!((0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&v));
            let (r2, g2, b2) = hsv_to_rgb_pixel(h, s, v);
            assert!((r2 - r).abs() < 1e-9, "r {r2} vs {r}");
            assert!((g2 - g).abs() < 1e-9, "g {g2} vs {g}");
            assert!((b2 - b).abs() < 1e-9, "b {b2} vs {b}");
        }
    }

    #[test]
    fn hsv_known_values() {
        // Pure red -> hue 0, s 1, v 1.
        let (h, s, v) = rgb_to_hsv_pixel(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1e-9 && (s - 1.0).abs() < 1e-9 && (v - 1.0).abs() < 1e-9);
        // Pure green -> hue 120.
        let (h, _, _) = rgb_to_hsv_pixel(0.0, 1.0, 0.0);
        assert!((h - 120.0).abs() < 1e-9);
        // Pure blue -> hue 240.
        let (h, _, _) = rgb_to_hsv_pixel(0.0, 0.0, 1.0);
        assert!((h - 240.0).abs() < 1e-9);
    }
}