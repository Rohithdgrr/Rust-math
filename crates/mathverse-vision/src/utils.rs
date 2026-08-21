//! Utility functions for computer vision.

use crate::Image;

/// Computes the mean pixel value of an image.
///
/// # Returns
///
/// The mean as an `f64`.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, utils::mean};
///
/// let mut img = Image::new(2, 2);
/// img.set(0, 0, 10.0);
/// img.set(1, 0, 20.0);
/// img.set(0, 1, 30.0);
/// img.set(1, 1, 40.0);
/// let m = mean(&img);
/// assert!((m - 25.0).abs() < 1e-9);
/// ```
pub fn mean(img: &Image) -> f64 {
    let sum: f64 = img.data.iter().sum();
    sum / (img.w * img.h) as f64
}

/// Computes the standard deviation of pixel values in an image.
///
/// # Returns
///
/// The standard deviation as an `f64`.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, utils::std_dev};
///
/// let mut img = Image::new(2, 2);
/// img.set(0, 0, 10.0);
/// img.set(1, 0, 20.0);
/// img.set(0, 1, 30.0);
/// img.set(1, 1, 40.0);
/// let s = std_dev(&img);
// population std dev of [10, 20, 30, 40] = sqrt(500/4) = sqrt(125) ≈ 11.18
/// let expected = 125.0_f64.sqrt();
/// assert!((s - expected).abs() < 1e-9);
/// ```
pub fn std_dev(img: &Image) -> f64 {
    let mean_val = mean(img);
    let variance: f64 = img.data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (img.w * img.h) as f64;
    variance.sqrt()
}

/// Computes the min and max pixel values and their locations.
///
/// # Returns
///
/// `(min_val, max_val, min_loc, max_loc)` where `min_loc` and `max_loc`
/// are `(x, y)` tuples.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, utils::min_max};
///
/// let mut img = Image::new(2, 2);
/// img.set(0, 0, 10.0);
/// img.set(1, 0, 50.0);
/// img.set(0, 1, 20.0);
/// img.set(1, 1, 5.0);
/// let (min_v, max_v, min_l, max_l) = min_max(&img);
/// assert!((min_v - 5.0).abs() < 1e-9);
/// assert!((max_v - 50.0).abs() < 1e-9);
/// assert_eq!(min_l, (1, 1)); // (x, y) of min value
/// assert_eq!(max_l, (1, 0)); // (x, y) of max value
/// ```
pub fn min_max(img: &Image) -> (f64, f64, (usize, usize), (usize, usize)) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut min_loc = (0usize, 0usize);
    let mut max_loc = (0usize, 0usize);
    
    for y in 0..img.h {
        for x in 0..img.w {
            let idx = y * img.w + x;
            let val = img.data[idx];
            if val < min_val {
                min_val = val;
                min_loc = (x, y);
            }
            if val > max_val {
                max_val = val;
                max_loc = (x, y);
            }
        }
    }
    
    (min_val, max_val, min_loc, max_loc)
}

/// Draws a simple cross (plus sign) on an image at the specified location.
///
/// # Arguments
///
/// * `img` - Mutable reference to the image
/// * `center` - Center point (x, y)
/// * `size` - Half the size of the cross
/// * `color` - Color value
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, utils::cross};
///
/// let mut img = Image::new(10, 10);
/// cross(&mut img, (5, 5), 3, 1.0);
/// ```
pub fn cross(img: &mut Image, center: (usize, usize), size: usize, color: f64) {
    let (cx, cy) = center;
    
    // Horizontal line
    let s = size as i32;
    for dx in -s..=s {
        let x = (cx as i32 + dx) as usize;
        let y = cy;
        if x < img.w && y < img.h {
            img.data[y * img.w + x] = color;
        }
    }
    
    // Vertical line
    for dy in -s..=s {
        let x = cx;
        let y = (cy as i32 + dy) as usize;
        if x < img.w && y < img.h {
            img.data[y * img.w + x] = color;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_basic() {
        let mut img = Image::new(2, 2);
        img.set(0, 0, 10.0);
        img.set(1, 0, 20.0);
        img.set(0, 1, 30.0);
        img.set(1, 1, 40.0);
        let m = mean(&img);
        assert!((m - 25.0).abs() < 1e-9);
    }

    #[test]
    fn min_max_basic() {
        let mut img = Image::new(2, 2);
        img.set(0, 0, 10.0);
        img.set(1, 0, 50.0);
        img.set(0, 1, 20.0);
        img.set(1, 1, 5.0);
        let (min_v, max_v, min_l, max_l) = min_max(&img);
        assert!((min_v - 5.0).abs() < 1e-9);
        assert!((max_v - 50.0).abs() < 1e-9);
        // Min (5.0) lives at (1, 1); max (50.0) at (1, 0).
        assert_eq!(min_l, (1, 1));
        assert_eq!(max_l, (1, 0));
    }
}