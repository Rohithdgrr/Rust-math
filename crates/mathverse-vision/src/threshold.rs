#! Thresholding operations.

use crate::Image;

/// Applies binary thresholding to a grayscale image.
///
/// Pixels above `thresh` are set to `maxval`, pixels below or equal are set to 0.
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `thresh` - Threshold value
/// * `maxval` - Maximum value to use with pixels that exceed the threshold
///
/// # Returns
///
/// A new `Image` with threshold applied.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, threshold::binary};
///
/// let mut img = Image::new(4, 4);
/// for i in 0..img.data.len() { img.data[i] = (i as f64 / 15.0) * 255.0; }
/// let thresh = binary(&img, 128.0, 255.0);
/// // Pixels originally > 128 should now be 255
/// // Pixels originally <= 128 should now be 0
/// ```
pub fn binary(img: &Image, thresh: f64, maxval: f64) -> Image {
    let mut out = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        out.data[i] = if img.data[i] > thresh { maxval } else { 0.0 };
    }
    out
}

/// Applies adaptive (local) thresholding to a grayscale image.
///
/// The image is divided into `block_size × block_size` regions,
/// and a separate threshold is computed for each region.
/// The threshold for each region is the mean of the pixels in that region
/// minus `constant`.
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `block_size` - Size of a square neighborhood (must be odd, >= 3)
/// * `constant` - Constant subtracted from the mean threshold
///
/// # Returns
///
/// A new `Image` with adaptive threshold applied.
///
/// # Example
///
/// ```
/// use mathverse_vision::{Image, threshold::adaptive};
///
/// let mut img = Image::new(16, 16);
/// for i in 0..img.data.len() { img.data[i] = (i as f64 / 255.0) * 255.0; }
/// let adapt = adaptive(&img, 15, 5.0);
/// ```
pub fn adaptive(img: &Image, block_size: usize, constant: f64) -> Image {
    assert!(block_size >= 3 && block_size % 2 == 1, "block_size must be odd and >= 3");
    let mut out = Image::new(img.w, img.h);
    
    let half = block_size / 2;
    
    for y in 0..img.h {
        for x in 0..img.w {
            // Compute mean in the block around (x, y)
            let (mut sum, mut count) = (0.0, 0usize);
            
            let y_min = y.saturating_sub(half);
            let y_max = (y + half).min(img.h - 1);
            let x_min = x.saturating_sub(half);
            let x_max = (x + half).min(img.w - 1);
            
            for j in y_min..=y_max {
                for i in x_min..=x_max {
                    sum += img.data[j * img.w + i];
                    count += 1;
                }
            }
            
            let mean = if count > 0 { sum / (count as f64) } else { 0.0 };
            let threshold = mean - constant;
            
            out.data[y * img.w + x] = if img.data[y * img.w + x] > threshold { 255.0 } else { 0.0 };
    }
    }
    out
}

/// Inverse binary thresholding: pixels `> thresh` become `0`, the rest `maxval`.
///
/// Equivalent to `cv2.threshold(img, thresh, maxval, THRESH_BINARY_INV)`.
pub fn binary_inv(img: &Image, thresh: f64, maxval: f64) -> Image {
    let mut out = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        out.data[i] = if img.data[i] > thresh { 0.0 } else { maxval };
    }
    out
}

/// Truncating thresholding: pixels `> thresh` are capped at `thresh`.
///
/// Equivalent to `cv2.threshold(img, thresh, maxval, THRESH_TRUNC)`.
pub fn trunc(img: &Image, thresh: f64) -> Image {
    let mut out = img.clone();
    for v in &mut out.data {
        if *v > thresh {
            *v = thresh;
        }
    }
    out
}

/// Zero thresholding: pixels `<= thresh` are zeroed, the rest unchanged.
///
/// Equivalent to `cv2.threshold(img, thresh, maxval, THRESH_TOZERO)`.
pub fn tozero(img: &Image, thresh: f64) -> Image {
    let mut out = img.clone();
    for v in &mut out.data {
        if *v <= thresh {
            *v = 0.0;
        }
    }
    out
}

/// Inverse zero thresholding: pixels `> thresh` are zeroed, the rest unchanged.
///
/// Equivalent to `cv2.threshold(img, thresh, maxval, THRESH_TOZERO_INV)`.
pub fn tozero_inv(img: &Image, thresh: f64) -> Image {
    let mut out = img.clone();
    for v in &mut out.data {
        if *v > thresh {
            *v = 0.0;
        }
    }
    out
}

/// Computes Otsu's optimal threshold over the `[0, 1]` range using a 256-bin
/// histogram, returning `(threshold, binary_image)`.
///
/// The binary image uses `maxval` for pixels above the threshold and `0`
/// otherwise — equivalent to `cv2.threshold(img, 0, maxval, THRESH_BINARY + THRESH_OTSU)`
/// (the returned threshold approximates OpenCV's `retval`).
pub fn otsu(img: &Image, maxval: f64) -> (f64, Image) {
    let hist = crate::ops::histogram(img);
    let total = img.data.len() as f64;
    if total == 0.0 {
        return (0.0, Image::new(img.w, img.h));
    }
    // Cumulative sums and sums of values for between-class variance.
    let mut sum = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        sum += i as f64 * c as f64;
    }
    let mut sum_b = 0.0f64;
    let mut w_b = 0.0f64;
    let mut best_var = 0.0f64;
    let mut best_t = 0usize;
    for t in 0..256usize {
        w_b += hist[t] as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * hist[t] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let var = w_b * w_f * (m_b - m_f).powi(2);
        if var > best_var {
            best_var = var;
            best_t = t;
        }
    }
    // The threshold is the bin *above* the best split: pixels in bins ≤ t are
    // classified as background (equivalent to OpenCV's 8-bit `src > thresh`).
    let thresh = (best_t as f64 + 1.0) / 256.0;
    (thresh, binary(img, thresh, maxval))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_threshold() {
        let img = Image::new(4, 4);
        let mut img_data = vec![0.0; 16];
        // Set high values in right half
        for i in 8..16 { img_data[i] = 200.0; }
        let mut img2 = Image::from_data(4, 4, img_data);
        let result = binary(&img2, 100.0, 255.0);
        // Right half pixels (200 > 100) should be 255
        assert_eq!(result.data[8], 255.0);
        assert_eq!(result.data[9], 255.0);
        // Left half pixels (0 <= 100) should be 0
        assert_eq!(result.data[0], 0.0);
        assert_eq!(result.data[1], 0.0);
    }

    #[test]
    fn adaptive_threshold_odd_block() {
        let img = Image::new(8, 8);
        let mut data = vec![0.0; 64];
        // Create a gradient
        for i in 0..64 { data[i] = (i as f64 / 63.0) * 255.0; }
        let img2 = Image::from_data(8, 8, data);
        let result = adaptive(&img2, 3, 0.0); // 3x3 block
        assert_eq!(result.w, 8);
        assert_eq!(result.h, 8);
    }

    #[test]
    fn threshold_type_variants() {
        let mut img = Image::new(1, 5);
        img.data = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let bin_inv = binary_inv(&img, 0.5, 1.0);
        assert_eq!(bin_inv.data, vec![1.0, 1.0, 1.0, 0.0, 0.0]);
        let tr = trunc(&img, 0.5);
        assert_eq!(tr.data, vec![0.1, 0.3, 0.5, 0.5, 0.5]);
        let tz = tozero(&img, 0.5);
        assert_eq!(tz.data, vec![0.0, 0.0, 0.0, 0.7, 0.9]);
        let tzi = tozero_inv(&img, 0.5);
        assert_eq!(tzi.data, vec![0.1, 0.3, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn otsu_separates_bimodal() {
        // Two clear clusters: 0.1 and 0.9.
        let mut img = Image::new(1, 20);
        for i in 0..10 {
            img.data[i] = 0.1;
        }
        for i in 10..20 {
            img.data[i] = 0.9;
        }
        let (t, bin) = otsu(&img, 1.0);
        // Any threshold strictly between the clusters maximizes the
        // between-class variance; histogram quantization picks one.
        assert!(t > 0.05 && t < 0.85, "threshold {t}");
        assert!(bin.data[..10].iter().all(|&v| v == 0.0));
        assert!(bin.data[10..].iter().all(|&v| v == 1.0));
    }
}