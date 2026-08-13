//! Arithmetic and bitwise operations between images.
//!
//! These mirror the basic OpenCV `cv2` functions (`add`, `subtract`,
//! `multiply`, `divide`, `addWeighted`, `bitwise_and`, …). Pixel values are
//! treated as lying in `[0.0, 1.0]` and results are saturated to that range.

use crate::Image;

/// Adds two images pixel-wise, saturating to `[0.0, 1.0]`.
///
/// Equivalent to `cv2.add(a, b)`.
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn add(a: &Image, b: &Image) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "add: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = (x + y).clamp(0.0, 1.0);
    }
    out
}

/// Adds a constant to every pixel, saturating to `[0.0, 1.0]`.
pub fn add_scalar(img: &Image, s: f64) -> Image {
    let mut out = img.clone();
    for v in &mut out.data {
        *v = (*v + s).clamp(0.0, 1.0);
    }
    out
}

/// Subtracts image `b` from image `a` pixel-wise, saturating to `[0.0, 1.0]`.
///
/// Equivalent to `cv2.subtract(a, b)`.
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn subtract(a: &Image, b: &Image) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "subtract: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = (x - y).clamp(0.0, 1.0);
    }
    out
}

/// Subtracts a constant from every pixel, saturating to `[0.0, 1.0]`.
pub fn subtract_scalar(img: &Image, s: f64) -> Image {
    let mut out = img.clone();
    for v in &mut out.data {
        *v = (*v - s).clamp(0.0, 1.0);
    }
    out
}

/// Absolute difference between two images, `|a - b|` per pixel.
///
/// Equivalent to `cv2.absdiff(a, b)`.
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn absdiff(a: &Image, b: &Image) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "absdiff: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = (x - y).abs();
    }
    out
}

/// Multiplies two images pixel-wise, saturating to `[0.0, 1.0]`.
///
/// Equivalent to `cv2.multiply(a, b)`.
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn multiply(a: &Image, b: &Image) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "multiply: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = (x * y).clamp(0.0, 1.0);
    }
    out
}

/// Multiplies every pixel by a constant, saturating to `[0.0, 1.0]`.
pub fn multiply_scalar(img: &Image, s: f64) -> Image {
    let mut out = img.clone();
    for v in &mut out.data {
        *v = (*v * s).clamp(0.0, 1.0);
    }
    out
}

/// Divides image `a` by image `b` pixel-wise, saturating to `[0.0, 1.0]`.
///
/// Where `b` is zero the result is `0.0`. Equivalent to `cv2.divide(a, b)`.
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn divide(a: &Image, b: &Image) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "divide: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = if *y != 0.0 { (x / y).clamp(0.0, 1.0) } else { 0.0 };
    }
    out
}

/// Divides every pixel by a constant, saturating to `[0.0, 1.0]`.
pub fn divide_scalar(img: &Image, s: f64) -> Image {
    let mut out = img.clone();
    for v in &mut out.data {
        *v = if s != 0.0 { (*v / s).clamp(0.0, 1.0) } else { 0.0 };
    }
    out
}

/// Weighted sum of two images: `alpha · a + beta · b + gamma`, saturated to `[0.0, 1.0]`.
///
/// Equivalent to `cv2.addWeighted(a, alpha, b, beta, gamma)`. With
/// `alpha + beta = 1` and `gamma = 0` this is a linear blend (cross-fade).
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn add_weighted(a: &Image, alpha: f64, b: &Image, beta: f64, gamma: f64) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "add_weighted: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = (alpha * x + beta * y + gamma).clamp(0.0, 1.0);
    }
    out
}

/// Pixel-wise AND of two binary images.
///
/// A pixel is treated as "on" when its value is `> 0.5`; the result is `1.0`
/// where both inputs are on, else `0.0`. Equivalent to `cv2.bitwise_and`.
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn bitwise_and(a: &Image, b: &Image) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "bitwise_and: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = if *x > 0.5 && *y > 0.5 { 1.0 } else { 0.0 };
    }
    out
}

/// Pixel-wise OR of two binary images.
///
/// A pixel is treated as "on" when its value is `> 0.5`; the result is `1.0`
/// where either input is on, else `0.0`. Equivalent to `cv2.bitwise_or`.
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn bitwise_or(a: &Image, b: &Image) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "bitwise_or: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = if *x > 0.5 || *y > 0.5 { 1.0 } else { 0.0 };
    }
    out
}

/// Pixel-wise XOR of two binary images.
///
/// A pixel is treated as "on" when its value is `> 0.5`; the result is `1.0`
/// where exactly one input is on, else `0.0`. Equivalent to `cv2.bitwise_xor`.
///
/// # Panics
///
/// Panics if the images have different dimensions.
pub fn bitwise_xor(a: &Image, b: &Image) -> Image {
    assert_eq!((a.w, a.h), (b.w, b.h), "bitwise_xor: image dimensions must match");
    let mut out = Image::new(a.w, a.h);
    for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
        out.data[i] = if (*x > 0.5) != (*y > 0.5) { 1.0 } else { 0.0 };
    }
    out
}

/// Inverts a binary image (`1.0 − v` per pixel).
///
/// Equivalent to `cv2.bitwise_not`.
pub fn bitwise_not(img: &Image) -> Image {
    let mut out = img.clone();
    for v in &mut out.data {
        *v = 1.0 - *v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img2() -> Image {
        let mut a = Image::new(2, 2);
        a.data = vec![0.2, 0.4, 0.6, 0.8];
        a
    }

    #[test]
    fn add_saturates() {
        let a = img2();
        let mut b = Image::new(2, 2);
        b.data = vec![0.9, 0.8, 0.7, 0.6];
        let r = add(&a, &b);
        assert!((r.data[0] - 1.0).abs() < 1e-12);
        assert!((r.data[1] - 1.0).abs() < 1e-12);
        assert!((r.data[2] - 1.0).abs() < 1e-12);
        assert!((r.data[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn subtract_and_absdiff() {
        let a = img2();
        let mut b = Image::new(2, 2);
        b.data = vec![0.1, 0.1, 0.1, 0.1];
        let r = subtract(&a, &b);
        assert!((r.data[0] - 0.1).abs() < 1e-12);
        let d = absdiff(&b, &a);
        assert!((d.data[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn multiply_divide() {
        let a = img2();
        let mut b = Image::new(2, 2);
        b.data = vec![0.5; 4];
        let r = multiply(&a, &b);
        assert!((r.data[0] - 0.1).abs() < 1e-12);
        let d = divide(&a, &b);
        assert!((d.data[0] - 0.4).abs() < 1e-12);
        // divide by zero yields 0
        let mut z = Image::new(2, 2);
        z.data = vec![0.0; 4];
        let dz = divide(&a, &z);
        assert!(dz.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn weighted_blend() {
        let a = img2();
        let b = img2();
        // Blend of image with itself should reproduce the image.
        let r = add_weighted(&a, 0.5, &b, 0.5, 0.0);
        for i in 0..4 {
            assert!((r.data[i] - a.data[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn bitwise_ops() {
        let mut a = Image::new(1, 4);
        a.data = vec![0.0, 1.0, 0.0, 1.0];
        let mut b = Image::new(1, 4);
        b.data = vec![0.0, 0.0, 1.0, 1.0];
        let and = bitwise_and(&a, &b);
        assert_eq!(and.data, vec![0.0, 0.0, 0.0, 1.0]);
        let or = bitwise_or(&a, &b);
        assert_eq!(or.data, vec![0.0, 1.0, 1.0, 1.0]);
        let xor = bitwise_xor(&a, &b);
        assert_eq!(xor.data, vec![0.0, 1.0, 1.0, 0.0]);
        let not = bitwise_not(&a);
        assert_eq!(not.data, vec![1.0, 0.0, 1.0, 0.0]);
    }
}
