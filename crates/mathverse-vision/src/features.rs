//! Feature detection: Harris corner detector using Sobel gradients.

use crate::Image;

pub fn harris(img: &Image, _sigma: f64, k: f64) -> Image {
    const GX: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const GY: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let dx = img.convolve3(&GX);
    let dy = img.convolve3(&GY);
    let mut ixx = Image::new(img.w, img.h);
    let mut iyy = Image::new(img.w, img.h);
    let mut ixy = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        ixx.data[i] = dx.data[i] * dx.data[i];
        iyy.data[i] = dy.data[i] * dy.data[i];
        ixy.data[i] = dx.data[i] * dy.data[i];
    }
    let ixx = ixx.gaussian_blur(2, 1.0);
    let iyy = iyy.gaussian_blur(2, 1.0);
    let ixy = ixy.gaussian_blur(2, 1.0);
    let mut r = Image::new(img.w, img.h);
    for i in 0..img.data.len() {
        let (a, b, c) = (ixx.data[i], iyy.data[i], ixy.data[i]);
        r.data[i] = (a * b - c * c) - k * (a + b).powi(2);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_vs_flat() {
        let mut img = Image::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                img.set(x, y, if (x / 6 + y / 6) % 2 == 0 { 1.0 } else { 0.0 });
            }
        }
        let r = harris(&img, 1.0, 0.04);
        let at_corner = r.get(6, 6);
        let at_flat = r.get(2, 2);
        assert!(at_corner > at_flat + 1e-4, "corner {} flat {}", at_corner, at_flat);
    }

    #[test]
    fn edge_low_response() {
        let mut img = Image::new(24, 24);
        for y in 0..24 {
            for x in 0..24 {
                img.set(x, y, if x < 12 { 0.0 } else { 1.0 });
            }
        }
        let r = harris(&img, 1.0, 0.04);
        let mx = r.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(mx < 1e-4, "edge max {}", mx);
    }
}
