//! Binary morphology: erosion, dilation, open, close on thresholded images.

use crate::GrayImage;

/// Threshold `[0,1]` image to 0/1.
pub fn binarize(img: &GrayImage, t: f64) -> GrayImage {
    let mut out = GrayImage::new(img.w, img.h).unwrap();
    for (i, &v) in img.data.iter().enumerate() {
        out.data[i] = if v >= t { 1.0 } else { 0.0 };
    }
    out
}

/// Erode with 3×3 cross: 1 if center and 4-neighbors are 1.
pub fn erode(img: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(img.w, img.h).unwrap();
    for y in 0..img.h {
        for x in 0..img.w {
            if img.get(x, y) < 0.5 {
                continue;
            }
            let ok = [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)]
                .iter()
                .all(|(dx, dy)| {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    nx >= 0 && ny >= 0 && nx < img.w as i64 && ny < img.h as i64 && img.get(nx as usize, ny as usize) >= 0.5
                });
            if ok {
                out.set(x, y, 1.0);
            }
        }
    }
    out
}

/// Dilate with 3×3 cross: 1 if any 4-neighbor is 1.
pub fn dilate(img: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(img.w, img.h).unwrap();
    for y in 0..img.h {
        for x in 0..img.w {
            if img.get(x, y) >= 0.5 {
                out.set(x, y, 1.0);
                continue;
            }
            let hit = [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)].iter().any(|(dx, dy)| {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                nx >= 0 && ny >= 0 && nx < img.w as i64 && ny < img.h as i64
                    && img.get(nx as usize, ny as usize) >= 0.5
            });
            if hit {
                out.set(x, y, 1.0);
            }
        }
    }
    out
}

pub fn open(img: &GrayImage) -> GrayImage {
    dilate(&erode(img))
}

pub fn close(img: &GrayImage) -> GrayImage {
    erode(&dilate(img))
}

/// Sum of pixel values.
pub fn sum(img: &GrayImage) -> f64 {
    img.data.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erode_dilate() {
        // 4x4 square, single-pixel border removed by erode, restored by close
        let mut img = GrayImage::new(16, 16).unwrap();
        for y in 5..10 {
            for x in 5..10 {
                img.set(x, y, 1.0);
            }
        }
        let e = erode(&img);
        assert!((sum(&e) - 9.0).abs() < 1e-9); // 3x3 after border removal
        let d = dilate(&img);
        assert!((sum(&d) - 45.0).abs() < 1e-9); // 5x5 plus one-pixel strips
        // cross structuring element: close restores the block, open rounds corners
        assert!((sum(&close(&img)) - 25.0).abs() < 1e-9);
        assert!((sum(&open(&img)) - 21.0).abs() < 1e-9);
        // erode of a 2x2 block vanishes
        let mut tiny = GrayImage::new(8, 8).unwrap();
        tiny.set(4, 4, 1.0);
        tiny.set(4, 5, 1.0);
        tiny.set(5, 4, 1.0);
        tiny.set(5, 5, 1.0);
        assert!(sum(&erode(&tiny)) < 1e-9);
    }
}
