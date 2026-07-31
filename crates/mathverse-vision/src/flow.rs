use crate::Image;

pub fn lucas_kanade(a: &Image, b: &Image) -> (Image, Image) {
    assert_eq!((a.w, a.h), (b.w, b.h));
    const GX: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const GY: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let (w, h) = (a.w, a.h);
    let ax = a.convolve3(&GX);
    let ay = a.convolve3(&GY);
    let mut u = Image::new(w, h);
    let mut v = Image::new(w, h);
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let (mut sxx, mut syy, mut sxy, mut sxt, mut syt) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for dy in -2..=2i32 {
                for dx in -2..=2i32 {
                    let (px, py) = ((x as i32 + dx) as usize, (y as i32 + dy) as usize);
                    let gx = ax.get(px, py);
                    let gy = ay.get(px, py);
                    let it = b.get(px, py) - a.get(px, py);
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                    sxt += gx * it;
                    syt += gy * it;
                }
            }
            let det = sxx * syy - sxy * sxy;
            if det.abs() > 1e-8 {
                u.set(x, y, (-sxt * syy + syt * sxy) / det);
                v.set(x, y, (-syt * sxx + sxt * sxy) / det);
            }
        }
    }
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_image_zero_flow() {
        let flat = Image::from_data(16, 16, vec![0.3; 256]);
        let (u, v) = lucas_kanade(&flat, &flat);
        assert!(u.data.iter().all(|p| *p == 0.0));
        assert!(v.data.iter().all(|p| *p == 0.0));
    }

    #[test]
    fn blob_translation() {
        let (w, h) = (32, 32);
        let mut a = Image::new(w, h);
        let mut b = Image::new(w, h);
        for y in 8..24 {
            for x in 8..24 {
                a.set(x, y, 1.0);
                b.set(x + 1, y + 1, 1.0);
            }
        }
        let (u, v) = lucas_kanade(&a, &b);
        let um = u.data.iter().cloned().fold(0.0f64, f64::max);
        let vm = v.data.iter().cloned().fold(0.0f64, f64::max);
        assert!((um - 1.0).abs() < 0.3, "um {}", um);
        assert!((vm - 1.0).abs() < 0.3, "vm {}", vm);
    }
}
