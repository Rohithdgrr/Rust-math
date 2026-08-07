//! Connected-component labelling via 4-connectivity flood fill.

use crate::GrayImage;

/// Label connected components in a binary image using 4-connectivity.
///
/// Pixels with value `>= threshold` are treated as foreground. Each connected
/// component receives a unique integer label starting from `1`. Background
/// pixels receive label `0`.
///
/// # Returns
///
/// A tuple of `(labelled_image, component_count)`:
/// - `labelled_image` — a `GrayImage` where each pixel stores the component
///   label divided by `component_count` (so values lie in `[0, 1]`)
/// - `component_count` — number of distinct connected components found
pub fn connected_components(img: &GrayImage, threshold: f64) -> (GrayImage, usize) {
    use std::collections::VecDeque;

    let mut labels = vec![0usize; img.w * img.h];
    let mut next_label: usize = 1;

    for y in 0..img.h {
        for x in 0..img.w {
            if img.get(x, y) < threshold {
                continue;
            }
            if labels[y * img.w + x] != 0 {
                continue;
            }

            // flood-fill
            let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
            queue.push_back((x, y));
            labels[y * img.w + x] = next_label;

            while let Some((cx, cy)) = queue.pop_front() {
                for &(dx, dy) in &[(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx as i64 + dx;
                    let ny = cy as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= img.w as i64 || ny >= img.h as i64 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if img.get(nx, ny) >= threshold && labels[ny * img.w + nx] == 0 {
                        labels[ny * img.w + nx] = next_label;
                        queue.push_back((nx, ny));
                    }
                }
            }
            next_label += 1;
        }
    }

    let count = next_label - 1;
    let mut out = GrayImage::new(img.w, img.h).unwrap();
    if count > 0 {
        for i in 0..labels.len() {
            out.data[i] = labels[i] as f64 / count as f64;
        }
    }
    (out, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_component() {
        let mut img = GrayImage::new(4, 4).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                img.set(x, y, 1.0);
            }
        }
        let (labels, count) = connected_components(&img, 0.5);
        assert_eq!(count, 1);
        assert!(labels.data.iter().all(|v| (*v - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_two_components() {
        let mut img = GrayImage::new(6, 3).unwrap();
        // left 3x3 block = 1, right 3x3 block = 1, gap at column 2
        for y in 0..3 {
            for x in 0..6 {
                img.set(x, y, if x < 2 { 1.0 } else if x > 3 { 1.0 } else { 0.0 });
            }
        }
        let (_labels, count) = connected_components(&img, 0.5);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_no_components() {
        let img = GrayImage::new(4, 4).unwrap(); // all zeros
        let (labels, count) = connected_components(&img, 0.5);
        assert_eq!(count, 0);
        assert!(labels.data.iter().all(|v| *v == 0.0));
    }
}
