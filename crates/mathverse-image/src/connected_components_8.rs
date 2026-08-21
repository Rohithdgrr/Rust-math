//! 8-Connectivity Connected Components
//!
//! Detects and labels connected components in binary images using 8-connectivity.
//! A pixel is connected to its 8 neighbors (horizontal, vertical, and diagonal).
//!
//! # Algorithm
//!
//! 1. Scan image pixels in row-major order
//! 2. When an unvisited foreground pixel (value > 0) is found, initiate flood fill
//! 3. Use 8-connectivity: a pixel is connected to its 8 surrounding neighbors
//!    - North, South, East, West, and the 4 diagonals
//! 4. Assign a unique component label to each connected region
//! 5. Track component statistics: area, bounding box, centroid
//! 6. Return labeled image and component list
//!
//! # Typical Usage
//!
//! ```rust
//! use mathverse_image::connected_components_8::connected_components_8;
//! use mathverse_image::GrayImage;
//!
//! let mut img = GrayImage::new(64, 64).unwrap();
//! // Create two white squares
//! for y in 0..64 {
//!     for x in 0..64 {
//!         let in_square1 = (x > 10 && x < 20 && y > 10 && y < 20);
//!         let in_square2 = (x > 40 && x < 50 && y > 40 && y < 50);
//!         let v = if in_square1 || in_square2 { 1.0 } else { 0.0 };
//!         img.set(x, y, v);
//!     }
//! }
//! // Label connected components
//! let (labeled, components) = connected_components_8(&img);
//! println!("Found {} components", components.len());
//! assert_eq!(components.len(), 2);
//! ```
//!
//! # Returns
//!
//! `(labeled_img, components)` where:
//! - `labeled_img`: GrayImage whose pixel data holds each pixel's component number
//!   (1-indexed as raw `f64` label values; 0.0 = background). Labels are written
//!   directly into the data buffer so values above 1 are preserved rather than
//!   clamped by [`GrayImage::set`].
//! - `components`: Vec<Component> with statistics for each connected region,
//!   ordered by label (discovery order).

use crate::GrayImage;

/// Statistics for one connected component.
#[derive(Debug, Clone, PartialEq)]
pub struct Component {
    /// 1-based component label (matches the value written into the labeled image)
    pub label: u32,
    /// Number of foreground pixels in the component
    pub area: usize,
    /// Inclusive bounding box `(min_x, min_y, max_x, max_y)`
    pub bbox: (usize, usize, usize, usize),
    /// Centroid in pixel coordinates `(cx, cy)`
    pub centroid: (f64, f64),
}

/// Labels connected components (8-connectivity) via iterative flood fill.
///
/// Foreground pixels are those with a value strictly greater than `0.0`.
/// The flood fill uses an explicit stack, so large components cannot blow
/// the call stack.
///
/// # Example
///
/// ```rust
/// use mathverse_image::connected_components_8::connected_components_8;
/// use mathverse_image::GrayImage;
///
/// let mut img = GrayImage::new(16, 16).unwrap();
/// for y in 0..16 {
///     for x in 0..16 {
///         let fg = x >= 3 && x <= 6 && y >= 3 && y <= 6;
///         img.set(x, y, if fg { 1.0 } else { 0.0 });
///     }
/// }
/// let (labeled, components) = connected_components_8(&img);
/// assert_eq!(components.len(), 1);
/// assert_eq!(components[0].area, 16); // 4x4 block
/// assert_eq!(labeled.get(4, 4), 1.0);
/// assert_eq!(labeled.get(0, 0), 0.0); // background
/// ```
pub fn connected_components_8(img: &GrayImage) -> (GrayImage, Vec<Component>) {
    let w = img.w;
    let h = img.h;
    let mut labeled = GrayImage::new(w, h).unwrap();
    let mut visited = vec![false; w * h];
    let mut components: Vec<Component> = Vec::new();

    // 8-connected neighborhood offsets.
    const NEIGHBORS: [(i64, i64); 8] = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ];

    for sy in 0..h {
        for sx in 0..w {
            let start = sy * w + sx;
            if visited[start] || img.data[start] <= 0.0 {
                continue;
            }
            let label = (components.len() + 1) as u32;

            let mut area = 0usize;
            let (mut min_x, mut min_y) = (sx, sy);
            let (mut max_x, mut max_y) = (sx, sy);
            let mut sum_x = 0usize;
            let mut sum_y = 0usize;

            // Iterative flood fill with an explicit stack.
            let mut stack: Vec<(usize, usize)> = vec![(sx, sy)];
            visited[start] = true;

            while let Some((x, y)) = stack.pop() {
                area += 1;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                sum_x += x;
                sum_y += y;

                labeled.data[y * w + x] = label as f64;

                for &(dx, dy) in &NEIGHBORS {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    let idx = ny * w + nx;
                    if !visited[idx] && img.data[idx] > 0.0 {
                        visited[idx] = true;
                        stack.push((nx, ny));
                    }
                }
            }

            let n = area as f64;
            components.push(Component {
                label,
                area,
                bbox: (min_x, min_y, max_x, max_y),
                centroid: (sum_x as f64 / n, sum_y as f64 / n),
            });
        }
    }

    (labeled, components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrayImage;

    #[test]
    fn two_squares() {
        let mut img = GrayImage::new(64, 64).unwrap();
        for y in 0..64 {
            for x in 0..64 {
                let s1 = x > 10 && x < 20 && y > 10 && y < 20;
                let s2 = x > 40 && x < 50 && y > 40 && y < 50;
                img.set(x, y, if s1 || s2 { 1.0 } else { 0.0 });
            }
        }
        let (labeled, components) = connected_components_8(&img);
        assert_eq!(components.len(), 2);
        // Discovery order follows row-major scan: top square first.
        assert_eq!(components[0].bbox.0, 11);
        assert_eq!(components[0].bbox.2, 19);
        assert_eq!(components[1].bbox.0, 41);
        assert_eq!(components[0].label, 1);
        assert_eq!(components[1].label, 2);
        // Labels written directly into the data buffer survive unclamped.
        assert_eq!(labeled.get(15, 15), 1.0);
        assert_eq!(labeled.get(45, 45), 2.0);
        assert_eq!(labeled.get(30, 30), 0.0);
    }

    #[test]
    fn diagonal_pixels_are_connected() {
        let mut img = GrayImage::new(8, 8).unwrap();
        img.set(2, 2, 1.0);
        img.set(3, 3, 1.0); // only diagonally adjacent
        let (_, components) = connected_components_8(&img);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].area, 2);
    }

    #[test]
    fn empty_image_has_no_components() {
        let img = GrayImage::new(10, 10).unwrap();
        let (labeled, components) = connected_components_8(&img);
        assert!(components.is_empty());
        assert!(labeled.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn centroid_and_bbox() {
        let mut img = GrayImage::new(10, 10).unwrap();
        for y in 2..5 {
            for x in 4..7 {
                img.set(x, y, 1.0);
            }
        }
        let (_, components) = connected_components_8(&img);
        assert_eq!(components.len(), 1);
        let c = &components[0];
        assert_eq!(c.area, 9);
        assert_eq!(c.bbox, (4, 2, 6, 4));
        assert!((c.centroid.0 - 5.0).abs() < 1e-12);
        assert!((c.centroid.1 - 3.0).abs() < 1e-12);
    }
}
