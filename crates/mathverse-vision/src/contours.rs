//! Contour extraction and analysis — the `cv2.findContours` family.
//!
//! Contours are sequences of boundary pixel coordinates `(x, y)` extracted
//! from binary images (foreground = value `> 0.5`).

use crate::Image;

/// The eight clockwise directions starting at the top-left neighbor, in image
/// coordinates (y grows downward).
const DIRS: [(i64, i64); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
];

/// Finds the outer boundary contour of every foreground component using
/// Moore-neighbor tracing (a simplified `cv2.findContours` returning outer
/// contours only, in list mode).
///
/// The returned contours are ordered by the top-left position of their
/// component. Pixels are `(x, y)` tuples.
pub fn find_contours(img: &Image, connectivity: usize) -> Vec<Vec<(usize, usize)>> {
    let (labels, count) = crate::connected_components::connected_components(img, connectivity);
    let (w, h) = (img.w, img.h);
    let mut contours = Vec::new();
    for label in 1..=count {
        let is_fg = |x: i64, y: i64| -> bool {
            x >= 0
                && y >= 0
                && x < w as i64
                && y < h as i64
                && labels[y as usize * w + x as usize] == label
        };

        // Find the top-left pixel of this component.
        let mut start = None;
        'outer: for y in 0..h {
            for x in 0..w {
                if labels[y * w + x] == label {
                    start = Some((x as i64, y as i64));
                    break 'outer;
                }
            }
        }
        let Some((sx, sy)) = start else {
            continue;
        };

        let mut contour = Vec::new();
        let (mut b0x, mut b0y) = (sx, sy);
        // b1 is a background 8-neighbor of b0; start west of the start pixel.
        let (mut b1x, mut b1y) = (b0x - 1, b0y);
        while is_fg(b1x, b1y) {
            b1x -= 1;
        }
        let (first_b0, first_b1) = ((b0x, b0y), (b1x, b1y));
        let mut at_start = true;
        loop {
            // Direction from b0 to b1.
            let dir = dir_index(b1x - b0x, b1y - b0y);
            let mut found = false;
            let (mut cx, mut cy) = (0i64, 0i64);
            let (mut nb1x, mut nb1y) = (0i64, 0i64);
            // Search clockwise from the pixel after b1.
            for k in 1..=8 {
                let (dx, dy) = DIRS[(dir + k) % 8];
                let (nx, ny) = (b0x + dx, b0y + dy);
                if is_fg(nx, ny) {
                    (cx, cy) = (nx, ny);
                    let (bdx, bdy) = DIRS[(dir + k - 1) % 8];
                    (nb1x, nb1y) = (b0x + bdx, b0y + bdy);
                    found = true;
                    break;
                }
            }
            if !found {
                // Isolated single-pixel component.
                contour.push((b0x as usize, b0y as usize));
                break;
            }
            contour.push((b0x as usize, b0y as usize));
            (b0x, b0y) = (cx, cy);
            (b1x, b1y) = (nb1x, nb1y);
            // Jacob's stopping criterion: back at the start with the same entry.
            if !at_start && (b0x, b0y) == first_b0 && (b1x, b1y) == first_b1 {
                break;
            }
            at_start = false;
        }
        // Moore tracing may repeat a pixel when backtracking; drop duplicates.
        contour.dedup();
        contours.push(contour);
    }
    contours
}

fn dir_index(dx: i64, dy: i64) -> usize {
    DIRS.iter().position(|&(x, y)| x == dx && y == dy).unwrap_or(0)
}

/// Computes the area of a contour using the shoelace formula.
///
/// Returns the absolute area. Equivalent to `cv2.contourArea(contour)`.
pub fn contour_area(contour: &[(usize, usize)]) -> f64 {
    if contour.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..contour.len() {
        let (x1, y1) = contour[i];
        let (x2, y2) = contour[(i + 1) % contour.len()];
        area += x1 as f64 * y2 as f64 - x2 as f64 * y1 as f64;
    }
    area.abs() / 2.0
}

/// Computes the perimeter (arc length) of a contour.
///
/// When `closed` is true the distance from the last point back to the first is
/// included. Equivalent to `cv2.arcLength(contour, closed)`.
pub fn arc_length(contour: &[(usize, usize)], closed: bool) -> f64 {
    if contour.len() < 2 {
        return 0.0;
    }
    let mut len = 0.0;
    for pair in contour.windows(2) {
        let (x1, y1) = pair[0];
        let (x2, y2) = pair[1];
        len += ((x2 as f64 - x1 as f64).powi(2) + (y2 as f64 - y1 as f64).powi(2)).sqrt();
    }
    if closed {
        let (x1, y1) = *contour.last().unwrap();
        let (x2, y2) = contour[0];
        len += ((x2 as f64 - x1 as f64).powi(2) + (y2 as f64 - y1 as f64).powi(2)).sqrt();
    }
    len
}

/// Computes the axis-aligned bounding rectangle `(x, y, w, h)` of a contour.
///
/// Returns `None` for empty contours. Equivalent to
/// `cv2.boundingRect(contour)` (which returns `(x, y, w, h)`).
pub fn bounding_rect(contour: &[(usize, usize)]) -> Option<(usize, usize, usize, usize)> {
    if contour.is_empty() {
        return None;
    }
    let min_x = contour.iter().map(|&(x, _)| x).min().unwrap();
    let max_x = contour.iter().map(|&(x, _)| x).max().unwrap();
    let min_y = contour.iter().map(|&(_, y)| y).min().unwrap();
    let max_y = contour.iter().map(|&(_, y)| y).max().unwrap();
    Some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// Computes the convex hull of a contour using Andrew's monotone chain.
///
/// Returns the hull vertices in counter-clockwise order. Equivalent to
/// `cv2.convexHull(contour)` with the default `clockwise = false`.
pub fn convex_hull(contour: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if contour.len() < 3 {
        return contour.to_vec();
    }
    let mut pts: Vec<(i64, i64)> = contour.iter().map(|&(x, y)| (x as i64, y as i64)).collect();
    pts.sort_unstable();
    pts.dedup();
    if pts.len() < 3 {
        return contour.to_vec();
    }

    // Cross product sign for orientation (y down: keep clockwise hull).
    let cross = |o: (i64, i64), a: (i64, i64), b: (i64, i64)| -> i64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut lower: Vec<(i64, i64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(i64, i64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower.into_iter().map(|(x, y)| (x as usize, y as usize)).collect()
}

/// Approximates a contour with a polygon using the Douglas–Peucker algorithm.
///
/// `epsilon` is the maximum allowed distance between the original contour and
/// the approximation. Equivalent to `cv2.approxPolyDP(contour, epsilon, closed)`.
pub fn approx_poly_dp(contour: &[(usize, usize)], epsilon: f64, closed: bool) -> Vec<(usize, usize)> {
    let pts: Vec<(f64, f64)> = contour.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    if pts.len() <= 2 {
        return contour.to_vec();
    }

    let dist = |p: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let denom = (dx * dx + dy * dy).sqrt();
        if denom < 1e-12 {
            return ((p.0 - a.0).powi(2) + (p.1 - a.1).powi(2)).sqrt();
        }
        ((dx * (a.1 - p.1) - (a.0 - p.0) * dy).abs()) / denom
    };

    fn dp(
        pts: &[(f64, f64)],
        start: usize,
        end: usize,
        eps: f64,
        dist: &dyn Fn((f64, f64), (f64, f64), (f64, f64)) -> f64,
        out: &mut Vec<(usize, usize)>,
    ) {
        let (a, b) = (pts[start], pts[end]);
        let mut max_d = 0.0f64;
        let mut idx = start;
        for i in (start + 1)..end {
            let d = dist(pts[i], a, b);
            if d > max_d {
                max_d = d;
                idx = i;
            }
        }
        if max_d > eps {
            dp(pts, start, idx, eps, dist, out);
            out.push((pts[idx].0 as usize, pts[idx].1 as usize));
            dp(pts, idx, end, eps, dist, out);
        }
    }

    if !closed {
        let mut result = vec![contour[0]];
        dp(&pts, 0, pts.len() - 1, epsilon, &dist, &mut result);
        result.push(*contour.last().unwrap());
        return result;
    }

    // Closed: split at the farthest pair, approximate each arc, and stitch
    // the results without repeating the split points (the polygon is
    // implicitly closed, matching OpenCV's output).
    let mut far_i = 0;
    let mut far_j = 0;
    let mut far_d = 0.0;
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d = (pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2);
            if d > far_d {
                far_d = d;
                far_i = i;
                far_j = j;
            }
        }
    }
    let mut result = vec![contour[far_i]];
    // Arc i -> j (exclusive of endpoints; they are added around it).
    let mut arc: Vec<(f64, f64)> = Vec::new();
    let mut k = far_i;
    loop {
        k = (k + 1) % pts.len();
        if k == far_j {
            break;
        }
        arc.push(pts[k]);
    }
    if arc.len() >= 2 {
        dp(&arc, 0, arc.len() - 1, epsilon, &dist, &mut result);
    }
    result.push(contour[far_j]);
    // Arc j -> i (wrapping, exclusive of endpoints).
    let mut arc2: Vec<(f64, f64)> = Vec::new();
    let mut k = far_j;
    loop {
        k = (k + 1) % pts.len();
        if k == far_i {
            break;
        }
        arc2.push(pts[k]);
    }
    if arc2.len() >= 2 {
        dp(&arc2, 0, arc2.len() - 1, epsilon, &dist, &mut result);
    }
    result
}

/// Draws contours onto an image using [`crate::drawing::polylines`].
///
/// Each contour is drawn as a closed polyline with the given `color` and
/// `thickness`. Equivalent to `cv2.drawContours(img, contours, -1, color, thickness)`.
pub fn draw_contours(img: &mut Image, contours: &[Vec<(usize, usize)>], color: f64, thickness: usize) {
    for contour in contours {
        if contour.is_empty() {
            continue;
        }
        crate::drawing::polylines(img, contour, true, color, thickness);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Image {
        let mut img = Image::new(w, h);
        for y in y0..y1 {
            for x in x0..x1 {
                img.set(x, y, 1.0);
            }
        }
        img
    }

    #[test]
    fn contour_of_square() {
        let img = square(10, 10, 2, 2, 8, 8); // 6×6 filled square.
        let contours = find_contours(&img, 8);
        assert_eq!(contours.len(), 1);
        let c = &contours[0];
        let area = contour_area(c);
        assert!((area - 25.0).abs() < 1e-9, "area {area}"); // 5×5 boundary pixels.
        let (x, y, w, h) = bounding_rect(c).unwrap();
        assert_eq!((x, y, w, h), (2, 2, 6, 6));
    }

    #[test]
    fn two_contours() {
        let mut img = square(20, 20, 2, 2, 6, 6);
        for y in 12..16 {
            for x in 12..16 {
                img.set(x, y, 1.0);
            }
        }
        let contours = find_contours(&img, 8);
        assert_eq!(contours.len(), 2);
        let areas: Vec<f64> = contours.iter().map(|c| contour_area(c)).collect();
        assert!(areas.iter().all(|&a| (a - 9.0).abs() < 1e-9), "areas {areas:?}");
    }

    #[test]
    fn arc_length_of_rectangle() {
        let img = square(10, 10, 2, 2, 8, 8);
        let contours = find_contours(&img, 8);
        let c = &contours[0];
        let perim = arc_length(c, true);
        // Perimeter of a 5×5 boundary ring ≈ 4 * 5 = 20 (grid distances).
        assert!(perim > 19.0 && perim < 22.0, "perim {perim}");
    }

    #[test]
    fn convex_hull_of_l_shape() {
        // L-shape: hull should be a 4-corner polygon with area 9.
        let mut img = Image::new(6, 6);
        for y in 1..5 {
            for x in 1..4 {
                img.set(x, y, 1.0);
            }
        }
        for y in 1..3 {
            for x in 3..5 {
                img.set(x, y, 1.0);
            }
        }
        let contours = find_contours(&img, 8);
        let c = &contours[0];
        let hull = convex_hull(c);
        let hull_area = contour_area(&hull);
        // The hull of the L-shape is the pentagon (1,1),(4,1),(4,2),(3,4),(1,4)
        // with shoelace area 8.
        assert!((hull_area - 8.0).abs() < 1e-9, "hull area {hull_area}");
    }

    #[test]
    fn approx_poly_rect() {
        let img = square(10, 10, 2, 2, 8, 8);
        let contours = find_contours(&img, 8);
        let c = &contours[0];
        let approx = approx_poly_dp(c, 0.5, true);
        assert_eq!(approx.len(), 4);
    }

    #[test]
    fn draw_contours_marks_pixels() {
        let mut canvas = Image::new(10, 10);
        let img = square(10, 10, 2, 2, 8, 8);
        let contours = find_contours(&img, 8);
        draw_contours(&mut canvas, &contours, 1.0, 1);
        assert!(canvas.data.iter().any(|&v| v > 0.5));
    }
}
