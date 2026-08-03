//! Hough transform for line detection in binary edge images.

/// Hough transform: accumulate votes for lines in a binary edge image.
///
/// For each edge pixel `(x, y)`, votes on all `r = x·cos θ + y·sin θ` curves
/// over 180 one-degree angles `θ ∈ [0°, 180°)`. The returned accumulator has
/// shape `[2·max_r + 1][180]`, where `max_r = √(width² + height²)` and row
/// `r + max_r` indexes offset `r`. Feed the result to [`hough_find_lines`].
pub fn hough_line_accumulator(edges: &[(usize, usize)], width: usize, height: usize) -> Vec<Vec<usize>> {
    let max_r = ((width * width + height * height) as f64).sqrt() as usize;
    let n_theta = 180;
    let mut accumulator = vec![vec![0; n_theta]; 2 * max_r + 1];
    for &(x, y) in edges {
        for theta_idx in 0..n_theta {
            let theta = theta_idx as f64 * core::f64::consts::PI / 180.0;
            let r = x as f64 * theta.cos() + y as f64 * theta.sin();
            let r_idx = (r as usize) + max_r;
            if r_idx <= 2 * max_r { accumulator[r_idx][theta_idx] += 1; }
        }
    }
    accumulator
}

/// Extract line peaks from a Hough accumulator as 4-connected local maxima.
///
/// A cell `(r, θ)` is a line when its vote is `>= threshold` and is at least
/// as large as its four orthogonal neighbors. Returns `(offset, θ)` pairs,
/// where `offset` is the signed distance from the origin (`row - max_r`) and
/// `θ` is the one-degree angle index.
pub fn hough_find_lines(accumulator: &[Vec<usize>], threshold: usize, max_r: usize) -> Vec<(usize, usize)> {
    let mut lines = Vec::new();
    for r in 1..accumulator.len() - 1 {
        for theta in 1..accumulator[0].len() - 1 {
            if accumulator[r][theta] >= threshold
                && accumulator[r][theta] >= accumulator[r-1][theta]
                && accumulator[r][theta] >= accumulator[r+1][theta]
                && accumulator[r][theta] >= accumulator[r][theta-1]
                && accumulator[r][theta] >= accumulator[r][theta+1] {
                lines.push((r - max_r, theta));
            }
        }
    }
    lines
}

/// Hough transform for circles with radii in `[min_r, max_r]`.
///
/// Each edge pixel votes for centers at distance `r` along 360 one-degree
/// angles. The returned accumulator has shape
/// `[width][height][max_r - min_r + 1]`; cell `[cx][cy][r - min_r]` counts
/// circles of radius `r` centered at `(cx, cy)`.
pub fn hough_circle_accumulator(edges: &[(usize, usize)], width: usize, height: usize, min_r: usize, max_r: usize) -> Vec<Vec<Vec<usize>>> {
    let mut acc = vec![vec![vec![0; max_r - min_r + 1]; height]; width];
    for &(x, y) in edges {
        for r in min_r..=max_r {
            for theta in 0..360 {
                let t = theta as f64 * core::f64::consts::PI / 180.0;
                let cx = x as f64 - r as f64 * t.cos();
                let cy = y as f64 - r as f64 * t.sin();
                if cx >= 0.0 && cx < width as f64 && cy >= 0.0 && cy < height as f64 {
                    acc[cx as usize][cy as usize][r - min_r] += 1;
                }
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hough_test() {
        let edges = vec![(10, 10), (11, 10), (12, 10)];
        let acc = hough_line_accumulator(&edges, 20, 20);
        assert!(!acc.is_empty());
    }
}
