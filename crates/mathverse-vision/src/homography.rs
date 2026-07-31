#[derive(Debug, Clone)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let h = &self.0;
        let w = h[6] * x + h[7] * y + h[8];
        ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
    }
}

fn solve_9x9(a: &[[f64; 9]; 9], b: &[f64; 9]) -> Option<[f64; 9]> {
    let mut sys = [[0.0f64; 10]; 9];
    for i in 0..9 {
        sys[i][..9].copy_from_slice(&a[i]);
        sys[i][9] = b[i];
    }
    for col in 0..9 {
        let mut max_row = col;
        for row in (col + 1)..9 {
            if sys[row][col].abs() > sys[max_row][col].abs() {
                max_row = row;
            }
        }
        sys.swap(col, max_row);
        if sys[col][col].abs() < 1e-30 {
            return None;
        }
        let pivot = sys[col][col];
        for j in col..=9 {
            sys[col][j] /= pivot;
        }
        for row in 0..9 {
            if row != col && sys[row][col].abs() > 1e-30 {
                let f = sys[row][col];
                for j in col..=9 {
                    sys[row][j] -= f * sys[col][j];
                }
            }
        }
    }
    let mut x = [0.0f64; 9];
    for i in 0..9 {
        x[i] = sys[i][9];
    }
    Some(x)
}

fn smallest_eigenvector(a: &[[f64; 9]; 9]) -> Option<[f64; 9]> {
    // Estimate largest eigenvalue via power iteration
    let mut v = [1.0f64 / 3.0; 9];
    let mut _lambda_max = 0.0;
    for _ in 0..60 {
        let mut w = [0.0f64; 9];
        for i in 0..9 {
            for j in 0..9 {
                w[i] += a[i][j] * v[j];
            }
        }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        _lambda_max = norm;
        for i in 0..9 {
            v[i] = w[i] / norm;
        }
    }
    // Inverse iteration with shift to find smallest eigenvector
    let shift = 0.001;
    let mut b = [1.0f64 / 3.0; 9];
    for _ in 0..50 {
        // Build (A - shift*I) matrix
        let mut m = [[0.0f64; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                m[i][j] = a[i][j] - if i == j { shift } else { 0.0 };
            }
        }
        let Some(mut x) = solve_9x9(&m, &b) else {
            return None;
        };
        let norm: f64 = x.iter().map(|t| t * t).sum::<f64>().sqrt();
        if norm < 1e-30 {
            return None;
        }
        for i in 0..9 {
            x[i] /= norm;
        }
        b = x;
    }
    Some(b)
}

pub fn homography_dlt(src: &[(f64, f64)], dst: &[(f64, f64)]) -> Option<Homography> {
    let n = src.len();
    if n < 4 || n != dst.len() {
        return None;
    }
    // Build 2n×9 DLT matrix, compute A^T A (9×9)
    let mut ata = [[0.0f64; 9]; 9];
    for (_i, ((sx, sy), (dx, dy))) in src.iter().zip(dst).enumerate() {
        // Row 2i:   [-sx, -sy, -1, 0, 0, 0, dx*sx, dx*sy, dx]
        // Row 2i+1:  [0, 0, 0, -sx, -sy, -1, dy*sx, dy*sy, dy]
        let rows: [[f64; 9]; 2] = [
            [-sx, -sy, -1.0, 0.0, 0.0, 0.0, dx * sx, dx * sy, *dx],
            [0.0, 0.0, 0.0, -sx, -sy, -1.0, dy * sx, dy * sy, *dy],
        ];
        for row in &rows {
            for ci in 0..9 {
                for cj in 0..9 {
                    ata[ci][cj] += row[ci] * row[cj];
                }
            }
        }
    }
    let v = smallest_eigenvector(&ata)?;
    Some(Homography(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    #[test]
    fn identity() {
        let src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dst = src;
        let h = homography_dlt(&src, &dst).unwrap();
        for ((sx, sy), (dx, dy)) in src.iter().zip(&dst) {
            let (px, py) = h.apply(*sx, *sy);
            assert!((px - dx).abs() < EPS && (py - dy).abs() < EPS);
        }
    }

    #[test]
    fn scale_2x() {
        let src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dst = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let h = homography_dlt(&src, &dst).unwrap();
        let (px, py) = h.apply(0.5, 0.5);
        assert!((px - 1.0).abs() < EPS && (py - 1.0).abs() < EPS);
    }

    #[test]
    fn linear_transform() {
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..3i32 {
            for j in 0..2i32 {
                src.push((i as f64, j as f64));
                dst.push((2.0 * i as f64 + 0.3 * j as f64 + 1.0, j as f64 - 0.1 * i as f64 - 2.0));
            }
        }
        let h = homography_dlt(&src, &dst).unwrap();
        let err: f64 = src
            .iter()
            .zip(&dst)
            .map(|(s, d)| {
                let (px, py) = h.apply(s.0, s.1);
                (px - d.0).powi(2) + (py - d.1).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-3, "reprojection error: {}", err);
    }
}
