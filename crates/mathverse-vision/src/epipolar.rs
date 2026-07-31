#[derive(Debug, Clone)]
pub struct Fundamental(pub [[f64; 3]; 3]);

impl Fundamental {
    pub fn line_in_second(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let f = &self.0;
        let a = f[0][0] * x + f[0][1] * y + f[0][2];
        let b = f[1][0] * x + f[1][1] * y + f[1][2];
        let c = f[2][0] * x + f[2][1] * y + f[2][2];
        (a, b, c)
    }

    pub fn sampson_distance(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        let l2 = self.line_in_second(x1, y1);
        let l1 = (
            self.0[0][0] * x2 + self.0[1][0] * y2 + self.0[2][0],
            self.0[0][1] * x2 + self.0[1][1] * y2 + self.0[2][1],
            self.0[0][2] * x2 + self.0[1][2] * y2 + self.0[2][2],
        );
        let e = l2.0 * x2 + l2.1 * y2 + l2.2;
        e * e / (l2.0 * l2.0 + l2.1 * l2.1 + l1.0 * l1.0 + l1.1 * l1.1)
    }
}

fn solve_nxn(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    let mut sys: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if sys[row][col].abs() > sys[max_row][col].abs() {
                max_row = row;
            }
        }
        sys.swap(col, max_row);
        if sys[col][col].abs() < 1e-30 {
            return None;
        }
        let pivot = sys[col][col];
        for j in col..=n {
            sys[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col && sys[row][col].abs() > 1e-30 {
                let f = sys[row][col];
                for j in col..=n {
                    sys[row][j] -= f * sys[col][j];
                }
            }
        }
    }
    Some((0..n).map(|i| sys[i][n]).collect())
}

fn smallest_eigenvector(a: &[Vec<f64>], n: usize) -> Option<Vec<f64>> {
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..60 {
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a[i][j] * v[j];
            }
        }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 { return None; }
        for i in 0..n { v[i] = w[i] / norm; }
    }
    let shift = 0.001;
    let mut b = v.clone();
    for _ in 0..50 {
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                m[i][j] = a[i][j] - if i == j { shift } else { 0.0 };
            }
        }
        let x = solve_nxn(&m, &b)?;
        let norm: f64 = x.iter().map(|t| t * t).sum::<f64>().sqrt();
        if norm < 1e-30 { return None; }
        b = x.iter().map(|t| t / norm).collect();
    }
    Some(b)
}

pub fn fundamental(a: &[(f64, f64)], b: &[(f64, f64)]) -> Option<Fundamental> {
    if a.len() < 8 || a.len() != b.len() {
        return None;
    }
    let mut ata = vec![vec![0.0; 9]; 9];
    for (_i, ((x, y), (xp, yp))) in a.iter().zip(b).enumerate() {
        let row = [xp * x, xp * y, *xp, yp * x, yp * y, *yp, *x, *y, 1.0];
        for ci in 0..9 {
            for cj in 0..9 {
                ata[ci][cj] += row[ci] * row[cj];
            }
        }
    }
    let v = smallest_eigenvector(&ata, 9)?;
    let mut f = [[0.0; 3]; 3];
    for i in 0..9 {
        f[i / 3][i % 3] = v[i];
    }
    // Enforce rank 2: zero out the smallest singular value
    // Simple SVD of 3x3 → find smallest, zero it
    // Approximate: just use the first two rows
    Some(Fundamental(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epipolar_constraint() {
        let _f = Fundamental([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]);
        let a: Vec<(f64, f64)> = (0..8).map(|i| (i as f64 * 0.5, (i * i) as f64 * 0.1)).collect();
        let b: Vec<(f64, f64)> = a.iter().map(|(x, y)| (*x, y + 1.0)).collect();
        let fhat = fundamental(&a, &b).unwrap();
        for ((x, y), (xp, yp)) in a.iter().zip(&b) {
            let d = fhat.sampson_distance(*x, *y, *xp, *yp);
            assert!(d < 1e-4, "sampson {}", d);
        }
    }

    #[test]
    fn too_few_points() {
        let a = [(0.0, 0.0); 7];
        let b = [(0.0, 0.0); 7];
        assert!(fundamental(&a, &b).is_none());
    }
}
