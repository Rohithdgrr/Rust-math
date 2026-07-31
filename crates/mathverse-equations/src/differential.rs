pub fn euler(f: impl Fn(f64, f64) -> f64, y0: f64, t0: f64, tf: f64, h: f64) -> Vec<(f64, f64)> {
    let mut result = vec![(t0, y0)];
    let (mut t, mut y) = (t0, y0);
    while t < tf - 1e-15 {
        let step = h.min(tf - t);
        y += step * f(t, y);
        t += step;
        result.push((t, y));
    }
    result
}

pub fn runge_kutta4(f: impl Fn(f64, f64) -> f64, y0: f64, t0: f64, tf: f64, h: f64) -> Vec<(f64, f64)> {
    let mut result = vec![(t0, y0)];
    let (mut t, mut y) = (t0, y0);
    while t < tf - 1e-15 {
        let step = h.min(tf - t);
        let k1 = f(t, y);
        let k2 = f(t + step / 2.0, y + step * k1 / 2.0);
        let k3 = f(t + step / 2.0, y + step * k2 / 2.0);
        let k4 = f(t + step, y + step * k3);
        y += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        t += step;
        result.push((t, y));
    }
    result
}

pub fn euler_system(f: &impl Fn(f64, &[f64]) -> Vec<f64>, y0: &[f64], t0: f64, tf: f64, h: f64) -> Vec<(f64, Vec<f64>)> {
    let mut result = vec![(t0, y0.to_vec())];
    let (mut t, mut y) = (t0, y0.to_vec());
    while t < tf - 1e-15 {
        let step = h.min(tf - t);
        let dy = f(t, &y);
        for i in 0..y.len() { y[i] += step * dy[i]; }
        t += step;
        result.push((t, y.clone()));
    }
    result
}

pub fn rk4_system(f: &impl Fn(f64, &[f64]) -> Vec<f64>, y0: &[f64], t0: f64, tf: f64, h: f64) -> Vec<(f64, Vec<f64>)> {
    let n = y0.len();
    let mut result = vec![(t0, y0.to_vec())];
    let (mut t, mut y) = (t0, y0.to_vec());
    while t < tf - 1e-15 {
        let step = h.min(tf - t);
        let k1 = f(t, &y);
        let y2: Vec<f64> = y.iter().zip(&k1).map(|(yi, ki)| yi + step * ki / 2.0).collect();
        let k2 = f(t + step / 2.0, &y2);
        let y3: Vec<f64> = y.iter().zip(&k2).map(|(yi, ki)| yi + step * ki / 2.0).collect();
        let k3 = f(t + step / 2.0, &y3);
        let y4: Vec<f64> = y.iter().zip(&k3).map(|(yi, ki)| yi + step * ki).collect();
        let k4 = f(t + step, &y4);
        for i in 0..n {
            y[i] += step * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
        }
        t += step;
        result.push((t, y.clone()));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euler_exp() {
        let sol = euler(|_t, y| y, 1.0, 0.0, 1.0, 0.01);
        let last = sol.last().unwrap().1;
        assert!((last - 1.0_f64.exp()).abs() < 0.05);
    }

    #[test]
    fn rk4_exp() {
        let sol = runge_kutta4(|_t, y| y, 1.0, 0.0, 1.0, 0.01);
        let last = sol.last().unwrap().1;
        assert!((last - 1.0_f64.exp()).abs() < 0.0001);
    }
}
