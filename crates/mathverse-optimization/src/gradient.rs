pub fn gradient_descent(grad: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], lr: f64, tol: f64, max_iters: usize) -> Vec<f64> {
    let mut x = x0.to_vec();
    for _ in 0..max_iters {
        let g = grad(&x);
        let next: Vec<f64> = x.iter().zip(&g).map(|(xi, gi)| xi - lr * gi).collect();
        if next.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < tol { return next; }
        x = next;
    }
    x
}

pub fn sgd(grad: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], lr: f64, momentum: f64, tol: f64, max_iters: usize) -> Vec<f64> {
    let mut x = x0.to_vec();
    let mut v = vec![0.0; x.len()];
    for _ in 0..max_iters {
        let g = grad(&x);
        let mut next = vec![0.0; x.len()];
        for i in 0..x.len() {
            v[i] = momentum * v[i] - lr * g[i];
            next[i] = x[i] + v[i];
        }
        if next.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < tol { return next; }
        x = next;
    }
    x
}

pub fn adam(grad: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], lr: f64, beta1: f64, beta2: f64, eps: f64, tol: f64, max_iters: usize) -> Vec<f64> {
    let mut x = x0.to_vec();
    let mut m = vec![0.0; x.len()];
    let mut v = vec![0.0; x.len()];
    for t in 1..=max_iters {
        let g = grad(&x);
        let mut next = x.clone();
        for i in 0..x.len() {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            let mh = m[i] / (1.0 - beta1.powi(t as i32));
            let vh = v[i] / (1.0 - beta2.powi(t as i32));
            next[i] -= lr * mh / (vh.sqrt() + eps);
        }
        if next.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < tol { return next; }
        x = next;
    }
    x
}

pub fn rmsprop(grad: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], lr: f64, decay: f64, eps: f64, tol: f64, max_iters: usize) -> Vec<f64> {
    let mut x = x0.to_vec();
    let mut acc = vec![0.0; x.len()];
    for _ in 0..max_iters {
        let g = grad(&x);
        let mut next = x.clone();
        for i in 0..x.len() {
            acc[i] = decay * acc[i] + (1.0 - decay) * g[i] * g[i];
            next[i] -= lr * g[i] / (acc[i].sqrt() + eps);
        }
        if next.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < tol { return next; }
        x = next;
    }
    x
}

pub fn adagrad(grad: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], lr: f64, eps: f64, tol: f64, max_iters: usize) -> Vec<f64> {
    let mut x = x0.to_vec();
    let mut acc = vec![0.0; x.len()];
    for _ in 0..max_iters {
        let g = grad(&x);
        let mut next = x.clone();
        for i in 0..x.len() {
            acc[i] += g[i] * g[i];
            next[i] -= lr * g[i] / (acc[i].sqrt() + eps);
        }
        if next.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < tol { return next; }
        x = next;
    }
    x
}

pub fn nadam(grad: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], lr: f64, beta1: f64, beta2: f64, eps: f64, tol: f64, max_iters: usize) -> Vec<f64> {
    let mut x = x0.to_vec();
    let mut m = vec![0.0; x.len()];
    let mut v = vec![0.0; x.len()];
    for t in 1..=max_iters {
        let g = grad(&x);
        let mut next = x.clone();
        for i in 0..x.len() {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            let mh = m[i] / (1.0 - beta1.powi(t as i32));
            let vh = v[i] / (1.0 - beta2.powi(t as i32));
            next[i] -= lr * (beta1 * mh + (1.0 - beta1) * g[i] / (1.0 - beta1.powi(t as i32))) / (vh.sqrt() + eps);
        }
        if next.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < tol { return next; }
        x = next;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gd_test() {
        let x = gradient_descent(&|x: &[f64]| x.iter().map(|v| 2.0 * v).collect(), &[10.0, -10.0], 0.1, 1e-6, 10000);
        assert!(x.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn adam_test() {
        let x = adam(&|x: &[f64]| x.iter().map(|v| 2.0 * v).collect(), &[1.0, 1.0], 1e-3, 0.9, 0.999, 1e-8, 1e-6, 10000);
        assert!(x.iter().all(|v| v.abs() < 1e-4));
    }
}
