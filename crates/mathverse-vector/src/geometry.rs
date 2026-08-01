pub fn angle(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x,y)| x*y).sum();
    let ma = a.iter().map(|x| x*x).sum::<f64>().sqrt();
    let mb = b.iter().map(|x| x*x).sum::<f64>().sqrt();
    if ma == 0.0 || mb == 0.0 { 0.0 } else { (dot / (ma*mb)).clamp(-1.0, 1.0).acos() }
}
pub fn distance(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt() }
pub fn project(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dot: f64 = a.iter().zip(b).map(|(x,y)| x*y).sum();
    let mag: f64 = b.iter().map(|x| x*x).sum::<f64>().sqrt();
    if mag == 0.0 { return vec![0.0; a.len()]; }
    let s = dot / (mag * mag);
    b.iter().map(|x| x * s).collect()
}
pub fn reject(a: &[f64], b: &[f64]) -> Vec<f64> {
    let p = project(a, b);
    a.iter().zip(&p).map(|(x, px)| x - px).collect()
}
pub fn triple_product(a: &[f64], b: &[f64], c: &[f64]) -> f64 {
    if a.len() != 3 || b.len() != 3 || c.len() != 3 { return 0.0; }
    a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])
}
#[allow(clippy::ptr_arg)]
pub fn gram_schmidt(vectors: &mut [Vec<f64>]) {
    let n = vectors.len();
    for i in 0..n {
        for j in 0..i {
            let dot: f64 = vectors[i].iter().zip(&vectors[j]).map(|(a,b)| a*b).sum();
            let mag: f64 = vectors[j].iter().map(|x| x*x).sum::<f64>().sqrt();
            if mag > 1e-15 {
                let s = dot / (mag * mag);
                for k in 0..vectors[i].len() { vectors[i][k] -= s * vectors[j][k]; }
            }
        }
        let mag: f64 = vectors[i].iter().map(|x| x*x).sum::<f64>().sqrt();
        if mag > 1e-15 { for v in vectors[i].iter_mut() { *v /= mag; } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn angle_test() {
        let a = angle(&[1.0,0.0], &[0.0,1.0]);
        assert!((a - core::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }
}
