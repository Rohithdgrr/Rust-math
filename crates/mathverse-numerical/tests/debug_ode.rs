use mathverse_numerical::ode::*;
use mathverse_core::error::MathResult;

#[test]
fn debug_rkf45() {
    let rkf45 = RKF45::new(1e-6, 1.0, 1e-8, 1e-8);
    let f = |_: f64, y: &[f64]| vec![y[0]];
    let result = rkf45.integrate(&f, 0.0, &[1.0], 1.0);
    match &result {
        Ok(r) => {
            eprintln!("RKF45 Steps: {}", r.len());
            eprintln!("RKF45 Final t: {}", r.last().unwrap().t);
            eprintln!("RKF45 Final y: {}", r.last().unwrap().y[0]);
            eprintln!("RKF45 Expected e: {}", std::f64::consts::E);
            eprintln!("RKF45 Error: {}", (r.last().unwrap().y[0] - std::f64::consts::E).abs());
            for (i, s) in r.iter().take(10).enumerate() {
                eprintln!("  Step {}: t={}, y={}", i, s.t, s.y[0]);
            }
        }
        Err(e) => eprintln!("RKF45 Error: {:?}", e),
    }
}

#[test]
fn debug_dormand_prince() {
    let dp = DormandPrince::new(1e-6, 1.0, 1e-8, 1e-8);
    let f = |_: f64, y: &[f64]| vec![y[0]];
    let result = dp.integrate(&f, 0.0, &[1.0], 1.0);
    match &result {
        Ok(r) => {
            eprintln!("DP Steps: {}", r.len());
            eprintln!("DP Final t: {}", r.last().unwrap().t);
            eprintln!("DP Final y: {}", r.last().unwrap().y[0]);
            eprintln!("DP Expected e: {}", std::f64::consts::E);
        }
        Err(e) => eprintln!("DP Error: {:?}", e),
    }
}

#[test]
fn debug_backward_euler() {
    let be = BackwardEuler::new(100, 1e-10);
    let f = |_: f64, y: &[f64]| vec![-10.0 * y[0]];
    let jac = |_: f64, _: &[f64]| vec![vec![-10.0]];
    let result = be.integrate(&f, &jac, 0.0, &[1.0], 1.0, 100);
    match &result {
        Ok(r) => {
            eprintln!("BE Steps: {}", r.len());
            eprintln!("BE Final t: {}", r.last().unwrap().t);
            eprintln!("BE Final y: {}", r.last().unwrap().y[0]);
            eprintln!("BE Expected exp(-10): {}", (-10.0f64).exp());
        }
        Err(e) => eprintln!("BE Error: {:?}", e),
    }
}
