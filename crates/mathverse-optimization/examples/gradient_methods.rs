//! Gradient-based optimization examples.
//!
//! This example demonstrates gradient descent, SGD, Adam, and RMSProp.

use mathverse_optimization::{gradient_descent, sgd, adam, rmsprop, GdConfig, SgdConfig, AdamConfig, RmsPropConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize f(x) = x² + y², gradient = (2x, 2y)
    let grad = |x: &[f64]| x.iter().map(|v| 2.0 * v).collect();
    
    println!("Minimizing f(x) = x² + y²");
    println!("Starting point: [10.0, -10.0]");
    
    // Gradient Descent
    let gd_cfg = GdConfig::default();
    let gd_result = gradient_descent(&grad, &[10.0, -10.0], &gd_cfg)?;
    println!("Gradient Descent result: {:?}", gd_result);
    println!("Final value: {:.6}", gd_result.iter().map(|v| v * v).sum::<f64>());
    
    // SGD with momentum
    let sgd_cfg = SgdConfig::default();
    let sgd_result = sgd(&grad, &[10.0, -10.0], &sgd_cfg)?;
    println!("SGD result: {:?}", sgd_result);
    println!("Final value: {:.6}", sgd_result.iter().map(|v| v * v).sum::<f64>());
    
    // Adam
    let adam_cfg = AdamConfig::default();
    let adam_result = adam(&grad, &[10.0, -10.0], &adam_cfg)?;
    println!("Adam result: {:?}", adam_result);
    println!("Final value: {:.6}", adam_result.iter().map(|v| v * v).sum::<f64>());
    
    // RMSProp
    let rms_cfg = RmsPropConfig::default();
    let rms_result = rmsprop(&grad, &[10.0, -10.0], &rms_cfg)?;
    println!("RMSProp result: {:?}", rms_result);
    println!("Final value: {:.6}", rms_result.iter().map(|v| v * v).sum::<f64>());
    
    Ok(())
}
