//! Criterion benchmarks for optimizers, activations, and losses in the
//! `mathverse-ai` crate.

// `criterion_group!` generates undocumented items, which trips the
// workspace-wide `missing_docs` lint; silence it for this generated code.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use mathverse_ai::{Tensor, optimizers};

/// Benchmark one Adam step over 64 10x10 parameter matrices.
fn bench_adam_step(c: &mut Criterion) {
    let params_t: Vec<Tensor> = (0..64).map(|_| Tensor::randn(&[10, 10])).collect();
    let grad_data: Vec<Vec<f64>> = params_t.iter().map(|t| t.data().to_vec()).collect();
    let scaled: Vec<Vec<f64>> = grad_data
        .iter()
        .map(|d| d.iter().map(|&x| x * 0.01).collect())
        .collect();
    c.bench_function("adam_step_64", |b| {
        b.iter(|| {
            let mut params: Vec<f64> = params_t.iter().flat_map(|t| t.data().to_vec()).collect();
            let grads: Vec<f64> = scaled.iter().flatten().copied().collect();
            let mut opt = optimizers::Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0);
            opt.step(&mut params, &grads);
            black_box(params.len())
        });
    });
}

/// Benchmark one SGD-with-momentum step over 64 parameter matrices.
fn bench_sgd_step(c: &mut Criterion) {
    let params_t: Vec<Tensor> = (0..64).map(|_| Tensor::randn(&[10, 10])).collect();
    let grad_data: Vec<Vec<f64>> = params_t.iter().map(|t| t.data().to_vec()).collect();
    let scaled: Vec<Vec<f64>> = grad_data
        .iter()
        .map(|d| d.iter().map(|&x| x * 0.01).collect())
        .collect();
    c.bench_function("sgd_step_64", |b| {
        b.iter(|| {
            let mut params: Vec<f64> = params_t.iter().flat_map(|t| t.data().to_vec()).collect();
            let grads: Vec<f64> = scaled.iter().flatten().copied().collect();
            let mut opt = optimizers::Sgd::new(0.01, 0.9, 0.0);
            opt.step(&mut params, &grads);
            black_box(params.len())
        });
    });
}

/// Benchmark one Adam step over 512 100x100 parameter matrices.
fn bench_optimizer_large(c: &mut Criterion) {
    let params_t: Vec<Tensor> = (0..512).map(|_| Tensor::randn(&[100, 100])).collect();
    let grad_data: Vec<Vec<f64>> = params_t.iter().map(|t| t.data().to_vec()).collect();
    let scaled: Vec<Vec<f64>> = grad_data
        .iter()
        .map(|d| d.iter().map(|&x| x * 0.001).collect())
        .collect();
    c.bench_function("adam_step_512", |b| {
        b.iter(|| {
            let mut params: Vec<f64> = params_t.iter().flat_map(|t| t.data().to_vec()).collect();
            let grads: Vec<f64> = scaled.iter().flatten().copied().collect();
            let mut opt = optimizers::Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0);
            opt.step(&mut params, &grads);
            black_box(params.len())
        });
    });
}

/// Benchmark one Adam step over four 2x2 parameters.
fn bench_optimizer_small(c: &mut Criterion) {
    let params_t: Vec<Tensor> = (0..4).map(|_| Tensor::randn(&[2, 2])).collect();
    let grad_data: Vec<Vec<f64>> = params_t.iter().map(|t| t.data().to_vec()).collect();
    let scaled: Vec<Vec<f64>> = grad_data
        .iter()
        .map(|d| d.iter().map(|&x| x * 0.1).collect())
        .collect();
    c.bench_function("adam_step_4", |b| {
        b.iter(|| {
            let mut params: Vec<f64> = params_t.iter().flat_map(|t| t.data().to_vec()).collect();
            let grads: Vec<f64> = scaled.iter().flatten().copied().collect();
            let mut opt = optimizers::Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0);
            opt.step(&mut params, &grads);
            black_box(params.len())
        });
    });
}

/// Benchmark ReLU forward pass on 1000 elements.
fn bench_relu_forward(c: &mut Criterion) {
    let t = Tensor::randn(&[1000]);
    c.bench_function("relu_forward", |b| {
        b.iter(|| mathverse_ai::activations::relu(&t));
    });
}

/// Benchmark sigmoid forward pass on 1000 elements.
fn bench_sigmoid_forward(c: &mut Criterion) {
    let t = Tensor::randn(&[1000]);
    c.bench_function("sigmoid_forward", |b| {
        b.iter(|| mathverse_ai::activations::sigmoid(&t));
    });
}

/// Benchmark MSE loss on 100x100 tensors.
fn bench_mse_loss(c: &mut Criterion) {
    let pred = Tensor::randn(&[100, 100]);
    let target = Tensor::randn(&[100, 100]);
    c.bench_function("mse_loss", |b| {
        b.iter(|| mathverse_ai::losses::mse(&pred, &target).unwrap());
    });
}

criterion_group!(
    benches,
    bench_adam_step,
    bench_sgd_step,
    bench_optimizer_large,
    bench_optimizer_small,
    bench_relu_forward,
    bench_sigmoid_forward,
    bench_mse_loss
);
criterion_main!(benches);
