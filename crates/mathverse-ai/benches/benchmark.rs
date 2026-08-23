//! Criterion benchmarks for tensor primitives, activations, losses and
//! matmul in the `mathverse-ai` crate.

// `criterion_group!` generates undocumented items, which trips the
// workspace-wide `missing_docs` lint; silence it for this generated code.
#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use mathverse_ai::Tensor;

/// Benchmark zero-filled tensor allocation.
fn bench_tensor_zeros(c: &mut Criterion) {
    c.bench_function("tensor_zeros", |b| b.iter(|| Tensor::zeros(&[100, 100])));
}

/// Benchmark one-filled tensor allocation.
fn bench_tensor_ones(c: &mut Criterion) {
    c.bench_function("tensor_ones", |b| b.iter(|| Tensor::ones(&[100, 100])));
}

/// Benchmark random-normal tensor generation.
fn bench_tensor_randn(c: &mut Criterion) {
    c.bench_function("tensor_randn", |b| b.iter(|| Tensor::randn(&[100, 100])));
}

/// Benchmark ReLU activation on 1000 elements.
fn bench_relu(c: &mut Criterion) {
    let t = Tensor::randn(&[1000]);
    c.bench_function("relu", |b| b.iter(|| mathverse_ai::activations::relu(&t)));
}

/// Benchmark sigmoid activation on 1000 elements.
fn bench_sigmoid(c: &mut Criterion) {
    let t = Tensor::randn(&[1000]);
    c.bench_function("sigmoid", |b| {
        b.iter(|| mathverse_ai::activations::sigmoid(&t));
    });
}

/// Benchmark 100x100 matrix multiplication.
fn bench_matmul(c: &mut Criterion) {
    let a = Tensor::randn(&[100, 100]);
    let b_t = Tensor::randn(&[100, 100]);
    c.bench_function("matmul", |b| b.iter(|| a.matmul(&b_t).unwrap()));
}

/// Benchmark softmax over a 100x10 tensor.
fn bench_softmax(c: &mut Criterion) {
    let t = Tensor::randn(&[100, 10]);
    c.bench_function("softmax", |b| {
        b.iter(|| std::hint::black_box(mathverse_ai::activations::softmax(&t, 1).unwrap()));
    });
}

/// Benchmark MSE loss on 100x100 tensors.
fn bench_mse(c: &mut Criterion) {
    let pred = Tensor::randn(&[100, 100]);
    let target = Tensor::randn(&[100, 100]);
    c.bench_function("mse", |b| {
        b.iter(|| std::hint::black_box(mathverse_ai::losses::mse(&pred, &target).unwrap()));
    });
}

criterion_group!(
    benches,
    bench_tensor_zeros,
    bench_tensor_ones,
    bench_tensor_randn,
    bench_relu,
    bench_sigmoid,
    bench_matmul,
    bench_softmax,
    bench_mse
);
criterion_main!(benches);
