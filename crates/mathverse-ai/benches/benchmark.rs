use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mathverse_ai::Tensor;

fn bench_tensor_zeros(c: &mut Criterion) {
    c.bench_function("tensor_zeros", |b| b.iter(|| Tensor::zeros(&[100, 100])));
}

fn bench_tensor_ones(c: &mut Criterion) {
    c.bench_function("tensor_ones", |b| b.iter(|| Tensor::ones(&[100, 100])));
}

fn bench_tensor_randn(c: &mut Criterion) {
    c.bench_function("tensor_randn", |b| b.iter(|| Tensor::randn(&[100, 100])));
}

fn bench_relu(c: &mut Criterion) {
    let t = Tensor::randn(&[1000]);
    c.bench_function("relu", |b| b.iter(|| mathverse_ai::activations::relu(&t)));
}

fn bench_sigmoid(c: &mut Criterion) {
    let t = Tensor::randn(&[1000]);
    c.bench_function("sigmoid", |b| b.iter(|| mathverse_ai::activations::sigmoid(&t)));
}

fn bench_matmul(c: &mut Criterion) {
    let a = Tensor::randn(&[100, 100]);
    let b_t = Tensor::randn(&[100, 100]);
    c.bench_function("matmul", |_b| _b.iter(|| a.matmul(&b_t).unwrap()));
}

fn bench_softmax(c: &mut Criterion) {
    let t = Tensor::randn(&[100, 10]);
    c.bench_function("softmax", |b| b.iter(|| mathverse_ai::activations::softmax(&t, 1).unwrap()));
}

fn bench_mse(c: &mut Criterion) {
    let pred = Tensor::randn(&[100, 100]);
    let target = Tensor::randn(&[100, 100]);
    c.bench_function("mse", |b| b.iter(|| mathverse_ai::losses::mse(&pred, &target).unwrap()));
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