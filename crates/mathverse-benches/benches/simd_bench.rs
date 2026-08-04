use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_simd_dot(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..10_000).map(|i| (i + 1) as f64).collect();

    c.bench_function("simd_dot_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::dot(&a, &b)))
    });
}

fn bench_simd_add(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..10_000).map(|i| (i * 2) as f64).collect();
    let mut out = vec![0.0; 10_000];

    c.bench_function("simd_add_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::add(&a, &b, &mut out)))
    });
}

fn bench_simd_scale(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let mut out = vec![0.0; 10_000];

    c.bench_function("simd_scale_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::scale(&a, 2.5, &mut out)))
    });
}

fn bench_simd_sum(c: &mut Criterion) {
    let a: Vec<f64> = (0..100_000).map(|i| i as f64).collect();

    c.bench_function("simd_sum_100k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::sum(&a)))
    });
}

fn bench_simd_exp(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.01).collect();
    let mut out = vec![0.0; 10_000];

    c.bench_function("simd_exp_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::exp(&a, &mut out)))
    });
}

fn bench_simd_sigmoid(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| (i as f64 - 5000.0) * 0.001).collect();
    let mut out = vec![0.0; 10_000];

    c.bench_function("simd_sigmoid_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::sigmoid(&a, &mut out)))
    });
}

fn bench_simd_softmax(c: &mut Criterion) {
    let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let mut out = vec![0.0; 1000];

    c.bench_function("simd_softmax_1k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::softmax(&a, &mut out)))
    });
}

fn bench_simd_axpy(c: &mut Criterion) {
    let x: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..10_000).map(|i| (i * 2) as f64).collect();
    let mut out = vec![0.0; 10_000];

    c.bench_function("simd_axpy_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::axpy(2.0, &x, &y, &mut out)))
    });
}

fn bench_simd_gemv(c: &mut Criterion) {
    let m = 256;
    let n = 256;
    let a: Vec<f64> = (0..m * n).map(|i| i as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut out = vec![0.0; m];

    c.bench_function("simd_gemv_256", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::gemv(&a, n, &x, &mut out)))
    });
}

fn bench_simd_l2_norm(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();

    c.bench_function("simd_l2_norm_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_simd::l2_norm(&a)))
    });
}

criterion_group!(
    simd_benches,
    bench_simd_dot,
    bench_simd_add,
    bench_simd_scale,
    bench_simd_sum,
    bench_simd_exp,
    bench_simd_sigmoid,
    bench_simd_softmax,
    bench_simd_axpy,
    bench_simd_gemv,
    bench_simd_l2_norm,
);
criterion_main!(simd_benches);
