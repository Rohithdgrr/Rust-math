use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_lazy_vs_eager_add(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..10_000).map(|i| (i * 2) as f64).collect();

    c.bench_function("eager_add_10k", |bencher| {
        bencher.iter(|| {
            let va = mathverse_vector::Vector::new(a.clone());
            let vb = mathverse_vector::Vector::new(b.clone());
            black_box(va.add(&vb))
        })
    });

    c.bench_function("lazy_add_10k", |bencher| {
        bencher.iter(|| {
            let va = mathverse_lazy::LazyVec::new(a.clone());
            let vb = mathverse_lazy::LazyVec::new(b.clone());
            black_box(va.add(&vb).eval())
        })
    });
}

fn bench_lazy_vs_eager_fused(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..10_000).map(|i| (i * 2) as f64).collect();
    let c_data: Vec<f64> = (0..10_000).map(|i| (i * 3) as f64).collect();

    c.bench_function("eager_mul_add_10k", |bencher| {
        bencher.iter(|| {
            let va = mathverse_vector::Vector::new(a.clone());
            let vb = mathverse_vector::Vector::new(b.clone());
            let vc = mathverse_vector::Vector::new(c_data.clone());
            let product = vb.scale(1.0);
            // Manual fused: a * b + c
            black_box(va.add(&product.add(&vc)))
        })
    });

    c.bench_function("fused_mul_add_10k", |bencher| {
        bencher.iter(|| {
            let va = mathverse_lazy::LazyVec::new(a.clone());
            let vb = mathverse_lazy::LazyVec::new(b.clone());
            let vc = mathverse_lazy::LazyVec::new(c_data.clone());
            black_box(va.mul_add(&vb, &vc).eval())
        })
    });
}

fn bench_lazy_chained_ops(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..10_000).map(|i| (i * 2) as f64).collect();

    c.bench_function("lazy_chain_add_scale_sub_10k", |bencher| {
        bencher.iter(|| {
            let va = mathverse_lazy::LazyVec::new(a.clone());
            let vb = mathverse_lazy::LazyVec::new(b.clone());
            let vc = mathverse_lazy::LazyVec::new(a.clone());
            black_box(va.add(&vb).scale(2.0).sub(&vc).eval())
        })
    });
}

fn bench_fused_kernel(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..10_000).map(|i| (i * 2) as f64).collect();
    let c_data: Vec<f64> = (0..10_000).map(|i| (i * 3) as f64).collect();
    let mut out = vec![0.0; 10_000];

    c.bench_function("fused_kernel_mul_add_10k", |bencher| {
        bencher.iter(|| {
            black_box(mathverse_lazy::FusedMulAdd::new(&a, &b, &c_data).eval(&mut out))
        })
    });
}

criterion_group!(
    lazy_benches,
    bench_lazy_vs_eager_add,
    bench_lazy_vs_eager_fused,
    bench_lazy_chained_ops,
    bench_fused_kernel,
);
criterion_main!(lazy_benches);
