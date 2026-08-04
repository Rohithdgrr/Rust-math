use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_vector_dot(c: &mut Criterion) {
    let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..1000).map(|i| (i + 1) as f64).collect();

    c.bench_function("vector_dot_1k", |bencher| {
        bencher.iter(|| {
            let va = mathverse_vector::Vector::new(a.clone());
            let vb = mathverse_vector::Vector::new(b.clone());
            black_box(va.dot(&vb))
        })
    });
}

fn bench_vector_add(c: &mut Criterion) {
    let a: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..10_000).map(|i| (i * 2) as f64).collect();

    c.bench_function("vector_add_10k", |bencher| {
        bencher.iter(|| {
            let va = mathverse_vector::Vector::new(a.clone());
            let vb = mathverse_vector::Vector::new(b.clone());
            black_box(va.add(&vb))
        })
    });
}

fn bench_vector_norm(c: &mut Criterion) {
    let v: Vec<f64> = (0..10_000).map(|i| i as f64).collect();

    c.bench_function("vector_l2_norm_10k", |bencher| {
        bencher.iter(|| {
            let vv = mathverse_vector::Vector::new(v.clone());
            black_box(vv.norm())
        })
    });
}

fn bench_vector_scale(c: &mut Criterion) {
    let v: Vec<f64> = (0..10_000).map(|i| i as f64).collect();

    c.bench_function("vector_scale_10k", |bencher| {
        bencher.iter(|| {
            let vv = mathverse_vector::Vector::new(v.clone());
            black_box(vv.scale(2.5))
        })
    });
}

fn bench_vector_normalized(c: &mut Criterion) {
    let v: Vec<f64> = (0..1000).map(|i| i as f64).collect();

    c.bench_function("vector_normalize_1k", |bencher| {
        bencher.iter(|| {
            let vv = mathverse_vector::Vector::new(v.clone());
            black_box(vv.normalized().unwrap())
        })
    });
}

criterion_group!(
    vector_benches,
    bench_vector_dot,
    bench_vector_add,
    bench_vector_norm,
    bench_vector_scale,
    bench_vector_normalized,
);
criterion_main!(vector_benches);
