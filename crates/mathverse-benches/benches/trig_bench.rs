//! Criterion benchmarks for `mathverse-trigonometry`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use mathverse_trigonometry::{
    cosd, cospi, sin, sin_cos_deg, sin_double, sind, sinpi, sum_sin, tan, tan_half,
};

const N: usize = 1_000_000;

fn bench_scalar_sincos(c: &mut Criterion) {
    let xs: Vec<f64> = (0..N).map(|i| (i % 360) as f64).collect();
    c.bench_function("trig_sincos_f64_1m", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0;
            for &x in black_box(&xs) {
                acc += sin(x) + cosd(x) + tan(x);
            }
            acc
        })
    });
}

fn bench_sinpi_cospi(c: &mut Criterion) {
    let xs: Vec<f64> = (0..N).map(|i| (i % 1000) as f64 / 16.0).collect();
    c.bench_function("trig_sinpi_cospi_1m", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0;
            for &x in &xs {
                acc += sinpi(x) + cospi(x);
            }
            acc
        })
    });
}

fn bench_degree_variants(c: &mut Criterion) {
    let xs: Vec<f64> = (0..N).map(|i| (i % 360) as f64).collect();
    c.bench_function("trig_degree_variants_1m", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0;
            for &x in &xs {
                acc += sind(x) + cosd(x) + sin_cos_deg(x).0;
            }
            acc
        })
    });
}

fn bench_batch_sumsin(c: &mut Criterion) {
    let xs: Vec<f64> = (0..N).map(|i| (i % 1000) as f64 / 97.0).collect();
    c.bench_function("trig_sum_sin_1m", |bencher| {
        bencher.iter(|| sum_sin(black_box(&xs)))
    });
}

fn bench_identities(c: &mut Criterion) {
    let xs: Vec<f64> = (0..N).map(|i| (i % 1000) as f64 / 137.0).collect();
    c.bench_function("trig_identities_1m", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0;
            for &x in &xs {
                acc += sin_double(x) + tan_half(x / 2.0);
            }
            acc
        })
    });
}

criterion_group!(
    trig_benches,
    bench_scalar_sincos,
    bench_sinpi_cospi,
    bench_degree_variants,
    bench_batch_sumsin,
    bench_identities
);
criterion_main!(trig_benches);
