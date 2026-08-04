use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_stats_mean(c: &mut Criterion) {
    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();

    c.bench_function("stats_mean_100k", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::mean(&data)))
    });
}

fn bench_stats_variance(c: &mut Criterion) {
    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();

    c.bench_function("stats_variance_100k", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::variance_sample(&data)))
    });
}

fn bench_stats_median(c: &mut Criterion) {
    let mut data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    data.reverse(); // unsorted

    c.bench_function("stats_median_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::median(&mut data)))
    });
}

fn bench_stats_covariance_matrix(c: &mut Criterion) {
    let n = 500;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![i as f64, (i * 2) as f64, (i * 3) as f64])
        .collect();

    c.bench_function("stats_cov_matrix_500x3", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::covariance_matrix(&data)))
    });
}

fn bench_stats_correlation_matrix(c: &mut Criterion) {
    let n = 500;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![i as f64, (i * 2) as f64, (i * 3) as f64])
        .collect();

    c.bench_function("stats_corr_matrix_500x3", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::correlation_matrix(&data)))
    });
}

fn bench_stats_pca(c: &mut Criterion) {
    let n = 500;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![i as f64, (i * 2) as f64, (i * 3) as f64])
        .collect();

    c.bench_function("stats_pca_500x3", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::pca(&data, 2)))
    });
}

fn bench_stats_skewness(c: &mut Criterion) {
    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();

    c.bench_function("stats_skewness_100k", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::skewness(&data)))
    });
}

fn bench_stats_kurtosis(c: &mut Criterion) {
    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();

    c.bench_function("stats_kurtosis_100k", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::kurtosis(&data)))
    });
}

fn bench_stats_linear_regression(c: &mut Criterion) {
    let x: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..10_000).map(|i| i as f64 * 2.0 + 1.0).collect();

    c.bench_function("stats_linear_regression_10k", |bencher| {
        bencher.iter(|| black_box(mathverse_statistics::linear_regression(&x, &y)))
    });
}

criterion_group!(
    stats_benches,
    bench_stats_mean,
    bench_stats_variance,
    bench_stats_median,
    bench_stats_covariance_matrix,
    bench_stats_correlation_matrix,
    bench_stats_pca,
    bench_stats_skewness,
    bench_stats_kurtosis,
    bench_stats_linear_regression,
);
criterion_main!(stats_benches);
