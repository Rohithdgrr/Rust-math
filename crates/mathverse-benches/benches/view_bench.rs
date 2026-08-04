use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_owned_row_extract(c: &mut Criterion) {
    let data: Vec<f64> = (0..256 * 256).map(|i| i as f64).collect();
    let m = mathverse_matrix::Matrix {
        rows: 256,
        cols: 256,
        data,
    };

    c.bench_function("owned_row_extract_256", |bencher| {
        bencher.iter(|| black_box(m.row(128)))
    });
}

fn bench_view_row_extract(c: &mut Criterion) {
    let data: Vec<f64> = (0..256 * 256).map(|i| i as f64).collect();
    let view = mathverse_views::MatView::new(&data, 256, 256);

    c.bench_function("view_row_extract_256", |bencher| {
        bencher.iter(|| black_box(view.row_slice(128)))
    });
}

fn bench_owned_col_extract(c: &mut Criterion) {
    let data: Vec<f64> = (0..256 * 256).map(|i| i as f64).collect();
    let m = mathverse_matrix::Matrix {
        rows: 256,
        cols: 256,
        data,
    };

    c.bench_function("owned_col_extract_256", |bencher| {
        bencher.iter(|| black_box(m.col(128)))
    });
}

fn bench_view_col_extract(c: &mut Criterion) {
    let data: Vec<f64> = (0..256 * 256).map(|i| i as f64).collect();
    let view = mathverse_views::MatView::new(&data, 256, 256);

    c.bench_function("view_col_extract_256", |bencher| {
        bencher.iter(|| black_box(view.col(128)))
    });
}

fn bench_vec_view_ops(c: &mut Criterion) {
    let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    let view = mathverse_views::VecView::new(&data);

    c.bench_function("vec_view_sum_10k", |bencher| {
        bencher.iter(|| black_box(view.sum()))
    });

    c.bench_function("vec_view_norm_10k", |bencher| {
        bencher.iter(|| black_box(view.norm()))
    });

    c.bench_function("vec_view_mean_10k", |bencher| {
        bencher.iter(|| black_box(view.mean()))
    });
}

fn bench_mat_view_trace(c: &mut Criterion) {
    let data: Vec<f64> = (0..256 * 256).map(|i| i as f64).collect();
    let view = mathverse_views::MatView::new(&data, 256, 256);

    c.bench_function("mat_view_trace_256", |bencher| {
        bencher.iter(|| black_box(view.trace().unwrap()))
    });
}

criterion_group!(
    view_benches,
    bench_owned_row_extract,
    bench_view_row_extract,
    bench_owned_col_extract,
    bench_view_col_extract,
    bench_vec_view_ops,
    bench_mat_view_trace,
);
criterion_main!(view_benches);
