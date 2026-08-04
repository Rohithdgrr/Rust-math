//! Benchmarks for mathverse-plot rendering and downsampling hot paths.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mathverse_plot::common::{downsample_lttb, DataPoint, DataSeries, PlotConfig};
use mathverse_plot::SvgPlot;

/// Build `n` points of a smooth curve.
fn make_points(n: usize) -> Vec<DataPoint> {
    (0..n)
        .map(|i| {
            let x = i as f64 / n as f64 * 10.0;
            DataPoint::new(x, x.sin() * x.cos())
        })
        .collect()
}

fn bench_svg_small(c: &mut Criterion) {
    let config = PlotConfig::new().with_title("bench");
    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("s".into(), make_points(1_000)));
    c.bench_function("svg_generate_1k", |b| b.iter(|| black_box(plot.generate())));
}

fn bench_svg_large(c: &mut Criterion) {
    let config = PlotConfig::new().with_title("bench");
    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("s".into(), make_points(100_000)));
    c.bench_function("svg_generate_100k", |b| {
        b.iter(|| black_box(plot.generate()))
    });
}

fn bench_downsample(c: &mut Criterion) {
    let points = make_points(200_000);
    c.bench_function("downsample_lttb_200k_to_1k", |b| {
        b.iter(|| black_box(downsample_lttb(&points, 1_000)))
    });
    c.bench_function("downsample_lttb_200k_to_5k", |b| {
        b.iter(|| black_box(downsample_lttb(&points, 5_000)))
    });
}

criterion_group!(benches, bench_svg_small, bench_svg_large, bench_downsample);
criterion_main!(benches);
