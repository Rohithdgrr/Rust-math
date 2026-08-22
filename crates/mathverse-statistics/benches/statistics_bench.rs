//! Criterion micro-benchmarks for the statistics crate.
// criterion_group! expands to undocumented functions; silence missing_docs
// for generated benchmark harness code only.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Baseline no-op benchmark used to calibrate harness overhead.
fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| black_box(1)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
