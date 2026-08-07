use criterion::{criterion_group, criterion_main, Criterion};

fn bench_quadratic(c: &mut Criterion) {
    c.bench_function("solve_quadratic", |b| {
        b.iter(|| mathverse_equations::polynomial::solve_quadratic(1.0, -3.0, 2.0))
    });
}

criterion_group!(benches, bench_quadratic);
criterion_main!(benches);
