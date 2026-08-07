//! Criterion benchmarks for mathverse-calculus.
//!
//! Run with: cargo bench -p mathverse-calculus

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use mathverse_calculus::prelude::*;

fn bench_derivatives(c: &mut Criterion) {
    let f = |x: f64| x.sin() * x.cos() + x * x;
    let mut group = c.benchmark_group("derivative");
    group.bench_function("first", |b| b.iter(|| derivative(&f, black_box(1.0))));
    group.bench_function("second", |b| b.iter(|| second_derivative(&f, black_box(1.0))));
    group.bench_function("nth_5", |b| b.iter(|| nth_derivative(&f, black_box(1.0), black_box(5))));
    group.bench_function("partial", |b| {
        let g = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        b.iter(|| partial_derivative(&g, black_box(&[1.0, 2.0]), black_box(0)));
    });
    group.bench_function("discrete_gradient_1000", |b| {
        let y: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        b.iter(|| discrete_gradient(black_box(&y), black_box(0.01)));
    });
    group.finish();
}

fn bench_integration(c: &mut Criterion) {
    let f = |x: f64| x.sin() * (-x * x).exp();
    let mut group = c.benchmark_group("integration");

    for &n in &[10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("trapezoid", n), &n, |b, &n| {
            b.iter(|| trapezoid(&f, black_box(0.0), black_box(1.0), n));
        });
    }

    for &n in &[10, 100] {
        group.bench_with_input(BenchmarkId::new("simpson", n), &n, |b, &n| {
            b.iter(|| simpson(&f, black_box(0.0), black_box(1.0), n));
        });
    }

    group.bench_function("adaptive", |b| {
        b.iter(|| integrate(&f, black_box(0.0), black_box(1.0), black_box(1e-10)));
    });

    for &n in &[3, 5, 10] {
        group.bench_with_input(BenchmarkId::new("gaussian", n), &n, |b, &n| {
            b.iter(|| gaussian_quadrature(&f, black_box(0.0), black_box(1.0), n));
        });
    }

    group.bench_function("romberg", |b| {
        b.iter(|| romberg(&f, black_box(0.0), black_box(1.0), black_box(10), black_box(1e-12)));
    });

    group.bench_function("integrate_2d", |b| {
        let g = |x: f64, y: f64| x * y;
        b.iter(|| integrate_2d(&g, black_box(0.0), black_box(1.0), black_box(0.0), black_box(1.0), black_box(5)));
    });

    group.finish();
}

fn bench_ode(c: &mut Criterion) {
    let f = |_: f64, y: f64| -y;
    let mut group = c.benchmark_group("ode");

    for &steps in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("euler", steps), &steps, |b, &steps| {
            b.iter(|| euler(&f, black_box(0.0), black_box(1.0), black_box(1.0), steps));
        });
    }

    for &steps in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("rk4", steps), &steps, |b, &steps| {
            b.iter(|| runge_kutta_4(&f, black_box(0.0), black_box(1.0), black_box(1.0), steps));
        });
    }

    group.bench_function("system_rk4", |b| {
        let sys = |_: f64, y: &[f64]| vec![y[1], -y[0]];
        b.iter(|| runge_kutta_4_system(&sys, black_box(0.0), black_box(&[1.0, 0.0]), black_box(1.0), black_box(1000)));
    });

    group.bench_function("builder", |b| {
        b.iter(|| {
            OdeProblem::new(&f, black_box((0.0, 1.0)), black_box(1.0))
                .method(OdeMethod::Rk4)
                .steps(1000)
                .solve()
        });
    });

    group.finish();
}

fn bench_vector_calculus(c: &mut Criterion) {
    let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    let x = vec![1.0, 2.0, 3.0];
    let mut group = c.benchmark_group("vector_calculus");

    group.bench_function("gradient", |b| b.iter(|| gradient(&f, black_box(&x))));
    group.bench_function("laplacian", |b| b.iter(|| laplacian(&f, black_box(&x))));

    group.bench_function("divergence", |b| {
        let vfield = |p: &[f64]| p.to_vec();
        b.iter(|| divergence(&vfield, black_box(&x)));
    });

    group.bench_function("curl", |b| {
        let vfield = |p: &[f64]| vec![p[1], p[2], p[0]];
        b.iter(|| curl(&vfield, black_box(&x)));
    });

    group.bench_function("jacobian", |b| {
        let vfield = |p: &[f64]| vec![p[0] * p[1], p[0] + p[1], p[0] - p[1]];
        b.iter(|| jacobian(&vfield, black_box(&x)));
    });

    group.bench_function("hessian", |b| b.iter(|| hessian(&f, black_box(&x))));

    group.bench_function("directional_derivative", |b| {
        let v = vec![1.0, 0.0, 0.0];
        b.iter(|| directional_derivative(&f, black_box(&x), black_box(&v)));
    });

    group.finish();
}

fn bench_root_finding(c: &mut Criterion) {
    let f = |x: f64| x * x - 4.0;
    let mut group = c.benchmark_group("root_finding");

    group.bench_function("newton_auto", |b| {
        b.iter(|| newton_raphson_auto(&f, black_box(3.0), black_box(1e-12), black_box(100)));
    });

    group.bench_function("critical_point", |b| {
        let g = |x: f64| x * x * x - 3.0 * x;
        b.iter(|| find_critical_point(&g, black_box(0.5), black_box(1e-12), black_box(100)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_derivatives,
    bench_integration,
    bench_ode,
    bench_vector_calculus,
    bench_root_finding
);
criterion_main!(benches);
