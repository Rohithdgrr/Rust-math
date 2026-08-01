//! Benchmarks for mathverse-core hot paths.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_gcd(c: &mut Criterion) {
    c.bench_function("gcd(48, 18)", |b| {
        b.iter(|| mathverse_core::algorithms::gcd(black_box(48), black_box(18)))
    });
    c.bench_function("gcd(123456, 789012)", |b| {
        b.iter(|| mathverse_core::algorithms::gcd(black_box(123456), black_box(789012)))
    });
}

fn bench_factorial(c: &mut Criterion) {
    c.bench_function("factorial(20)", |b| {
        b.iter(|| mathverse_core::algorithms::factorial(black_box(20)))
    });
}

fn bench_fibonacci(c: &mut Criterion) {
    c.bench_function("fibonacci(50)", |b| {
        b.iter(|| mathverse_core::algorithms::fibonacci(black_box(50)))
    });
}

fn bench_sieve(c: &mut Criterion) {
    c.bench_function("sieve(1000)", |b| {
        b.iter(|| mathverse_core::algorithms::sieve_of_eratosthenes(black_box(1000)))
    });
    c.bench_function("sieve(10000)", |b| {
        b.iter(|| mathverse_core::algorithms::sieve_of_eratosthenes(black_box(10000)))
    });
}

fn bench_is_prime(c: &mut Criterion) {
    c.bench_function("is_prime(997)", |b| {
        b.iter(|| mathverse_core::algorithms::is_prime(black_box(997)))
    });
    c.bench_function("is_prime_miller_rabin(997, 10)", |b| {
        b.iter(|| mathverse_core::algorithms::is_prime_miller_rabin(black_box(997), black_box(10)))
    });
}

fn bench_ops(c: &mut Criterion) {
    c.bench_function("lerp(0, 100, 0.5)", |b| {
        b.iter(|| mathverse_core::ops::lerp(black_box(0.0_f64), black_box(100.0), black_box(0.5)))
    });
    c.bench_function("smoothstep(0.5)", |b| {
        b.iter(|| mathverse_core::ops::smoothstep(black_box(0.5_f64)))
    });
    c.bench_function("hypot2(3, 4)", |b| {
        b.iter(|| mathverse_core::ops::hypot2(black_box(3.0_f64), black_box(4.0)))
    });
    c.bench_function("normalize([3,4,5,6,7,8,9,10])", |b| {
        b.iter(|| {
            mathverse_core::ops::normalize(black_box(&[
                3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            ]))
        })
    });
}

fn bench_precision(c: &mut Criterion) {
    c.bench_function("almost_eq(0.1+0.2, 0.3)", |b| {
        b.iter(|| {
            mathverse_core::precision::almost_eq(
                black_box(0.1_f64 + 0.2),
                black_box(0.3),
                black_box(1e-15),
            )
        })
    });
    c.bench_function("round_to(pi, 6)", |b| {
        b.iter(|| mathverse_core::precision::round_to(black_box(core::f64::consts::PI), black_box(6)))
    });
}

fn bench_combinatorics(c: &mut Criterion) {
    c.bench_function("binomial(30, 15)", |b| {
        b.iter(|| mathverse_core::algorithms::binomial(black_box(30), black_box(15)))
    });
    c.bench_function("catalan_number(15)", |b| {
        b.iter(|| mathverse_core::algorithms::catalan_number(black_box(15)))
    });
    c.bench_function("bell_number(15)", |b| {
        b.iter(|| mathverse_core::algorithms::bell_number(black_box(15)))
    });
    c.bench_function("partition_number(50)", |b| {
        b.iter(|| mathverse_core::algorithms::partition_number(black_box(50)))
    });
}

criterion_group!(
    benches,
    bench_gcd,
    bench_factorial,
    bench_fibonacci,
    bench_sieve,
    bench_is_prime,
    bench_ops,
    bench_precision,
    bench_combinatorics,
);
criterion_main!(benches);
