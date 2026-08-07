use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mathverse_ai::Tensor;

fn bench_tensor_zeros(c: &mut Criterion) {
    c.bench_function("tensor_zeros", |b| b.iter(|| Tensor::zeros(&[100, 100])));
}

criterion_group!(benches, bench_tensor_zeros);
criterion_main!(benches);
