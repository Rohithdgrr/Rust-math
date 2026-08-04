use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_matrix_mul_64(c: &mut Criterion) {
    let a = mathverse_matrix::Matrix::ones(64, 64);
    let b = mathverse_matrix::Matrix::ones(64, 64);

    c.bench_function("matrix_mul_64x64", |bencher| {
        bencher.iter(|| black_box(a.mul(&b).unwrap()))
    });
}

fn bench_matrix_mul_128(c: &mut Criterion) {
    let a = mathverse_matrix::Matrix::ones(128, 128);
    let b = mathverse_matrix::Matrix::ones(128, 128);

    c.bench_function("matrix_mul_128x128", |bencher| {
        bencher.iter(|| black_box(a.mul(&b).unwrap()))
    });
}

fn bench_matrix_det_16(c: &mut Criterion) {
    // Hilbert matrix (ill-conditioned, good stress test)
    let mut data = vec![0.0; 256];
    for i in 0..16 {
        for j in 0..16 {
            data[i * 16 + j] = 1.0 / ((i + j + 1) as f64);
        }
    }
    let m = mathverse_matrix::Matrix {
        rows: 16,
        cols: 16,
        data,
    };

    c.bench_function("matrix_det_16x16", |bencher| {
        bencher.iter(|| black_box(m.det().unwrap()))
    });
}

fn bench_matrix_inverse_8(c: &mut Criterion) {
    let m = mathverse_matrix::Matrix::identity(8);

    c.bench_function("matrix_inverse_8x8", |bencher| {
        bencher.iter(|| black_box(m.inverse().unwrap()))
    });
}

fn bench_matrix_transpose_256(c: &mut Criterion) {
    let m = mathverse_matrix::Matrix::ones(256, 256);

    c.bench_function("matrix_transpose_256x256", |bencher| {
        bencher.iter(|| black_box(m.transpose()))
    });
}

fn bench_matrix_vec_mul(c: &mut Criterion) {
    let m = mathverse_matrix::Matrix::ones(256, 256);
    let v = mathverse_vector::Vector::new(vec![1.0; 256]);

    c.bench_function("matrix_vec_mul_256", |bencher| {
        bencher.iter(|| black_box(m.mul_vec(&v).unwrap()))
    });
}

criterion_group!(
    matrix_benches,
    bench_matrix_mul_64,
    bench_matrix_mul_128,
    bench_matrix_det_16,
    bench_matrix_inverse_8,
    bench_matrix_transpose_256,
    bench_matrix_vec_mul,
);
criterion_main!(matrix_benches);
