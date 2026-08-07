use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mathverse_geometry::{Point2, Circle, Triangle};

fn bench_circle_area(c: &mut Criterion) {
    c.bench_function("circle_area", |b| {
        let circle = Circle::new(Point2::new(0.0, 0.0), 5.0);
        b.iter(|| black_box(circle.area()));
    });
}

fn bench_triangle_area(c: &mut Criterion) {
    c.bench_function("triangle_area", |b| {
        let tri = Triangle::new(
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(0.0, 3.0),
        );
        b.iter(|| black_box(tri.area()));
    });
}

criterion_group!(benches, bench_circle_area, bench_triangle_area);
criterion_main!(benches);
