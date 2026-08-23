//! Criterion benchmarks for the machine-learning crate: kNN, k-means,
//! decision tree, random forest, linear regression, and PCA.

// `criterion_group!` generates undocumented items, which trips the
// workspace-wide `missing_docs` lint; silence it for this generated code.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mathverse_machine_learning::forest::RandomForest;
use mathverse_machine_learning::kmeans;
use mathverse_machine_learning::knn;
use mathverse_machine_learning::linear;
use mathverse_machine_learning::pca::PCA;
use mathverse_machine_learning::tree::DecisionTree;

/// Benchmark kNN classification on 500 training rows.
fn bench_knn(c: &mut Criterion) {
    let train_x: Vec<Vec<f64>> = (0..500)
        .map(|i| vec![f64::from(i), f64::from(i) * 0.5])
        .collect();
    let train_y: Vec<f64> = (0..500)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let test_x = vec![vec![100.0, 50.0]];
    c.bench_function("knn_classify_500", |b| {
        b.iter(|| {
            knn::classify(
                black_box(&train_x),
                black_box(&train_y),
                black_box(&test_x),
                3,
            )
        });
    });
}

/// Benchmark k-means clustering (3 clusters, 1000 points).
fn bench_kmeans(c: &mut Criterion) {
    let x: Vec<Vec<f64>> = (0..1000)
        .map(|i| vec![f64::from(i), f64::from(i * 2)])
        .collect();
    c.bench_function("kmeans_3_clusters_1000", |b| {
        b.iter(|| kmeans::kmeans(black_box(&x), 3, 10, 1e-6));
    });
}

/// Benchmark decision tree prediction.
fn bench_decision_tree(c: &mut Criterion) {
    let x: Vec<Vec<f64>> = (0..500)
        .map(|i| vec![f64::from(i), f64::from(i) * 0.5])
        .collect();
    let y: Vec<f64> = (0..500)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let mut tree = DecisionTree::new(5, 2);
    tree.fit(&x, &y);
    let test = vec![vec![100.0, 50.0]];
    c.bench_function("decision_tree_predict", |b| {
        b.iter(|| tree.predict(black_box(&test)));
    });
}

/// Benchmark random forest prediction.
fn bench_random_forest(c: &mut Criterion) {
    let x: Vec<Vec<f64>> = (0..500)
        .map(|i| vec![f64::from(i), f64::from(i) * 0.5])
        .collect();
    let y: Vec<f64> = (0..500)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();
    let mut rf = RandomForest::new(50, 5, 2);
    rf.fit(&x, &y);
    let test = vec![vec![100.0, 50.0]];
    c.bench_function("random_forest_predict", |b| {
        b.iter(|| rf.predict(black_box(&test)));
    });
}

/// Benchmark linear regression fit + predict on 1000 rows.
fn bench_linear_regression(c: &mut Criterion) {
    let x: Vec<Vec<f64>> = (0..1000).map(|i| vec![f64::from(i)]).collect();
    let y: Vec<f64> = (0..1000).map(|i| f64::from(i) * 2.0 + 1.0).collect();
    let test = vec![vec![500.0]];
    c.bench_function("linear_regression_fit_predict", |b| {
        b.iter(|| {
            let result = linear::fit(black_box(&x), black_box(&y)).unwrap();
            linear::predict(black_box(&test), &result.coefficients, result.intercept)
        });
    });
}

/// Benchmark PCA fit_transform on a 500x3 dataset.
fn bench_pca(c: &mut Criterion) {
    let x: Vec<Vec<f64>> = (0..500)
        .map(|i| vec![f64::from(i), f64::from(i) * 0.5, f64::from(i) * 0.3])
        .collect();
    c.bench_function("pca_fit_transform_500x3", |b| {
        b.iter(|| {
            let mut pca = PCA::new(2);
            pca.fit_transform(black_box(&x))
        });
    });
}

criterion_group!(
    benches,
    bench_knn,
    bench_kmeans,
    bench_decision_tree,
    bench_random_forest,
    bench_linear_regression,
    bench_pca,
);
criterion_main!(benches);
