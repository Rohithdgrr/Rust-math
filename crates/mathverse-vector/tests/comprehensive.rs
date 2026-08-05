//! Comprehensive integration tests for mathverse-vector.
//!
//! Covers: real-world problems, edge cases, degenerate inputs, panic-safety
//! (untrusted input must not crash the process), and cross-feature parity.

use mathverse_vector::*;
use std::panic::catch_unwind;

const EPS: f64 = 1e-10;

fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "expected {a} to be within {tol} of {b}"
    );
}

fn assert_vec_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert_close(*x, *y, tol);
    }
}

fn panics<F: FnOnce() + std::panic::UnwindSafe>(f: F) -> bool {
    catch_unwind(f).is_err()
}

// ---------------------------------------------------------------------------
// Vector type
// ---------------------------------------------------------------------------

#[test]
fn vector_basics() {
    let v = Vector::new(vec![1.0, 2.0, 3.0]);
    assert_eq!(v.len(), 3);
    assert!(!v.is_empty());
    assert!(Vector::new(vec![]).is_empty());
    assert_eq!(Vector::zeros(4).data, vec![0.0; 4]);
    assert_eq!(v.get(1), 2.0);
    let mut w = v.clone();
    w.set(0, 9.0);
    assert_eq!(w.get(0), 9.0);
}

#[test]
fn vector_arithmetic() {
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    assert_eq!(a.add(&b).data, vec![5.0, 7.0, 9.0]);
    assert_eq!(a.sub(&b).data, vec![-3.0, -3.0, -3.0]);
    assert_eq!(a.scale(2.0).data, vec![2.0, 4.0, 6.0]);
    assert_eq!(a.scale(-1.0).data, vec![-1.0, -2.0, -3.0]);
    assert_eq!(a.scale(0.0).data, vec![0.0; 3]);
    assert_close(a.dot(&b), 32.0, EPS);
}

#[test]
fn vector_normalized() {
    let u = Vector::new(vec![3.0, 4.0]).normalized().unwrap();
    assert_vec_close(&u.data, &[0.6, 0.8], 1e-12);
    assert!(Vector::zeros(2).normalized().is_err());
    assert!(Vector::new(vec![0.0]).normalized().is_err());
    let one = Vector::new(vec![5.0]).normalized().unwrap();
    assert_vec_close(&one.data, &[1.0], 1e-12);
}

#[test]
fn vector_cross3() {
    let i = Vector::new(vec![1.0, 0.0, 0.0]);
    let j = Vector::new(vec![0.0, 1.0, 0.0]);
    let k = Vector::new(vec![0.0, 0.0, 1.0]);
    assert_vec_close(&i.cross3(&j).unwrap().data, &k.data, EPS);
    assert_vec_close(&j.cross3(&k).unwrap().data, &i.data, EPS);
    assert_vec_close(&k.cross3(&i).unwrap().data, &j.data, EPS);
    assert_vec_close(&i.cross3(&k).unwrap().data, &vec![0.0, -1.0, 0.0], EPS);
    assert!(i.cross3(&j).unwrap().dot(&i) < 1e-12);
    assert!(i.cross3(&j).unwrap().dot(&j) < 1e-12);
    assert!(i.cross3(&Vector::new(vec![1.0])).is_err());
    assert!(i.cross3(&Vector::new(vec![1.0, 2.0])).is_err());
}

#[test]
fn vector_out_of_bounds_panics_are_catchable() {
    let v = Vector::new(vec![1.0, 2.0]);
    assert!(panics(|| { v.get(5); }));
    assert!(panics(|| {
        let mut w = v.clone();
        w.set(9, 1.0);
    }));
}

// ---------------------------------------------------------------------------
// operations
// ---------------------------------------------------------------------------

#[test]
fn operations_basic() {
    assert_eq!(add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
    assert_eq!(sub(&[1.0, 2.0], &[3.0, 4.0]), vec![-2.0, -2.0]);
    assert_eq!(scale(&[1.0, 2.0], 0.5), vec![0.5, 1.0]);
    assert_eq!(negate(&[1.0, -2.0]), vec![-1.0, 2.0]);
    assert_eq!(add_scalar(&[1.0, 2.0], 10.0), vec![11.0, 12.0]);
    assert_eq!(hadamard(&[1.0, 2.0], &[3.0, 4.0]), vec![3.0, 8.0]);
    assert_eq!(outer(&[1.0, 2.0], &[3.0, 4.0]), vec![vec![3.0, 4.0], vec![6.0, 8.0]]);
    assert_close(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0, EPS);
}

#[test]
fn operations_lerp_endpoints() {
    let a = [1.0, 2.0];
    let b = [5.0, 6.0];
    assert_eq!(lerp(&a, &b, 0.0), a);
    assert_eq!(lerp(&a, &b, 1.0), b);
    assert_eq!(lerp(&a, &b, 0.5), vec![3.0, 4.0]);
}

#[test]
fn operations_cross_shape_guard() {
    assert_eq!(cross(&[1.0, 0.0], &[0.0, 1.0, 0.0]), Vec::<f64>::new());
    assert_eq!(cross(&[1.0, 0.0, 0.0, 1.0], &[0.0, 1.0, 0.0, 0.0]), Vec::<f64>::new());
}

#[test]
fn operations_normalize_zero_and_unit() {
    assert_eq!(normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
    assert_eq!(normalize(&[1.0, 0.0]), vec![1.0, 0.0]);
    let n = normalize(&[3.0, 4.0]);
    assert_close(magnitude(&n), 1.0, EPS);
}

#[test]
fn operations_mismatched_length_truncates() {
    assert_eq!(add(&[1.0, 2.0, 3.0], &[1.0]), vec![2.0]);
    assert_close(dot(&[1.0, 2.0, 3.0], &[5.0]), 5.0, EPS);
}

// ---------------------------------------------------------------------------
// norms
// ---------------------------------------------------------------------------

#[test]
fn norms_classic() {
    assert_close(l1(&[1.0, -2.0, 3.0]), 6.0, EPS);
    assert_close(l2(&[3.0, 4.0]), 5.0, EPS);
    assert_close(linf(&[-5.0, 3.0, 7.0]), 7.0, EPS);
    assert_eq!(l0(&[0.0, 1.0, -2.0, 0.0]), 2);
    assert_close(l_neg_inf(&[0.5, 2.0, 8.0]), 0.5, EPS);
}

#[test]
fn norms_lp_reduces_to_known_norms() {
    let v = [3.0, 4.0];
    assert_close(lp(&v, 1.0), l1(&v), EPS);
    assert_close(lp(&v, 2.0), l2(&v), EPS);
    assert_close(lp(&v, 2.0), 5.0, EPS);
    assert_close(lp(&[1.0, 1.0], 3.0), 2.0f64.powf(1.0 / 3.0), EPS);
    assert_close(lp(&[2.0, 2.0], 0.5), 8.0, EPS); // (2^0.5 + 2^0.5)^2 = 8
}

#[test]
fn norms_empty_and_zeros() {
    assert_eq!(l1(&[]), 0.0);
    assert_eq!(l2(&[]), 0.0);
    assert_eq!(linf(&[]), 0.0);
    assert_eq!(l0(&[]), 0);
    assert_eq!(l1(&[0.0, 0.0, 0.0]), 0.0);
}

// ---------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------

#[test]
fn geometry_angle_cases() {
    assert_close(angle(&[1.0, 0.0], &[1.0, 0.0]), 0.0, EPS);
    assert_close(angle(&[1.0, 0.0], &[0.0, 1.0]), std::f64::consts::FRAC_PI_2, EPS);
    assert_close(angle(&[1.0, 0.0], &[-1.0, 0.0]), std::f64::consts::PI, EPS);
    assert_close(angle(&[1.0, 1.0], &[1.0, 0.0]), std::f64::consts::FRAC_PI_4, EPS);
    assert_eq!(angle(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    assert_eq!(angle(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
}

#[test]
fn geometry_distance() {
    assert_close(distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0, EPS);
    assert_eq!(distance(&[1.0, 1.0], &[1.0, 1.0]), 0.0);
}

#[test]
fn geometry_projection_real_world() {
    // Gravity down a 30-degree ramp: g = 9.81 m/s^2 downward.
    let g = [0.0, -9.81];
    let ramp = [3.0f64.sqrt() / 2.0, -0.5]; // exact unit downhill direction, 30 deg
    let p = project(&g, &ramp);
    // Component along the ramp = g * sin(30deg).
    assert_close(magnitude(&p), 9.81 * 0.5, 1e-9);
    assert_close(angle(&p, &ramp), 0.0, EPS);
    assert_close(p[0].powi(2) + p[1].powi(2), (9.81 * 0.5f64).powi(2), 1e-9);

    // Projection onto zero vector is the zero vector.
    assert_eq!(project(&[1.0, 2.0], &[0.0, 0.0]), vec![0.0, 0.0]);

    // Orthogonal input projects to zero.
    assert_vec_close(&project(&[0.0, 1.0], &[1.0, 0.0]), &[0.0, 0.0], EPS);
}

#[test]
fn geometry_rejection_decomposes() {
    let a = [3.0, 4.0];
    let b = [1.0, 0.0];
    let p = project(&a, &b);
    let r = reject(&a, &b);
    // a = p + r and p ⟂ r
    assert_vec_close(&add(&p, &r), &a, EPS);
    assert_close(dot(&p, &r), 0.0, EPS);
    // Rejection onto orthogonal vector leaves input unchanged.
    assert_vec_close(&reject(&[0.0, 1.0], &[1.0, 0.0]), &[0.0, 1.0], EPS);
}

#[test]
fn geometry_triple_product_volume() {
    let a = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    let c = [0.0, 0.0, 1.0];
    assert_close(triple_product(&a, &b, &c), 1.0, EPS);
    // Coplanar vectors => zero volume.
    assert_close(triple_product(&a, &b, &[2.0, 3.0, 0.0]), 0.0, EPS);
    // Non-3D => 0.0 sentinel.
    assert_eq!(triple_product(&[1.0, 0.0], &b, &c), 0.0);
}

#[test]
fn geometry_gram_schmidt_orthonormal() {
    let mut v = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
    gram_schmidt(&mut v);
    for row in &v {
        assert_close(magnitude(row), 1.0, 1e-10);
    }
    assert_close(dot(&v[0], &v[1]), 0.0, 1e-10);
    assert_close(dot(&v[0], &v[2]), 0.0, 1e-10);
    assert_close(dot(&v[1], &v[2]), 0.0, 1e-10);
}

#[test]
fn geometry_gram_schmidt_handles_dependent_inputs() {
    // rank-deficient: third vector = 2x first. Must not produce NaN/inf.
    let mut v = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 0.0]];
    gram_schmidt(&mut v);
    assert!(v.iter().all(|r| r.iter().all(|x| x.is_finite())));
    assert_close(dot(&v[0], &v[1]), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// linear algebra
// ---------------------------------------------------------------------------

#[test]
fn linalg_mat_vec_mul() {
    let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    assert_eq!(mat_vec_mul(&m, &[5.0, 6.0]), vec![17.0, 39.0]);
    assert_eq!(mat_vec_mul(&vec![vec![2.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]], &[3.0, 4.0, 5.0]), vec![6.0, 8.0]);
}

#[test]
fn linalg_determinants() {
    assert_close(det2x2(&[[1.0, 2.0], [3.0, 4.0]]), -2.0, EPS);
    assert_close(det3x3(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]]), 27.0, EPS);
    // Singular 3x3.
    assert_close(det3x3(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]), 0.0, 1e-9);
    // Identity.
    assert_close(det3x3(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 1.0, EPS);
}

#[test]
fn linalg_rank_full() {
    assert_eq!(rank(&[vec![1.0, 0.0], vec![0.0, 1.0]]), 2);
    assert_eq!(rank(&[vec![1.0, 2.0], vec![3.0, 4.0]]), 2);
    assert_eq!(rank(&[vec![2.0, 4.0], vec![1.0, 3.0]]), 2);
}

#[test]
fn linalg_rank_deficient() {
    // Row 2 = 1.5 * row 1  => rank 1
    assert_eq!(rank(&[vec![2.0, 4.0], vec![3.0, 6.0]]), 1);
    assert_eq!(rank(&[vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]]), 1);
    assert_eq!(rank(&[vec![1.0, 0.0], vec![0.0, 0.0], vec![2.0, 0.0]]), 1);
    // All zeros.
    assert_eq!(rank(&[vec![0.0, 0.0], vec![0.0, 0.0]]), 0);
    // Empty / single row / single column.
    assert_eq!(rank(&[] as &[Vec<f64>]), 0);
    assert_eq!(rank(&[vec![5.0]]), 1);
    assert_eq!(rank(&[vec![0.0]]), 0);
}

#[test]
fn linalg_rank_real_world() {
    // Points on the plane z = 2x + 3y: offsets (x, y, z) span a 2D subspace.
    let pts = vec![
        vec![1.0, 0.0, 2.0],
        vec![0.0, 1.0, 3.0],
        vec![2.0, 2.0, 10.0],
        vec![3.0, 1.0, 9.0],
    ];
    assert_eq!(rank(&pts), 2);
    // Four random-independent points in 3D span R^3.
    let pts3 = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 3.0, 5.0],
    ];
    assert_eq!(rank(&pts3), 3);
}

#[test]
fn linalg_is_orthogonal() {
    assert!(is_orthogonal(&[vec![1.0, 0.0], vec![0.0, 1.0]], 1e-10));
    assert!(!is_orthogonal(&[vec![1.0, 1.0], vec![1.0, 0.0]], 1e-10));
    assert!(is_orthogonal(&[vec![1.0, 2.0]], 1e-10));
    assert!(is_orthogonal(&[] as &[Vec<f64>], 1e-10));
}

// ---------------------------------------------------------------------------
// statistics
// ---------------------------------------------------------------------------

#[test]
fn stats_mean_variance_std() {
    assert_close(mean(&[1.0, 2.0, 3.0, 4.0]), 2.5, EPS);
    assert_close(variance(&[2.0, 2.0, 2.0]), 0.0, EPS);
    assert_close(variance(&[1.0, 2.0, 3.0]), 2.0 / 3.0, EPS); // population variance
    assert_close(std_dev(&[1.0, 2.0, 3.0]), (2.0f64 / 3.0).sqrt(), EPS);
    assert_eq!(variance(&[] as &[f64]).is_nan(), true); // documented degenerate
}

#[test]
fn stats_covariance_correlation() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [2.0, 4.0, 6.0, 8.0];
    assert_close(covariance(&a, &b), 2.5, EPS);
    assert_close(correlation(&a, &b), 1.0, EPS);
    assert_close(correlation(&a, &[-2.0, -4.0, -6.0, -8.0]), -1.0, EPS);
    assert_close(correlation(&a, &[1.0, 1.0, 1.0, 1.0]), 0.0, EPS); // zero variance
    assert_close(correlation(&[0.0, 0.0, 0.0], &a), 0.0, EPS);
}

#[test]
fn stats_real_world() {
    // Temperature (C) vs ice-cream sales — monotonic increasing.
    let temp = [15.0, 18.0, 21.0, 24.0, 27.0];
    let sales = [100.0, 120.0, 150.0, 190.0, 230.0];
    let c = correlation(&temp, &sales);
    assert!(c > 0.99, "expected strong positive correlation, got {c}");
}

// ---------------------------------------------------------------------------
// distance
// ---------------------------------------------------------------------------

#[test]
fn distance_metrics() {
    assert_close(euclidean(&[0.0, 0.0], &[3.0, 4.0]), 5.0, EPS);
    assert_close(manhattan(&[1.0, 2.0], &[4.0, 6.0]), 7.0, EPS);
    assert_close(chebyshev(&[1.0, 2.0], &[4.0, 6.0]), 4.0, EPS);
    assert_eq!(euclidean(&[0.0], &[0.0]), 0.0);
}

#[test]
fn distance_cosine_edge_cases() {
    assert_close(cosine(&[1.0, 0.0], &[1.0, 0.0]), 0.0, EPS);
    assert_close(cosine(&[1.0, 0.0], &[0.0, 1.0]), 1.0, EPS);
    assert_close(cosine(&[1.0, 0.0], &[-1.0, 0.0]), 2.0, EPS);
    assert_close(cosine(&[1.0, 2.0], &[2.0, 4.0]), 0.0, EPS);
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0); // zero vector guard
    assert_eq!(cosine(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
}

#[test]
fn distance_minkowski_family() {
    let a = [0.0, 0.0];
    let b = [3.0, 4.0];
    assert_close(minkowski(&a, &b, 1.0), manhattan(&a, &b), EPS);
    assert_close(minkowski(&a, &b, 2.0), euclidean(&a, &b), EPS);
    assert_close(minkowski(&a, &b, 3.0), (3.0f64.powi(3) + 4.0f64.powi(3)).powf(1.0 / 3.0), EPS);
}

#[test]
fn distance_mahalanobis() {
    let cov_inv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert_close(mahalanobis(&[0.0, 0.0], &[3.0, 4.0], &cov_inv), 5.0, EPS);
    // Axis-aligned anisotropic covariance: scale x by 2.
    let cov_inv = vec![vec![0.25, 0.0], vec![0.0, 1.0]]; // scale x by 2 => 1/s^2
    assert_close(mahalanobis(&[0.0, 0.0], &[4.0, 0.0], &cov_inv), 2.0, EPS);
    assert_eq!(mahalanobis(&[0.0, 0.0], &[0.0, 0.0], &cov_inv), 0.0);
}

// ---------------------------------------------------------------------------
// utils
// ---------------------------------------------------------------------------

#[test]
fn utils_linspace() {
    assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    assert_eq!(linspace(0.0, 1.0, 1), vec![0.0]);
    assert_eq!(linspace(5.0, 9.0, 0), Vec::<f64>::new());
    assert_eq!(linspace(-1.0, 1.0, 3), vec![-1.0, 0.0, 1.0]);
}

#[test]
fn utils_zeros_ones() {
    assert_eq!(zeros(3), vec![0.0; 3]);
    assert_eq!(ones(3), vec![1.0; 3]);
    assert_eq!(zeros(0), Vec::<f64>::new());
}

#[test]
fn utils_random_bounds_and_determinism() {
    let r = random(1000, -5.0, 5.0);
    assert_eq!(r.len(), 1000);
    assert!(r.iter().all(|x| *x >= -5.0 && *x < 5.0));
    assert_eq!(random(10, 0.0, 1.0), random(10, 0.0, 1.0)); // deterministic
}

#[test]
fn utils_argmax_argmin() {
    assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
    assert_eq!(argmin(&[1.0, 5.0, 3.0]), 0);
    assert_eq!(argmax(&[3.0, 1.0, 3.0]), 2); // ties -> last (max_by semantics)
    assert_eq!(argmin(&[-1.0, -2.0, -0.5]), 1);
    assert!(panics(|| { argmax(&[] as &[f64]); }));
    assert!(panics(|| { argmin(&[] as &[f64]); }));
}

#[test]
fn utils_min_max_sum_prod() {
    assert_eq!(max(&[1.0, 5.0, 3.0]), 5.0);
    assert_eq!(min(&[1.0, 5.0, 3.0]), 1.0);
    assert_eq!(sum(&[1.0, 2.0, 3.0]), 6.0);
    assert_eq!(prod(&[1.0, 2.0, 3.0, 4.0]), 24.0);
    assert_eq!(prod(&[] as &[f64]), 1.0);
    assert_eq!(max(&[] as &[f64]), f64::NEG_INFINITY);
    assert_eq!(min(&[] as &[f64]), f64::INFINITY);
}

#[test]
fn utils_clip() {
    let mut v = vec![-2.0, 0.5, 3.0];
    clip(&mut v, -1.0, 1.0);
    assert_eq!(v, vec![-1.0, 0.5, 1.0]);
}

#[test]
fn utils_reverse() {
    assert_eq!(reverse(&[1.0, 2.0, 3.0]), vec![3.0, 2.0, 1.0]);
    assert_eq!(reverse(&[] as &[f64]), Vec::<f64>::new());
}

// ---------------------------------------------------------------------------
// Non-finite / adversarial input (security): must never panic, must stay sane.
// ---------------------------------------------------------------------------

#[test]
fn adversarial_infinity_does_not_panic() {
    let v = [f64::INFINITY, 1.0];
    assert_eq!(l2(&v), f64::INFINITY);
    assert_eq!(linf(&v), f64::INFINITY);
    assert!(l1(&v).is_infinite());
    assert!(l_neg_inf(&v).is_finite());
    // normalize of an infinite component yields NaN (inf/inf) — documented
    // numerical limit of the naive divide-by-magnitude, but must not panic.
    let n = normalize(&v);
    assert!(n.iter().all(|x| x.is_nan() || x.is_finite()));
    let _ = Vector::new(vec![f64::INFINITY, 1.0]).normalized();
}

#[test]
fn adversarial_lp_limits() {
    // p -> 0 behaves like count of nonzero elements (L0), huge but defined.
    let l0ish = lp(&[1.0, 1.0, 1.0], 1e-9);
    assert!(l0ish > 1.0, "expected huge value for tiny p, got {l0ish}");
    // Moderate-large p pulls the norm toward the max element (5).
    let linfish = lp(&[1.0, 5.0, 3.0], 100.0);
    assert!((linfish - 5.0).abs() < 1e-1, "expected ~5, got {linfish}");
    // p = 0 exactly: each |x|^0 = 1, total n^inf = inf. Must not panic.
    let p0 = lp(&[1.0, 2.0], 0.0);
    assert!(p0.is_infinite());
    // Very large p (e.g. 1e6) overflows the sum-of-powers before the root is
    // taken (5^1e6 = inf) — known numerical limit of the naive Lp, not a panic.
    let huge = lp(&[1.0, 5.0, 3.0], 1e6);
    assert!(huge.is_finite() || huge.is_infinite());
}

#[test]
fn adversarial_length_mismatch_no_panic() {
    // All slice-pair functions zip-truncate and must not panic on mismatch.
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [1.0, 2.0];
    let _ = dot(&a, &b);
    let _ = distance(&a, &b);
    let _ = manhattan(&a, &b);
    let _ = chebyshev(&a, &b);
    let _ = minkowski(&a, &b, 2.0);
    let _ = angle(&a, &b);
    let _ = project(&a, &b);
    let _ = reject(&a, &b);
    let _ = covariance(&a, &b);
    let _ = correlation(&a, &b);
    let _ = lerp(&a, &b, 0.5);
    let _ = hadamard(&a, &b);
    assert!(true);
}

#[test]
fn adversarial_covariance_empty_pair_no_panic() {
    let e: &[f64] = &[];
    assert!(mean(&e).is_nan());
    assert!(correlation(&e, &e).is_nan());
    let _ = covariance(&e, &e);
}

#[test]
fn adversarial_huge_magnitude_stays_finite_or_inf() {
    // Very large but finite coordinates. Manhattan and Chebyshev stay finite;
    // the sum-of-squares path (euclidean/l2/magnitude) overflows to inf —
    // a documented numerical limit of the naive formula. No NaN, no panic.
    let a = [1e200, 1e200];
    let b = [-1e200, -1e200];
    assert!(manhattan(&a, &b).is_finite());
    assert!(chebyshev(&a, &b).is_finite());
    let d = euclidean(&a, &b);
    assert!(d.is_finite() || d.is_infinite(), "got NaN");
    assert!(!d.is_nan());
}

#[test]
fn adversarial_zero_cov_inverse_no_panic() {
    // Diagonal of zero -> sqrt(0) = 0, not NaN.
    let cov_inv = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    assert_eq!(mahalanobis(&[1.0, 2.0], &[1.0, 2.0], &cov_inv), 0.0);
}
