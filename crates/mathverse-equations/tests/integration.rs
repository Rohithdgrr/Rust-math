use mathverse_equations::*;

#[test]
fn quadratic_integrated() {
    let r = polynomial::solve_quadratic(1.0, -3.0, 2.0);
    assert_eq!(r.len(), 2);
}

#[test]
fn linear_system_integrated() {
    let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
    let b = vec![5.0, 7.0];
    let x = solve_linear_system(&a, &b).unwrap();
    assert!((x[0] - 1.6).abs() < 1e-10);
}
