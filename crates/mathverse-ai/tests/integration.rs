use mathverse_ai::{Tensor, Adam};

#[test]
fn tensor_create_and_reshape() {
    let t = Tensor::zeros(&[2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
}

#[test]
fn adam_step_decreases_loss() {
    let mut opt = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0);
    let mut x = [5.0];
    for _ in 0..100 {
        let g = [2.0 * x[0]];
        opt.step(&mut x, &g);
    }
    assert!(x[0] < 0.01);
}
