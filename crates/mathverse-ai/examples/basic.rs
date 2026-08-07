use mathverse_ai::{Tensor, Adam};

fn main() {
    let mut opt = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0);
    let mut x = [5.0];
    for _ in 0..100 {
        let g = [2.0 * x[0]];
        opt.step(&mut x, &g);
    }
    println!("x = {:.6}", x[0]);
}
