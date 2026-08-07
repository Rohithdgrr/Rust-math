use mathverse_equations::*;

fn main() {
    let roots = polynomial::solve_quadratic(1.0, -3.0, 2.0);
    println!("{:?}", roots);
    println!("Done.");
}
