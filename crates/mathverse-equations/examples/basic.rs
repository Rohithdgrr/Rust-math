//! Solves a quadratic equation and prints its roots.
use mathverse_equations::polynomial;

fn main() {
    let roots = polynomial::solve_quadratic(1.0, -3.0, 2.0);
    println!("{roots:?}");
    println!("Done.");
}
