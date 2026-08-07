//! Basic usage example.

use mathverse_geometry::{Circle, Point2, Triangle};

fn main() {
    let c = Circle::new(Point2::new(0.0, 0.0), 5.0);
    println!("Area: {:.2}", c.area());

    let t = Triangle::new(
        Point2::new(0.0, 0.0),
        Point2::new(4.0, 0.0),
        Point2::new(0.0, 3.0),
    );
    println!("Triangle area: {}", t.area());
}
