use mathverse_geometry::{Point2, Circle, Triangle, Polygon};

#[test]
fn circle_area_5() {
    let c = Circle::new(Point2::new(0.0, 0.0), 5.0);
    assert!((c.area() - 78.53981633974483).abs() < 1e-6);
}

#[test]
fn triangle_area_right() {
    let t = Triangle::new(
        Point2::new(0.0, 0.0),
        Point2::new(4.0, 0.0),
        Point2::new(0.0, 3.0),
    );
    assert_eq!(t.area(), 6.0);
}

#[test]
fn point_in_square() {
    let sq = Polygon::new(vec![
        Point2::new(0.0, 0.0),
        Point2::new(2.0, 0.0),
        Point2::new(2.0, 2.0),
        Point2::new(0.0, 2.0),
    ]);
    assert!(sq.contains(Point2::new(1.0, 1.0)));
}
