//! Shape transforms: rotation, scaling, translation for 2D shapes.

use super::shapes2d::{Circle, Ellipse, Point2, Polygon, Rectangle, Triangle};
use super::primitives2d::{Arc, LineSegment2, Sector};

// ---------------------------------------------------------------------------
// Transform2D trait
// ---------------------------------------------------------------------------

/// Generic transform for 2D shapes.
pub trait Transform2D {
    fn translate(self, dx: f64, dy: f64) -> Self;
    fn scale_xy(self, sx: f64, sy: f64) -> Self;
    fn rotate(self, angle: f64) -> Self;
    fn rotate_around(self, angle: f64, center: Point2) -> Self;
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl Transform2D for Point2 {
    fn translate(self, dx: f64, dy: f64) -> Self {
        self.translate(dx, dy)
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        Point2::new(self.x * sx, self.y * sy)
    }
    fn rotate(self, angle: f64) -> Self {
        self.rotate(angle)
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        self.rotate_around(angle, center)
    }
}

impl Transform2D for LineSegment2 {
    fn translate(self, dx: f64, dy: f64) -> Self {
        Self::new(self.a.translate(dx, dy), self.b.translate(dx, dy))
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        Self::new(
            Point2::new(self.a.x * sx, self.a.y * sy),
            Point2::new(self.b.x * sx, self.b.y * sy),
        )
    }
    fn rotate(self, angle: f64) -> Self {
        Self::new(self.a.rotate(angle), self.b.rotate(angle))
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        Self::new(self.a.rotate_around(angle, center), self.b.rotate_around(angle, center))
    }
}

impl Transform2D for Circle {
    fn translate(self, dx: f64, dy: f64) -> Self {
        Self::new(self.center.translate(dx, dy), self.radius)
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        let s = (sx.abs() * sy.abs()).sqrt();
        Self::new(Point2::new(self.center.x * sx, self.center.y * sy), self.radius * s)
    }
    fn rotate(self, _angle: f64) -> Self {
        self
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        Self::new(self.center.rotate_around(angle, center), self.radius)
    }
}

impl Transform2D for Rectangle {
    fn translate(self, dx: f64, dy: f64) -> Self {
        Self::new(self.x + dx, self.y + dy, self.width, self.height)
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        Self::new(self.x * sx, self.y * sy, self.width * sx.abs(), self.height * sy.abs())
    }
    fn rotate(self, angle: f64) -> Self {
        let c = self.centroid();
        let corners = [
            Point2::new(self.x, self.y),
            Point2::new(self.x + self.width, self.y),
            Point2::new(self.x + self.width, self.y + self.height),
            Point2::new(self.x, self.y + self.height),
        ];
        let _rotated: Vec<Point2> = corners.iter().map(|p| p.rotate_around(angle, c)).collect();
        self
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        let c = self.centroid();
        let new_center = c.rotate_around(angle, center);
        Self::new(
            new_center.x - self.width / 2.0,
            new_center.y - self.height / 2.0,
            self.width,
            self.height,
        )
    }
}

impl Transform2D for Triangle {
    fn translate(self, dx: f64, dy: f64) -> Self {
        Self::new(self.a.translate(dx, dy), self.b.translate(dx, dy), self.c.translate(dx, dy))
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        Self::new(
            Point2::new(self.a.x * sx, self.a.y * sy),
            Point2::new(self.b.x * sx, self.b.y * sy),
            Point2::new(self.c.x * sx, self.c.y * sy),
        )
    }
    fn rotate(self, angle: f64) -> Self {
        Self::new(self.a.rotate(angle), self.b.rotate(angle), self.c.rotate(angle))
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        Self::new(self.a.rotate_around(angle, center), self.b.rotate_around(angle, center), self.c.rotate_around(angle, center))
    }
}

impl Transform2D for Polygon {
    fn translate(self, dx: f64, dy: f64) -> Self {
        Self::new(self.points.into_iter().map(|p| p.translate(dx, dy)).collect())
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        Self::new(self.points.into_iter().map(|p| Point2::new(p.x * sx, p.y * sy)).collect())
    }
    fn rotate(self, angle: f64) -> Self {
        Self::new(self.points.into_iter().map(|p| p.rotate(angle)).collect())
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        Self::new(self.points.into_iter().map(|p| p.rotate_around(angle, center)).collect())
    }
}

impl Transform2D for Ellipse {
    fn translate(self, dx: f64, dy: f64) -> Self {
        Self::new(self.center.translate(dx, dy), self.rx, self.ry)
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        Self::new(Point2::new(self.center.x * sx, self.center.y * sy), self.rx * sx.abs(), self.ry * sy.abs())
    }
    fn rotate(self, _angle: f64) -> Self {
        self
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        Self::new(self.center.rotate_around(angle, center), self.rx, self.ry)
    }
}

impl Transform2D for Arc {
    fn translate(self, dx: f64, dy: f64) -> Self {
        Self::new(self.center.translate(dx, dy), self.radius, self.start, self.sweep)
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        let s = (sx.abs() * sy.abs()).sqrt();
        Self::new(Point2::new(self.center.x * sx, self.center.y * sy), self.radius * s, self.start, self.sweep)
    }
    fn rotate(self, angle: f64) -> Self {
        Self::new(self.center, self.radius, self.start + angle, self.sweep)
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        Self::new(self.center.rotate_around(angle, center), self.radius, self.start + angle, self.sweep)
    }
}

impl Transform2D for Sector {
    fn translate(self, dx: f64, dy: f64) -> Self {
        Self::new(self.center.translate(dx, dy), self.radius, self.start, self.sweep)
    }
    fn scale_xy(self, sx: f64, sy: f64) -> Self {
        let s = (sx.abs() * sy.abs()).sqrt();
        Self::new(Point2::new(self.center.x * sx, self.center.y * sy), self.radius * s, self.start, self.sweep)
    }
    fn rotate(self, angle: f64) -> Self {
        Self::new(self.center, self.radius, self.start + angle, self.sweep)
    }
    fn rotate_around(self, angle: f64, center: Point2) -> Self {
        Self::new(self.center.rotate_around(angle, center), self.radius, self.start + angle, self.sweep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::shapes2d::Polygon;

    fn pt(x: f64, y: f64) -> Point2 { Point2::new(x, y) }

    #[test]
    fn translate_triangle() {
        let t = Triangle::new(pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0));
        let moved = t.translate(2.0, 3.0);
        assert_eq!(moved.a, pt(2.0, 3.0));
        assert_eq!(moved.b, pt(3.0, 3.0));
        assert_eq!(moved.c, pt(2.0, 4.0));
    }

    #[test]
    fn rotate_polygon() {
        use mathverse_core::constants::PI;
        let sq = Polygon::new(vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 1.0)]);
        let rotated = sq.clone().rotate(PI);
        for (orig, rot) in sq.points.iter().zip(rotated.points.iter()) {
            let expected = orig.rotate(PI);
            assert!((rot.x - expected.x).abs() < 1e-12);
            assert!((rot.y - expected.y).abs() < 1e-12);
        }
    }

    #[test]
    fn circle_translate() {
        let c = Circle::new(pt(0.0, 0.0), 1.0);
        let moved = c.translate(3.0, 4.0);
        assert_eq!(moved.center, pt(3.0, 4.0));
        assert_eq!(moved.radius, 1.0);
    }
}
