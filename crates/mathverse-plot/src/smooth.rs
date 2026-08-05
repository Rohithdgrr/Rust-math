//! Smooth Bezier curve interpolation for polished line plots.

use crate::common::DataPoint;

/// Interpolation method for smooth curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Catmull-Rom spline (passes through all points, smooth).
    CatmullRom,
    /// Cubic Bezier with automatic control points.
    CubicBezier,
    /// Linear interpolation (no smoothing).
    Linear,
}

/// Configuration for smooth curve rendering.
#[derive(Debug, Clone)]
pub struct SmoothConfig {
    /// Interpolation method.
    pub interpolation: Interpolation,
    /// Tension for Catmull-Rom (0.0 = tight, 1.0 = loose). Default: 0.5.
    pub tension: f64,
    /// Number of subdivisions between points.
    pub subdivisions: usize,
}

impl Default for SmoothConfig {
    fn default() -> Self {
        Self {
            interpolation: Interpolation::CatmullRom,
            tension: 0.5,
            subdivisions: 20,
        }
    }
}

impl SmoothConfig {
    /// Create a new smooth config with Catmull-Rom interpolation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the interpolation method.
    pub fn with_interpolation(mut self, method: Interpolation) -> Self {
        self.interpolation = method;
        self
    }

    /// Set the tension (for Catmull-Rom).
    pub fn with_tension(mut self, tension: f64) -> Self {
        self.tension = tension.clamp(0.0, 1.0);
        self
    }

    /// Set the number of subdivisions between points.
    pub fn with_subdivisions(mut self, n: usize) -> Self {
        self.subdivisions = n.max(1);
        self
    }
}

/// Generate smooth SVG path data from a list of points.
///
/// Returns a string of SVG path commands (M, C, L) that can be used
/// in a `<path>` element.
pub fn smooth_path(points: &[DataPoint], config: &SmoothConfig) -> String {
    if points.is_empty() {
        return String::new();
    }
    if points.len() == 1 {
        return format!(
            "M{},{}",
            points[0].x, points[0].y
        );
    }

    match config.interpolation {
        Interpolation::Linear => linear_path(points),
        Interpolation::CatmullRom => catmull_rom_path(points, config.tension, config.subdivisions),
        Interpolation::CubicBezier => cubic_bezier_path(points, config.subdivisions),
    }
}

/// Generate smooth points (subsampled) from a list of control points.
pub fn smooth_points(points: &[DataPoint], config: &SmoothConfig) -> Vec<DataPoint> {
    if points.len() <= 1 {
        return points.to_vec();
    }

    match config.interpolation {
        Interpolation::Linear => points.to_vec(),
        Interpolation::CatmullRom => {
            catmull_rom_points(points, config.tension, config.subdivisions)
        }
        Interpolation::CubicBezier => cubic_bezier_points(points, config.subdivisions),
    }
}

/// Linear path (no smoothing).
fn linear_path(points: &[DataPoint]) -> String {
    let mut path = format!("M{},{}", points[0].x, points[0].y);
    for p in &points[1..] {
        path.push_str(&format!(" L{},{}", p.x, p.y));
    }
    path
}

/// Catmull-Rom spline path.
///
/// Reference: https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
fn catmull_rom_path(points: &[DataPoint], tension: f64, subdivisions: usize) -> String {
    let n = points.len();
    if n < 2 {
        return linear_path(points);
    }

    let alpha = 0.5; // centripetal
    let mut path = format!("M{},{}", points[0].x, points[0].y);

    for i in 0..n - 1 {
        let p0 = if i > 0 { points[i - 1] } else { points[i] };
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = if i + 2 < n { points[i + 2] } else { points[i + 1] };

        for j in 1..=subdivisions {
            let t = j as f64 / subdivisions as f64;
            let point = catmull_rom_interpolate(&p0, &p1, &p2, &p3, t, tension, alpha);
            path.push_str(&format!(" L{:.4},{:.4}", point.x, point.y));
        }
    }

    path
}

/// Catmull-Rom interpolation for a single point.
fn catmull_rom_interpolate(
    p0: &DataPoint,
    p1: &DataPoint,
    p2: &DataPoint,
    p3: &DataPoint,
    t: f64,
    tension: f64,
    alpha: f64,
) -> DataPoint {
    let t2 = t * t;
    let t3 = t2 * t;

    // Compute knot parameters
    let d1 = ((p1.x - p0.x).powf(2.0 * alpha) + (p1.y - p0.y).powf(2.0 * alpha)).powf(0.5 / alpha);
    let d2 = ((p2.x - p1.x).powf(2.0 * alpha) + (p2.y - p1.y).powf(2.0 * alpha)).powf(0.5 / alpha);
    let d3 = ((p3.x - p2.x).powf(2.0 * alpha) + (p3.y - p2.y).powf(2.0 * alpha)).powf(0.5 / alpha);

    let d1 = if d1 > 0.0 { d1 } else { 1.0 };
    let d2 = if d2 > 0.0 { d2 } else { 1.0 };
    let d3 = if d3 > 0.0 { d3 } else { 1.0 };

    // Tangent computation with tension
    let t1x = tension * (p2.x - p0.x) / (d1 + d2);
    let t1y = tension * (p2.y - p0.y) / (d1 + d2);
    let t2x = tension * (p3.x - p1.x) / (d2 + d3);
    let t2y = tension * (p3.y - p1.y) / (d2 + d3);

    // Hermite basis functions
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    DataPoint {
        x: h00 * p1.x + h10 * d2 * t1x + h01 * p2.x + h11 * d2 * t2x,
        y: h00 * p1.y + h10 * d2 * t1y + h01 * p2.y + h11 * d2 * t2y,
    }
}

/// Generate points along a Catmull-Rom spline.
fn catmull_rom_points(
    points: &[DataPoint],
    tension: f64,
    subdivisions: usize,
) -> Vec<DataPoint> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }

    let alpha = 0.5;
    let mut result = Vec::with_capacity(n * subdivisions);

    for i in 0..n - 1 {
        let p0 = if i > 0 { points[i - 1] } else { points[i] };
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = if i + 2 < n { points[i + 2] } else { points[i + 1] };

        for j in 0..subdivisions {
            let t = j as f64 / subdivisions as f64;
            result.push(catmull_rom_interpolate(&p0, &p1, &p2, &p3, t, tension, alpha));
        }
    }
    result.push(*points.last().expect("points is non-empty"));
    result
}

/// Cubic Bezier path with automatic control points.
fn cubic_bezier_path(points: &[DataPoint], subdivisions: usize) -> String {
    let n = points.len();
    if n < 2 {
        return linear_path(points);
    }
    if n == 2 {
        return format!("M{},{} L{},{}", points[0].x, points[0].y, points[1].x, points[1].y);
    }

    let mut path = format!("M{},{}", points[0].x, points[0].y);

    for i in 0..n - 1 {
        let p0 = points[i];
        let p1 = points[i + 1];

        // Compute control points
        let cp1 = if i > 0 {
            DataPoint {
                x: (points[i - 1].x + p1.x) / 2.0,
                y: (points[i - 1].y + p1.y) / 2.0,
            }
        } else {
            DataPoint {
                x: p0.x + (p1.x - p0.x) / 3.0,
                y: p0.y + (p1.y - p0.y) / 3.0,
            }
        };

        let cp2 = if i + 2 < n {
            DataPoint {
                x: (p0.x + points[i + 2].x) / 2.0,
                y: (p0.y + points[i + 2].y) / 2.0,
            }
        } else {
            DataPoint {
                x: p1.x - (p1.x - p0.x) / 3.0,
                y: p1.y - (p1.y - p0.y) / 3.0,
            }
        };

        // Subdivide the cubic Bezier
        for j in 1..=subdivisions {
            let t = j as f64 / subdivisions as f64;
            let point = cubic_bezier_eval(&p0, &cp1, &cp2, &p1, t);
            path.push_str(&format!(" L{:.4},{:.4}", point.x, point.y));
        }
    }

    path
}

/// Evaluate a cubic Bezier curve at parameter t.
fn cubic_bezier_eval(
    p0: &DataPoint,
    p1: &DataPoint,
    p2: &DataPoint,
    p3: &DataPoint,
    t: f64,
) -> DataPoint {
    let u = 1.0 - t;
    let u2 = u * u;
    let u3 = u2 * u;
    let t2 = t * t;
    let t3 = t2 * t;

    DataPoint {
        x: u3 * p0.x + 3.0 * u2 * t * p1.x + 3.0 * u * t2 * p2.x + t3 * p3.x,
        y: u3 * p0.y + 3.0 * u2 * t * p1.y + 3.0 * u * t2 * p2.y + t3 * p3.y,
    }
}

/// Generate points along a cubic Bezier curve.
fn cubic_bezier_points(points: &[DataPoint], subdivisions: usize) -> Vec<DataPoint> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }
    if n == 2 {
        return points.to_vec();
    }

    let mut result = Vec::with_capacity(n * subdivisions);

    for i in 0..n - 1 {
        let p0 = points[i];
        let p1 = points[i + 1];

        let cp1 = if i > 0 {
            DataPoint {
                x: (points[i - 1].x + p1.x) / 2.0,
                y: (points[i - 1].y + p1.y) / 2.0,
            }
        } else {
            DataPoint {
                x: p0.x + (p1.x - p0.x) / 3.0,
                y: p0.y + (p1.y - p0.y) / 3.0,
            }
        };

        let cp2 = if i + 2 < n {
            DataPoint {
                x: (p0.x + points[i + 2].x) / 2.0,
                y: (p0.y + points[i + 2].y) / 2.0,
            }
        } else {
            DataPoint {
                x: p1.x - (p1.x - p0.x) / 3.0,
                y: p1.y - (p1.y - p0.y) / 3.0,
            }
        };

        for j in 0..subdivisions {
            let t = j as f64 / subdivisions as f64;
            result.push(cubic_bezier_eval(&p0, &cp1, &cp2, &p1, t));
        }
    }
    result.push(*points.last().expect("points is non-empty"));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_path_linear() {
        let points = vec![DataPoint::new(0.0, 0.0), DataPoint::new(1.0, 1.0)];
        let config = SmoothConfig::new().with_interpolation(Interpolation::Linear);
        let path = smooth_path(&points, &config);
        assert!(path.starts_with("M"));
        assert!(path.contains("L"));
    }

    #[test]
    fn smooth_path_catmull_rom() {
        let points = vec![
            DataPoint::new(0.0, 0.0),
            DataPoint::new(1.0, 1.0),
            DataPoint::new(2.0, 0.0),
        ];
        let config = SmoothConfig::new();
        let path = smooth_path(&points, &config);
        assert!(path.starts_with("M"));
        // Should have many L commands (subdivisions)
        assert!(path.matches("L").count() > 2);
    }

    #[test]
    fn smooth_points_preserves_endpoints() {
        let points = vec![
            DataPoint::new(0.0, 0.0),
            DataPoint::new(1.0, 1.0),
            DataPoint::new(2.0, 0.0),
        ];
        let config = SmoothConfig::new().with_subdivisions(10);
        let smoothed = smooth_points(&points, &config);
        assert_eq!(smoothed.first().unwrap().x, 0.0);
        assert_eq!(smoothed.last().unwrap().x, 2.0);
    }

    #[test]
    fn catmull_rom_passes_through_endpoints() {
        let points = vec![
            DataPoint::new(0.0, 0.0),
            DataPoint::new(1.0, 1.0),
            DataPoint::new(2.0, 0.0),
        ];
        let config = SmoothConfig::new().with_subdivisions(10);
        let smoothed = smooth_points(&points, &config);
        // First and last points should be very close to originals
        assert!((smoothed[0].y - 0.0).abs() < 0.01);
        assert!((smoothed.last().unwrap().y - 0.0).abs() < 0.01);
    }
}
