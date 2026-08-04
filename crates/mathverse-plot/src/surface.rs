//! 3D surface / wireframe visualization via `mathverse-graphics` + `mathverse-vector`.
//!
//! Renders a parametric surface `z = f(x, y)` as an SVG wireframe
//! with perspective projection.

use mathverse_graphics::{look_at, perspective, rotation_y, translation};
use mathverse_matrix::Matrix;
use mathverse_vector::Vector;

use crate::common::{DataSeries, PlotConfig};
use crate::error::PlotResult;
use crate::svg::SvgPlot;

/// Configuration for a 3D surface plot.
#[derive(Debug, Clone)]
pub struct SurfaceConfig {
    /// Plot configuration (title, labels, dimensions).
    pub plot_config: PlotConfig,
    /// Camera distance from the origin.
    pub camera_distance: f64,
    /// Rotation angle around the Y axis (radians).
    pub rotation_y: f64,
    /// Field of view for perspective projection (radians).
    pub fovy: f64,
    /// Grid resolution (points per axis).
    pub resolution: usize,
}

impl SurfaceConfig {
    /// Create a new surface config with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plot_config: PlotConfig::new()
                .with_title("3D Surface".to_string()),
            camera_distance: 5.0,
            rotation_y: 0.4,
            fovy: 1.0,
            resolution: 40,
        }
    }

    /// Set the camera distance.
    #[must_use]
    pub fn with_camera_distance(mut self, d: f64) -> Self {
        self.camera_distance = d.max(0.1);
        self
    }

    /// Set the Y-axis rotation.
    #[must_use]
    pub fn with_rotation_y(mut self, angle: f64) -> Self {
        self.rotation_y = angle;
        self
    }

    /// Set the field of view.
    #[must_use]
    pub fn with_fovy(mut self, fovy: f64) -> Self {
        self.fovy = fovy.max(0.01);
        self
    }

    /// Set the grid resolution.
    #[must_use]
    pub fn with_resolution(mut self, res: usize) -> Self {
        self.resolution = res.max(2);
        self
    }
}

/// Project a 3D point onto the 2D image plane using perspective projection.
fn project_point(
    x: f64,
    y: f64,
    z: f64,
    camera_distance: f64,
    rotation_y_angle: f64,
    fovy: f64,
) -> (f64, f64) {
    let model = translation(0.0, 0.0, -camera_distance)
        .mul(&rotation_y(rotation_y_angle))
        .unwrap_or_else(|_| translation(0.0, 0.0, 0.0));

    let view = look_at(
        [0.0, camera_distance * 0.6, camera_distance],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    )
    .unwrap_or_else(|_| translation(0.0, 0.0, 0.0));

    let proj = perspective(
        fovy,
        1.0,
        0.1,
        camera_distance * 10.0,
    );

    let p = Vector::new(vec![x, y, z, 1.0]);
    let mvp = proj
        .mul(&view)
        .and_then(|vp| vp.mul(&model))
        .unwrap_or_else(|_| Matrix::from_rows(&[
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
        ]).unwrap());

    let transformed = mvp.mul_vec(&p).unwrap_or(p);

    let data = &transformed.data;
    let w = data[3];
    let x_ndc = if w.abs() > 1e-10 {
        data[0] / w
    } else {
        0.0
    };
    let y_ndc = if w.abs() > 1e-10 {
        data[1] / w
    } else {
        0.0
    };

    (x_ndc, y_ndc)
}

/// Render a 3D surface wireframe as SVG.
///
/// `f` computes `z = f(x, y)` for each grid point.
pub fn render_surface_wireframe<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    config: SurfaceConfig,
) -> PlotResult<String>
where
    F: Fn(f64, f64) -> f64,
{
    let res = config.resolution;
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;
    let dx = (x_max - x_min) / (res - 1) as f64;
    let dy = (y_max - y_min) / (res - 1) as f64;
    let camera_distance = config.camera_distance;
    let rotation_y_angle = config.rotation_y;
    let fovy = config.fovy;

    let mut plot = SvgPlot::new(config.plot_config);

    for i in 0..res {
        let x = x_min + i as f64 * dx;
        let mut row_points = Vec::with_capacity(res);
        for j in 0..res {
            let y = y_min + j as f64 * dy;
            let z = f(x, y);
            let (sx, sy) = project_point(x, y, z, camera_distance, rotation_y_angle, fovy);
            row_points.push(crate::DataPoint::new(sx, sy));
        }
        let series = DataSeries::new(format!("row_{}", i), row_points);
        plot.add_series(series);
    }

    for j in 0..res {
        let y = y_min + j as f64 * dy;
        let mut col_points = Vec::with_capacity(res);
        for i in 0..res {
            let x = x_min + i as f64 * dx;
            let z = f(x, y);
            let (sx, sy) = project_point(x, y, z, camera_distance, rotation_y_angle, fovy);
            col_points.push(crate::DataPoint::new(sx, sy));
        }
        let series = DataSeries::new(format!("col_{}", j), col_points);
        plot.add_series(series);
    }

    Ok(plot.generate())
}

/// Compute the surface mesh as a grid of (x, y, z) points.
pub fn compute_surface<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
) -> Vec<Vec<(f64, f64, f64)>>
where
    F: Fn(f64, f64) -> f64,
{
    let res = resolution.max(2);
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;
    let dx = (x_max - x_min) / (res - 1) as f64;
    let dy = (y_max - y_min) / (res - 1) as f64;

    (0..res)
        .map(|i| {
            let x = x_min + i as f64 * dx;
            (0..res)
                .map(|j| {
                    let y = y_min + j as f64 * dy;
                    let z = f(x, y);
                    (x, y, z)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_wireframe_renders_svg() {
        let config = SurfaceConfig::new();
        let svg = render_surface_wireframe(
            |x, y| x * x + y * y,
            (-2.0, 2.0),
            (-2.0, 2.0),
            config,
        )
        .unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn surface_wireframe_saddle() {
        let config = SurfaceConfig::new();
        let svg = render_surface_wireframe(
            |x, y| x * y,
            (-2.0, 2.0),
            (-2.0, 2.0),
            config,
        )
        .unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn compute_surface_has_correct_size() {
        let mesh = compute_surface(|x, y| x + y, (0.0, 1.0), (0.0, 1.0), 5);
        assert_eq!(mesh.len(), 5);
        assert_eq!(mesh[0].len(), 5);
    }

    #[test]
    fn compute_surface_identity() {
        let mesh = compute_surface(|x, y| x + y, (0.0, 1.0), (0.0, 1.0), 2);
        assert_eq!(mesh[0][0], (0.0, 0.0, 0.0));
        assert_eq!(mesh[1][1], (1.0, 1.0, 2.0));
    }
}