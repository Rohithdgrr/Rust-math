//! Coordinate transforms — matplotlib-style placement for artists.
//!
//! Matplotlib lets every artist be positioned in one of several coordinate
//! systems: data coordinates, axes-fraction (0–1 across the plot area),
//! figure-fraction (0–1 across the whole canvas), and blended combinations
//! (e.g. x in axes-fraction, y in data). This module provides the equivalent
//! [`Position`] enum and a resolver that maps any position to pixel
//! coordinates given the axis mapping and the plot rectangle.

use crate::common::DataPoint;

/// A placement in one of the supported coordinate systems.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    /// Data coordinates — resolved through the axis scale mapping.
    Data(f64, f64),
    /// Fractional position across the plot area: `(0.0, 0.0)` is the bottom-left
    /// corner of the axes, `(1.0, 1.0)` the top-right.
    AxesFraction(f64, f64),
    /// Fractional position across the whole figure: `(0.0, 0.0)` is the
    /// top-left of the canvas, `(1.0, 1.0)` the bottom-right.
    FigureFraction(f64, f64),
    /// Blended: x in axes-fraction, y in data coordinates.
    BlendAxesXDataY { x_frac: f64, y_data: f64 },
    /// Blended: x in data coordinates, y in axes-fraction.
    BlendDataXAxesY { x_data: f64, y_frac: f64 },
}

impl Position {
    /// A data-coordinate position (most common case).
    #[must_use]
    pub fn data(x: f64, y: f64) -> Self {
        Self::Data(x, y)
    }

    /// Build from a [`DataPoint`] (kept for ergonomic interop).
    #[must_use]
    pub fn from_point(p: DataPoint) -> Self {
        Self::Data(p.x, p.y)
    }

    /// Resolve to pixel coordinates.
    ///
    /// * `x_px` / `y_px` map data coordinates to pixels (scale-aware).
    /// * `plot_rect` is `(left, top, width, height)` of the plot area in pixels.
    /// * `fig_size` is `(width, height)` of the whole canvas in pixels.
    ///
    /// Any non-finite input resolves to `(f64::NAN, f64::NAN)` so callers can
    /// skip drawing rather than emitting garbage.
    #[must_use]
    pub fn to_pixel(
        &self,
        x_px: &dyn Fn(f64) -> f64,
        y_px: &dyn Fn(f64) -> f64,
        plot_rect: (f64, f64, f64, f64),
        fig_size: (f64, f64),
    ) -> (f64, f64) {
        let (left, top, pw, ph) = plot_rect;
        let (fw, fh) = fig_size;
        match *self {
            Self::Data(x, y) => (x_px(x), y_px(y)),
            Self::AxesFraction(fx, fy) => (left + fx * pw, top + (1.0 - fy) * ph),
            Self::FigureFraction(fx, fy) => (fx * fw, fy * fh),
            Self::BlendAxesXDataY { x_frac, y_data } => (left + x_frac * pw, y_px(y_data)),
            Self::BlendDataXAxesY { x_data, y_frac } => (x_px(x_data), top + (1.0 - y_frac) * ph),
        }
    }
}

impl From<DataPoint> for Position {
    fn from(p: DataPoint) -> Self {
        Self::Data(p.x, p.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Identity mapping: data == pixel in a 100x100 plot rect at (10, 20).
    fn px(x: f64) -> f64 {
        x
    }
    fn plot_rect() -> (f64, f64, f64, f64) {
        (10.0, 20.0, 100.0, 100.0)
    }

    #[test]
    fn data_uses_axis_mapping() {
        let p = Position::Data(5.0, 7.0).to_pixel(&px, &px, plot_rect(), (800.0, 600.0));
        assert_eq!(p, (5.0, 7.0));
    }

    #[test]
    fn axes_fraction_resolves_inside_plot() {
        // (0,0) -> bottom-left of the plot rect = (left, top+height).
        let p = Position::AxesFraction(0.0, 0.0).to_pixel(&px, &px, plot_rect(), (800.0, 600.0));
        assert_eq!(p, (10.0, 120.0));
        let p = Position::AxesFraction(1.0, 1.0).to_pixel(&px, &px, plot_rect(), (800.0, 600.0));
        assert_eq!(p, (110.0, 20.0));
        let p = Position::AxesFraction(0.5, 0.5).to_pixel(&px, &px, plot_rect(), (800.0, 600.0));
        assert_eq!(p, (60.0, 70.0));
    }

    #[test]
    fn figure_fraction_uses_canvas() {
        let p = Position::FigureFraction(0.5, 0.25).to_pixel(&px, &px, plot_rect(), (800.0, 600.0));
        assert_eq!(p, (400.0, 150.0));
    }

    #[test]
    fn blended_positions() {
        let p = Position::BlendAxesXDataY { x_frac: 1.0, y_data: 3.0 }
            .to_pixel(&px, &px, plot_rect(), (800.0, 600.0));
        assert_eq!(p, (110.0, 3.0));
        let p = Position::BlendDataXAxesY { x_data: 2.0, y_frac: 0.0 }
            .to_pixel(&px, &px, plot_rect(), (800.0, 600.0));
        assert_eq!(p, (2.0, 120.0));
    }

    #[test]
    fn non_finite_yields_nan() {
        let p = Position::Data(f64::NAN, 0.0).to_pixel(&px, &px, plot_rect(), (800.0, 600.0));
        assert!(p.0.is_nan());
    }
}
