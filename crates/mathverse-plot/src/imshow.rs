//! `imshow` — render an arbitrary 2D array as a colormapped image, the
//! matplotlib analogue of `plt.imshow`.
//!
//! Supports nearest / bilinear interpolation, explicit `vmin`/`vmax`,
//! data-space extent, alpha, and top/bottom origin. The image is emitted as
//! colored cells in data coordinates so every backend (SVG, PNG, PDF) can
//! draw it with the same data.

use crate::error::{PlotError, PlotResult};
use crate::heatmap::Colormap;
use crate::style::Color;

/// Interpolation applied when resampling the source grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Each output cell takes the nearest source value.
    #[default]
    Nearest,
    /// Bilinear interpolation between source values.
    Bilinear,
}

/// Whether row 0 of the grid is the top (`Upper`, matplotlib default) or the
/// bottom (`Lower`) of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageOrigin {
    /// Row 0 is the top edge.
    #[default]
    Upper,
    /// Row 0 is the bottom edge.
    Lower,
}

/// A 2D image to be drawn in data coordinates.
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Row-major value grid (`grid[r][c]`).
    pub grid: Vec<Vec<f64>>,
    /// Colormap sampling function.
    pub colormap: Colormap,
    /// Explicit color bounds (None = auto from data).
    pub vmin: Option<f64>,
    /// Explicit color bounds (None = auto from data).
    pub vmax: Option<f64>,
    /// Data-space horizontal extent `(left, right)`.
    pub x_extent: (f64, f64),
    /// Data-space vertical extent `(bottom, top)`.
    pub y_extent: (f64, f64),
    /// Resampling mode.
    pub interpolation: Interpolation,
    /// Opacity in `[0, 1]`.
    pub alpha: f64,
    /// Orientation of row 0.
    pub origin: ImageOrigin,
}

/// One colored cell of a resampled image, in data coordinates
/// (`x_lo, y_lo, x_hi, y_hi`, ascending y).
pub type ImageCell = ((f64, f64, f64, f64), Color);

impl ImageData {
    /// Create an image from a 2D grid.
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty or ragged grids.
    pub fn new(grid: Vec<Vec<f64>>, colormap: Colormap) -> PlotResult<Self> {
        if grid.is_empty() || grid[0].is_empty() {
            return Err(PlotError::InvalidData("empty image grid".into()));
        }
        let cols = grid[0].len();
        if grid.iter().any(|row| row.len() != cols) {
            return Err(PlotError::InvalidData("ragged image grid".into()));
        }
        let (rows, cols) = (grid.len(), grid[0].len());
        Ok(Self {
            grid,
            colormap,
            vmin: None,
            vmax: None,
            x_extent: (0.0, cols as f64),
            y_extent: (0.0, rows as f64),
            interpolation: Interpolation::Nearest,
            alpha: 1.0,
            origin: ImageOrigin::Upper,
        })
    }

    /// Set explicit color bounds (`vmin`, `vmax`).
    #[must_use]
    pub fn with_vmin_vmax(mut self, vmin: f64, vmax: f64) -> Self {
        self.vmin = Some(vmin);
        self.vmax = Some(vmax);
        self
    }

    /// Set the data-space extent `(xmin, xmax, ymin, ymax)` (matplotlib
    /// `extent=[left, right, bottom, top]`).
    #[must_use]
    pub fn with_extent(mut self, xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Self {
        self.x_extent = (xmin, xmax);
        self.y_extent = (ymin, ymax);
        self
    }

    /// Set the interpolation mode.
    #[must_use]
    pub fn with_interpolation(mut self, interpolation: Interpolation) -> Self {
        self.interpolation = interpolation;
        self
    }

    /// Set the opacity.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set whether row 0 is the top or the bottom of the image.
    #[must_use]
    pub fn with_origin(mut self, origin: ImageOrigin) -> Self {
        self.origin = origin;
        self
    }

    /// Number of rows in the source grid.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.grid.len()
    }

    /// Number of columns in the source grid.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.grid[0].len()
    }

    /// Auto color bounds (min, max across the grid).
    #[must_use]
    pub fn bounds(&self) -> (f64, f64) {
        let min = self
            .grid
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .grid
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }

    /// The effective color bounds (explicit `vmin`/`vmax` or auto).
    #[must_use]
    pub fn color_bounds(&self) -> (f64, f64) {
        let (min, max) = self.bounds();
        (
            self.vmin.unwrap_or(min),
            self.vmax.unwrap_or(max),
        )
    }

    /// Resample to `tw × th` output cells and color them.
    ///
    /// `tw`/`th` are the maximum output dimensions; the source grid is never
    /// upsampled beyond its own resolution in Nearest mode, but Bilinear mode
    /// may interpolate to any target size.
    #[must_use]
    pub fn resample(&self, tw: usize, th: usize) -> Vec<Vec<f64>> {
        let (rows, cols) = (self.rows(), self.cols());
        let tw = tw.max(1);
        let th = th.max(1);
        let (mut ow, mut oh) = (cols, rows);
        match self.interpolation {
            Interpolation::Nearest => {
                ow = ow.min(tw);
                oh = oh.min(th);
            }
            Interpolation::Bilinear => {
                ow = tw;
                oh = th;
            }
        }
        let mut out = vec![vec![0.0; ow]; oh];
        for r in 0..oh {
            for c in 0..ow {
                out[r][c] = self.sample(
                    (c as f64 + 0.5) * cols as f64 / ow as f64 - 0.5,
                    (r as f64 + 0.5) * rows as f64 / oh as f64 - 0.5,
                );
            }
        }
        out
    }

    fn sample(&self, x: f64, y: f64) -> f64 {
        let (rows, cols) = (self.rows(), self.cols());
        match self.interpolation {
            Interpolation::Nearest => {
                let c = (x.round() as isize).clamp(0, cols as isize - 1) as usize;
                let r = (y.round() as isize).clamp(0, rows as isize - 1) as usize;
                self.grid[r][c]
            }
            Interpolation::Bilinear => {
                let c0 = (x.floor() as isize).clamp(0, cols as isize - 1) as usize;
                let r0 = (y.floor() as isize).clamp(0, rows as isize - 1) as usize;
                let c1 = (c0 + 1).min(cols - 1);
                let r1 = (r0 + 1).min(rows - 1);
                let fx = (x - c0 as f64).clamp(0.0, 1.0);
                let fy = (y - r0 as f64).clamp(0.0, 1.0);
                let v00 = self.grid[r0][c0];
                let v01 = self.grid[r0][c1];
                let v10 = self.grid[r1][c0];
                let v11 = self.grid[r1][c1];
                let top = v00 + (v01 - v00) * fx;
                let bottom = v10 + (v11 - v10) * fx;
                top + (bottom - top) * fy
            }
        }
    }

    /// Produce colored cells in data coordinates for rendering.
    ///
    /// `max_cells_side` caps the longest output dimension (protects against
    /// huge grids exploding into millions of SVG elements).
    #[must_use]
    pub fn cells(&self, max_cells_side: usize) -> Vec<ImageCell> {
        let (rows, cols) = (self.rows(), self.cols());
        let tw = cols.min(max_cells_side.max(1));
        let th = rows.min(max_cells_side.max(1));
        let sampled = self.resample(tw, th);
        let (lo, hi) = self.color_bounds();
        let (x0, x1) = self.x_extent;
        let (y0, y1) = self.y_extent;
        let span = hi - lo;
        let t = |v: f64| {
            if span.abs() < f64::EPSILON {
                0.5
            } else {
                ((v - lo) / span).clamp(0.0, 1.0)
            }
        };
        let mut out = Vec::with_capacity(tw * th);
        for r in 0..th {
            let y_lo = if self.origin == ImageOrigin::Upper {
                y1 - (r + 1) as f64 / th as f64 * (y1 - y0)
            } else {
                y0 + r as f64 / th as f64 * (y1 - y0)
            };
            let y_hi = if self.origin == ImageOrigin::Upper {
                y1 - r as f64 / th as f64 * (y1 - y0)
            } else {
                y0 + (r + 1) as f64 / th as f64 * (y1 - y0)
            };
            for c in 0..tw {
                let x_lo = x0 + c as f64 / tw as f64 * (x1 - x0);
                let x_hi = x0 + (c + 1) as f64 / tw as f64 * (x1 - x0);
                let color = (self.colormap)(t(sampled[r][c]));
                out.push(((x_lo, y_lo.min(y_hi), x_hi, y_lo.max(y_hi)), color));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::viridis;

    fn checker() -> Vec<Vec<f64>> {
        vec![vec![0.0, 1.0], vec![1.0, 0.0]]
    }

    #[test]
    fn new_validates_grid() {
        assert!(ImageData::new(vec![], viridis).is_err());
        assert!(ImageData::new(vec![vec![1.0], vec![1.0, 2.0]], viridis).is_err());
        let img = ImageData::new(checker(), viridis).unwrap();
        assert_eq!(img.rows(), 2);
        assert_eq!(img.cols(), 2);
    }

    #[test]
    fn bounds_and_color_bounds() {
        let img = ImageData::new(checker(), viridis).unwrap();
        assert_eq!(img.bounds(), (0.0, 1.0));
        let clamped = img.with_vmin_vmax(0.0, 10.0);
        assert_eq!(clamped.color_bounds(), (0.0, 10.0));
    }

    #[test]
    fn nearest_resample_caps_dimensions() {
        let img = ImageData::new(checker(), viridis).unwrap();
        let s = img.resample(1, 1);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].len(), 1);
        // 2x2 source, nearest to 2x2 keeps identity.
        let s = img.resample(4, 4);
        assert_eq!((s.len(), s[0].len()), (2, 2));
    }

    #[test]
    fn bilinear_resamples() {
        let img = ImageData::new(checker(), viridis)
            .unwrap()
            .with_interpolation(Interpolation::Bilinear);
        let s = img.resample(4, 4);
        assert_eq!((s.len(), s[0].len()), (4, 4));
        // Top-left (0,0) -> 0, bottom-right -> 0, center -> 0.5.
        assert!((s[0][0] - 0.0).abs() < 1e-9);
        assert!((s[3][3] - 0.0).abs() < 1e-9);
        assert!((s[1][1] - 0.5).abs() < 0.15);
    }

    #[test]
    fn cells_respect_extent_and_origin() {
        let img = ImageData::new(checker(), viridis)
            .unwrap()
            .with_extent(0.0, 2.0, 10.0, 12.0);
        let cells = img.cells(16);
        assert_eq!(cells.len(), 4);
        // Upper origin: row 0 at the top (y = 12).
        let (rect0, _) = cells[0];
        assert!((rect0.1 - 11.0).abs() < 1e-9); // y_lo of row 0 (upper)
        assert!((rect0.3 - 12.0).abs() < 1e-9); // y_hi of row 0
        assert!((rect0.0 - 0.0).abs() < 1e-9);
        assert!((rect0.2 - 1.0).abs() < 1e-9);
        // Top-left cell holds value 0.0, top-right holds 1.0.
        assert_eq!(cells[0].1, viridis(0.0));
        assert_eq!(cells[1].1, viridis(1.0));
    }

    #[test]
    fn lower_origin_flips_y() {
        let img = ImageData::new(checker(), viridis)
            .unwrap()
            .with_origin(ImageOrigin::Lower);
        let cells = img.cells(16);
        let (rect0, _) = cells[0];
        assert!((rect0.1 - 0.0).abs() < 1e-9);
        assert!((rect0.3 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn constant_grid_maps_to_midpoint() {
        let img = ImageData::new(vec![vec![5.0; 3]; 3], viridis).unwrap();
        let cells = img.cells(16);
        assert!(cells.iter().all(|(_, c)| *c == viridis(0.5)));
    }
}
