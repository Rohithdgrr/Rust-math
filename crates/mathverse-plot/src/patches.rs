//! Paths, patches, and collections — the matplotlib analogue of `Path`,
//! `PathPatch`, `Rectangle`, `Circle`, `Polygon` and `LineCollection`.
//!
//! A [`Path`] is a sequence of data-coordinate points; a [`Patch`] styles it
//! with fill/stroke/opacity. [`LineCollection`] batch-renders many segments in
//! a single artist. The snapshot types ([`PathSnapshot`], [`LineSnapshot`])
//! are what the backends actually consume via [`crate::backend::PlotData`].

use crate::common::DataPoint;
use crate::style::Color;

/// An open or closed polyline in data coordinates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path {
    points: Vec<(f64, f64)>,
    closed: bool,
}

impl Path {
    /// Create an empty path (add points with [`Path::line_to`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a point.
    #[must_use]
    pub fn line_to(mut self, x: f64, y: f64) -> Self {
        self.points.push((x, y));
        self
    }

    /// Mark the path closed (a polygon).
    #[must_use]
    pub fn close(mut self) -> Self {
        self.closed = true;
        self
    }

    /// An open polyline through the given points.
    #[must_use]
    pub fn polyline(points: &[(f64, f64)]) -> Self {
        Self {
            points: points.to_vec(),
            closed: false,
        }
    }

    /// A closed polygon through the given points.
    #[must_use]
    pub fn polygon(points: &[(f64, f64)]) -> Self {
        Self {
            points: points.to_vec(),
            closed: true,
        }
    }

    /// The path points.
    #[must_use]
    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }

    /// Whether the path is closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

/// A styled path artist (fill + stroke).
#[derive(Debug, Clone)]
pub struct Patch {
    /// The path geometry.
    pub path: Path,
    /// Fill color (None = no fill).
    pub fill: Option<Color>,
    /// Stroke color (None = no stroke).
    pub stroke: Option<Color>,
    /// Stroke width in pixels.
    pub stroke_width: f64,
    /// Opacity in `[0, 1]`.
    pub opacity: f64,
}

impl Patch {
    /// A stroke-only patch with a black outline.
    #[must_use]
    pub fn new(path: Path) -> Self {
        Self {
            path,
            fill: None,
            stroke: Some(Color::BLACK),
            stroke_width: 1.0,
            opacity: 1.0,
        }
    }

    /// A solidly filled patch with no stroke.
    #[must_use]
    pub fn filled(path: Path, color: Color) -> Self {
        Self {
            path,
            fill: Some(color),
            stroke: None,
            stroke_width: 1.0,
            opacity: 1.0,
        }
    }

    /// A stroked patch.
    #[must_use]
    pub fn stroked(path: Path, color: Color, width: f64) -> Self {
        Self {
            path,
            fill: None,
            stroke: Some(color),
            stroke_width: width,
            opacity: 1.0,
        }
    }

    /// Set the fill color.
    #[must_use]
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set the stroke color.
    #[must_use]
    pub fn with_stroke(mut self, color: Color) -> Self {
        self.stroke = Some(color);
        self
    }

    /// Set the stroke width (pixels).
    #[must_use]
    pub fn with_stroke_width(mut self, width: f64) -> Self {
        self.stroke_width = width;
        self
    }

    /// Set the opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// An axis-aligned rectangle patch.
    #[must_use]
    pub fn rectangle(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self::polygon(&[
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ])
    }

    /// A circle patch approximated with `segments` points.
    #[must_use]
    pub fn circle(cx: f64, cy: f64, radius: f64) -> Self {
        Self::ellipse(cx, cy, radius, radius)
    }

    /// An ellipse patch approximated with 64 points.
    #[must_use]
    pub fn ellipse(cx: f64, cy: f64, rx: f64, ry: f64) -> Self {
        const SEGMENTS: usize = 64;
        let pts: Vec<(f64, f64)> = (0..SEGMENTS)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / SEGMENTS as f64;
                (cx + rx * a.cos(), cy + ry * a.sin())
            })
            .collect();
        Self::polygon(&pts)
    }

    /// A polygon patch through the given points.
    #[must_use]
    pub fn polygon(points: &[(f64, f64)]) -> Self {
        Self::new(Path::polygon(points))
    }
}

/// A batch of line segments sharing one style (matplotlib `LineCollection`).
#[derive(Debug, Clone)]
pub struct LineCollection {
    /// Segment endpoints in data coordinates.
    pub segments: Vec<(DataPoint, DataPoint)>,
    /// Stroke color.
    pub color: Color,
    /// Stroke width in pixels.
    pub width: f64,
}

impl LineCollection {
    /// Create a collection from endpoint pairs.
    #[must_use]
    pub fn new(segments: Vec<(DataPoint, DataPoint)>, color: Color, width: f64) -> Self {
        Self {
            segments,
            color,
            width,
        }
    }

    /// Build from parallel `(x1, y1, x2, y2)` slices; the shortest length wins.
    #[must_use]
    pub fn from_xy(x1: &[f64], y1: &[f64], x2: &[f64], y2: &[f64], color: Color, width: f64) -> Self {
        let n = x1.len().min(y1.len()).min(x2.len()).min(y2.len());
        let segments = (0..n)
            .map(|i| {
                (
                    DataPoint::new(x1[i], y1[i]),
                    DataPoint::new(x2[i], y2[i]),
                )
            })
            .collect();
        Self::new(segments, color, width)
    }
}

/// Backend-agnostic snapshot of a styled path.
#[derive(Debug, Clone)]
pub struct PathSnapshot {
    /// Points in data coordinates.
    pub points: Vec<(f64, f64)>,
    /// Whether the path is closed.
    pub closed: bool,
    /// Fill color.
    pub fill: Option<Color>,
    /// Stroke color.
    pub stroke: Option<Color>,
    /// Stroke width (pixels).
    pub stroke_width: f64,
    /// Opacity.
    pub opacity: f64,
}

impl From<&Patch> for PathSnapshot {
    fn from(p: &Patch) -> Self {
        Self {
            points: p.path.points().to_vec(),
            closed: p.path.is_closed(),
            fill: p.fill,
            stroke: p.stroke,
            stroke_width: p.stroke_width,
            opacity: p.opacity,
        }
    }
}

/// Backend-agnostic snapshot of one line segment.
#[derive(Debug, Clone, Copy)]
pub struct LineSnapshot {
    /// Start x.
    pub x1: f64,
    /// Start y.
    pub y1: f64,
    /// End x.
    pub x2: f64,
    /// End y.
    pub y2: f64,
    /// Stroke color.
    pub color: Color,
    /// Stroke width (pixels).
    pub width: f64,
}

impl From<&LineCollection> for Vec<LineSnapshot> {
    fn from(c: &LineCollection) -> Self {
        c.segments
            .iter()
            .map(|(a, b)| LineSnapshot {
                x1: a.x,
                y1: a.y,
                x2: b.x,
                y2: b.y,
                color: c.color,
                width: c.width,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_builders() {
        let p = Path::new().line_to(0.0, 0.0).line_to(1.0, 1.0).close();
        assert!(p.is_closed());
        assert_eq!(p.points(), &[(0.0, 0.0), (1.0, 1.0)]);
        let open = Path::polyline(&[(0.0, 0.0), (2.0, 2.0)]);
        assert!(!open.is_closed());
    }

    #[test]
    fn patch_styles() {
        let r = Patch::rectangle(0.0, 0.0, 4.0, 2.0).with_fill(Color::RED);
        assert_eq!(r.fill, Some(Color::RED));
        assert_eq!(r.path.points().len(), 4);
        let c = Patch::circle(0.0, 0.0, 1.0);
        assert_eq!(c.path.points().len(), 64);
        assert!(c.path.is_closed());
    }

    #[test]
    fn line_collection_snapshot() {
        let lc = LineCollection::from_xy(
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[1.0, 2.0],
            &[1.0, 2.0],
            Color::BLUE,
            2.0,
        );
        let snaps: Vec<LineSnapshot> = Vec::from(&lc);
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0].x2, 1.0);
    }

    #[test]
    fn patch_snapshot_roundtrip() {
        let p = Patch::polygon(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]);
        let s: PathSnapshot = PathSnapshot::from(&p);
        assert_eq!(s.points.len(), 3);
        assert!(s.closed);
    }
}
