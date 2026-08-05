//! Dual y-axes and broken axes support.
//!
//! # Dual Y-Axes
//!
//! Allows plotting two different metrics on the same chart with independent
//! y-axes (left and right). Useful when comparing metrics with different
//! scales or units.
//!
//! # Broken Axes
//!
//! Allows showing data with gaps or discontinuities by breaking the axis.
//! Common in scientific papers where data clusters at very different ranges.

use crate::axis_config::AxisConfig;
use crate::style::Color;

/// A break in an axis (gap between two ranges).
#[derive(Debug, Clone, Copy)]
pub struct AxisBreak {
    /// Start of the break (lower bound).
    pub start: f64,
    /// End of the break (upper bound).
    pub end: f64,
    /// Visual gap size in pixels.
    pub gap_px: f64,
    /// Break mark style.
    pub style: BreakStyle,
}

/// Visual style for axis breaks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BreakStyle {
    /// Zigzag / wavy line (common in scientific papers).
    Zigzag,
    /// Parallel diagonal lines (//).
    ParallelSlashes,
    /// Simple gap with no marks.
    Gap,
    /// Square bracket style.
    SquareBracket,
}

impl AxisBreak {
    /// Create a new axis break.
    pub fn new(start: f64, end: f64) -> Self {
        Self {
            start,
            end,
            gap_px: 20.0,
            style: BreakStyle::Zigzag,
        }
    }

    /// Set gap size in pixels.
    pub fn with_gap_px(mut self, gap: f64) -> Self {
        self.gap_px = gap;
        self
    }

    /// Set break style.
    pub fn with_style(mut self, style: BreakStyle) -> Self {
        self.style = style;
        self
    }

    /// Check if a value falls within the break range.
    pub fn contains(&self, value: f64) -> bool {
        value >= self.start && value <= self.end
    }

    /// Map a value through the break, accounting for the gap.
    pub fn map_value(&self, value: f64, _scale_min: f64, _scale_max: f64) -> f64 {
        if value < self.start {
            value
        } else if value > self.end {
            // Shift everything above the break down by the break range minus gap
            let break_range = self.end - self.start;
            value - break_range + self.gap_px
        } else {
            // Inside the break — shouldn't happen for valid data
            self.start
        }
    }

    /// Render break marks at a given x position and y range.
    pub fn render_svg(&self, x: f64, y_top: f64, y_bottom: f64) -> String {
        let mut svg = String::new();
        let _mid_y = (y_top + y_bottom) / 2.0;
        let mark_height = (y_bottom - y_top) * 0.3;

        match self.style {
            BreakStyle::Zigzag => {
                // Zigzag line (5 points: start, up, down, up, end)
                let points = format!(
                    "{},{} {},{} {},{} {},{} {},{}",
                    x - 6.0, y_top,
                    x + 6.0, y_top + mark_height * 0.25,
                    x - 6.0, y_top + mark_height * 0.5,
                    x + 6.0, y_top + mark_height * 0.75,
                    x - 6.0, y_top + mark_height,
                );
                svg.push_str(&format!(
                    "  <polyline points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"1.5\"/>\n",
                    points
                ));
                // Mirror at bottom
                let points2 = format!(
                    "{},{} {},{} {},{} {},{} {},{}",
                    x - 6.0, y_bottom,
                    x + 6.0, y_bottom - mark_height * 0.25,
                    x - 6.0, y_bottom - mark_height * 0.5,
                    x + 6.0, y_bottom - mark_height * 0.75,
                    x - 6.0, y_bottom - mark_height,
                );
                svg.push_str(&format!(
                    "  <polyline points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"1.5\"/>\n",
                    points2
                ));
            }
            BreakStyle::ParallelSlashes => {
                // Two diagonal slashes
                let offset = 4.0;
                svg.push_str(&format!(
                    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"black\" stroke-width=\"1.5\"/>\n",
                    x - offset, y_top + 2.0, x + offset, y_bottom - 2.0
                ));
                svg.push_str(&format!(
                    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"black\" stroke-width=\"1.5\"/>\n",
                    x + offset - 6.0, y_top + 2.0, x + offset + 6.0 - 6.0, y_bottom - 2.0
                ));
            }
            BreakStyle::Gap => {
                // Just a gap — no visual marks
            }
            BreakStyle::SquareBracket => {
                // Square bracket on left side
                let bracket_width = 8.0;
                svg.push_str(&format!(
                    "  <path d=\"M{},{} L{},{} L{},{} L{},{}\" fill=\"none\" stroke=\"black\" stroke-width=\"1.5\"/>\n",
                    x + bracket_width, y_top,
                    x, y_top,
                    x, y_bottom,
                    x + bracket_width, y_bottom,
                ));
            }
        }

        svg
    }
}

/// Dual y-axes configuration.
#[derive(Debug, Clone)]
pub struct DualYAxis {
    /// Primary (left) y-axis configuration.
    pub primary: AxisConfig,
    /// Secondary (right) y-axis configuration.
    pub secondary: AxisConfig,
    /// Whether the axes are synchronized (same limits).
    pub synchronized: bool,
    /// Whether to show a separator line between axes.
    pub show_separator: bool,
    /// Separator line color.
    pub separator_color: Color,
}

impl Default for DualYAxis {
    fn default() -> Self {
        Self {
            primary: AxisConfig::new(),
            secondary: AxisConfig::new().with_label(""),
            synchronized: false,
            show_separator: false,
            separator_color: Color::rgb(200, 200, 200),
        }
    }
}

impl DualYAxis {
    /// Create dual y-axes with labels.
    pub fn new(primary_label: impl Into<String>, secondary_label: impl Into<String>) -> Self {
        Self {
            primary: AxisConfig::new().with_label(primary_label),
            secondary: AxisConfig::new().with_label(secondary_label),
            ..Self::default()
        }
    }

    /// Set primary axis config.
    pub fn with_primary(mut self, config: AxisConfig) -> Self {
        self.primary = config;
        self
    }

    /// Set secondary axis config.
    pub fn with_secondary(mut self, config: AxisConfig) -> Self {
        self.secondary = config;
        self
    }

    /// Synchronize axis limits.
    pub fn synchronized(mut self) -> Self {
        self.synchronized = true;
        self
    }

    /// Show separator line between axes.
    pub fn with_separator(mut self, color: Color) -> Self {
        self.show_separator = true;
        self.separator_color = color;
        self
    }

    /// Map a primary-axis value to secondary-axis value.
    pub fn map_to_secondary(&self, primary_value: f64) -> f64 {
        let (p_min, p_max) = self.primary.limits.unwrap_or((0.0, 1.0));
        let (s_min, s_max) = self.secondary.limits.unwrap_or((0.0, 1.0));

        let t = (primary_value - p_min) / (p_max - p_min);
        s_min + t * (s_max - s_min)
    }

    /// Map a secondary-axis value to primary-axis value.
    pub fn map_to_primary(&self, secondary_value: f64) -> f64 {
        let (p_min, p_max) = self.primary.limits.unwrap_or((0.0, 1.0));
        let (s_min, s_max) = self.secondary.limits.unwrap_or((0.0, 1.0));

        let t = (secondary_value - s_min) / (s_max - s_min);
        p_min + t * (p_max - p_min)
    }
}

/// Broken axes configuration.
#[derive(Debug, Clone)]
pub struct BrokenAxis {
    /// Axis breaks.
    pub breaks: Vec<AxisBreak>,
    /// Whether the axis is horizontal or vertical.
    pub horizontal: bool,
    /// Axis configuration.
    pub config: AxisConfig,
}

impl Default for BrokenAxis {
    fn default() -> Self {
        Self {
            breaks: Vec::new(),
            horizontal: true,
            config: AxisConfig::new(),
        }
    }
}

impl BrokenAxis {
    /// Create a new broken axis.
    pub fn new(horizontal: bool) -> Self {
        Self {
            horizontal,
            ..Self::default()
        }
    }

    /// Add a break.
    pub fn with_break(mut self, break_: AxisBreak) -> Self {
        self.breaks.push(break_);
        self
    }

    /// Add a break at a range.
    pub fn break_at(mut self, start: f64, end: f64) -> Self {
        self.breaks.push(AxisBreak::new(start, end));
        self
    }

    /// Set axis config.
    pub fn with_config(mut self, config: AxisConfig) -> Self {
        self.config = config;
        self
    }

    /// Check if a value is inside any break.
    pub fn in_break(&self, value: f64) -> bool {
        self.breaks.iter().any(|b| b.contains(value))
    }

    /// Map a value through all breaks.
    pub fn map_value(&self, value: f64) -> f64 {
        let mut result = value;
        let mut shift = 0.0;

        // Sort breaks by start value
        let mut sorted_breaks = self.breaks.clone();
        sorted_breaks.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));

        for break_ in &sorted_breaks {
            if result > break_.end {
                shift += break_.gap_px - (break_.end - break_.start);
            } else if result >= break_.start {
                result = break_.start;
            }
        }

        result + shift
    }

    /// Get the total gap size in pixels from all breaks.
    pub fn total_gap_px(&self) -> f64 {
        self.breaks.iter().map(|b| b.gap_px).sum()
    }

    /// Render break marks at a specific position.
    pub fn render_breaks_svg(&self, x: f64, y_top: f64, y_bottom: f64) -> String {
        let mut svg = String::new();
        for break_ in &self.breaks {
            svg.push_str(&break_.render_svg(x, y_top, y_bottom));
        }
        svg
    }
}

/// Builder for creating common dual-axis and broken-axis configurations.
pub struct AxisBuilder;

impl AxisBuilder {
    /// Create a dual y-axis for temperature (°C) and precipitation (mm).
    pub fn temperature_precipitation() -> DualYAxis {
        DualYAxis::new("Temperature (°C)", "Precipitation (mm)")
            .with_primary(AxisConfig::new().with_label("Temperature (°C)"))
            .with_secondary(AxisConfig::new().with_label("Precipitation (mm)"))
    }

    /// Create a dual y-axis for count and percentage.
    pub fn count_percentage() -> DualYAxis {
        DualYAxis::new("Count", "Percentage (%)")
            .with_primary(AxisConfig::new().with_label("Count"))
            .with_secondary(AxisConfig::new().with_label("Percentage (%)"))
    }

    /// Create a broken y-axis with one break.
    pub fn broken_y(start: f64, end: f64) -> BrokenAxis {
        BrokenAxis::new(false).break_at(start, end)
    }

    /// Create a broken x-axis with one break.
    pub fn broken_x(start: f64, end: f64) -> BrokenAxis {
        BrokenAxis::new(true).break_at(start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_break_contains() {
        let break_ = AxisBreak::new(10.0, 20.0);
        assert!(!break_.contains(5.0));
        assert!(break_.contains(15.0));
        assert!(!break_.contains(25.0));
    }

    #[test]
    fn axis_break_gap_px() {
        let break_ = AxisBreak::new(10.0, 20.0).with_gap_px(30.0);
        assert_eq!(break_.gap_px, 30.0);
    }

    #[test]
    fn axis_break_render_svg() {
        let break_ = AxisBreak::new(10.0, 20.0);
        let svg = break_.render_svg(100.0, 50.0, 150.0);
        assert!(svg.contains("<polyline") || svg.contains("<line") || svg.contains("<path"));
    }

    #[test]
    fn axis_break_styles() {
        let zigzag = AxisBreak::new(0.0, 1.0).with_style(BreakStyle::Zigzag);
        let slashes = AxisBreak::new(0.0, 1.0).with_style(BreakStyle::ParallelSlashes);
        let gap = AxisBreak::new(0.0, 1.0).with_style(BreakStyle::Gap);
        let bracket = AxisBreak::new(0.0, 1.0).with_style(BreakStyle::SquareBracket);

        assert_eq!(zigzag.style, BreakStyle::Zigzag);
        assert_eq!(slashes.style, BreakStyle::ParallelSlashes);
        assert_eq!(gap.style, BreakStyle::Gap);
        assert_eq!(bracket.style, BreakStyle::SquareBracket);
    }

    #[test]
    fn dual_y_axis_creation() {
        let dual = DualYAxis::new("Left", "Right");
        assert_eq!(dual.primary.label, "Left");
        assert_eq!(dual.secondary.label, "Right");
    }

    #[test]
    fn dual_y_axis_mapping() {
        let dual = DualYAxis::new("A", "B")
            .with_primary(AxisConfig::new().with_limits(0.0, 100.0))
            .with_secondary(AxisConfig::new().with_limits(0.0, 1.0));

        let mapped = dual.map_to_secondary(50.0);
        assert!((mapped - 0.5).abs() < 0.001);

        let back = dual.map_to_primary(0.5);
        assert!((back - 50.0).abs() < 0.001);
    }

    #[test]
    fn broken_axis_creation() {
        let broken = BrokenAxis::new(true)
            .break_at(10.0, 20.0)
            .break_at(50.0, 60.0);

        assert_eq!(broken.breaks.len(), 2);
        assert!(broken.in_break(15.0));
        assert!(!broken.in_break(25.0));
    }

    #[test]
    fn broken_axis_total_gap() {
        let broken = BrokenAxis::new(false)
            .with_break(AxisBreak::new(10.0, 20.0).with_gap_px(15.0))
            .with_break(AxisBreak::new(50.0, 60.0).with_gap_px(25.0));

        assert_eq!(broken.total_gap_px(), 40.0);
    }

    #[test]
    fn broken_axis_render_breaks() {
        let broken = BrokenAxis::new(false)
            .break_at(10.0, 20.0);
        let svg = broken.render_breaks_svg(100.0, 50.0, 150.0);
        assert!(svg.contains("<polyline") || svg.contains("<line") || svg.contains("<path"));
    }

    #[test]
    fn axis_builder_presets() {
        let temp = AxisBuilder::temperature_precipitation();
        assert_eq!(temp.primary.label, "Temperature (°C)");
        assert_eq!(temp.secondary.label, "Precipitation (mm)");

        let cp = AxisBuilder::count_percentage();
        assert_eq!(cp.primary.label, "Count");
        assert_eq!(cp.secondary.label, "Percentage (%)");

        let broken_y = AxisBuilder::broken_y(10.0, 20.0);
        assert_eq!(broken_y.breaks.len(), 1);
        assert!(!broken_y.horizontal);

        let broken_x = AxisBuilder::broken_x(10.0, 20.0);
        assert!(broken_x.horizontal);
    }
}
