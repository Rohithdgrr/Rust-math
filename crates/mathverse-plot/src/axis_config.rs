//! Axis customization options.

use crate::style::Color;

/// Axis configuration for x or y axis.
#[derive(Debug, Clone)]
pub struct AxisConfig {
    /// Axis label.
    pub label: String,
    /// Label color.
    pub label_color: Color,
    /// Label font size.
    pub label_size: f64,
    /// Show label.
    pub show_label: bool,
    /// Tick label size.
    pub tick_size: f64,
    /// Tick label color.
    pub tick_color: Color,
    /// Show tick labels.
    pub show_tick_labels: bool,
    /// Tick rotation (degrees).
    pub tick_rotation: f64,
    /// Axis limits.
    pub limits: Option<(f64, f64)>,
    /// Scale type.
    pub scale: AxisScale,
    /// Axis color.
    pub color: Color,
    /// Line width.
    pub line_width: f64,
    /// Show axis line.
    pub show_line: bool,
    /// Show tick marks.
    pub show_ticks: bool,
    /// Tick length.
    pub tick_length: f64,
    /// Tick width.
    pub tick_width: f64,
    /// Minor ticks.
    pub minor_ticks: bool,
    /// Grid.
    pub grid: bool,
    /// Grid color.
    pub grid_color: Color,
    /// Grid style.
    pub grid_style: GridStyle,
    /// Invert axis.
    pub inverted: bool,
    /// Log base (if log scale).
    pub log_base: f64,
    /// Base for scientific notation.
    pub scientific_notation: bool,
    /// Tick values (if custom).
    pub tick_values: Option<Vec<f64>>,
    /// Tick labels (if custom).
    pub tick_labels: Option<Vec<String>>,
}

/// Axis scale type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisScale {
    Linear,
    Log,
    Symlog,
    Sqrt,
}

/// Grid line style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridStyle {
    None,
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

impl GridStyle {
    /// Get SVG stroke-dasharray.
    pub fn to_svg(&self) -> &str {
        match self {
            GridStyle::None => "none",
            GridStyle::Solid => "none",
            GridStyle::Dashed => "6,4",
            GridStyle::Dotted => "2,2",
            GridStyle::DashDot => "6,2,2,2",
        }
    }
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl AxisConfig {
    /// Create a new axis config.
    pub fn new() -> Self {
        Self {
            label: String::new(),
            label_color: Color::BLACK,
            label_size: 12.0,
            show_label: true,
            tick_size: 10.0,
            tick_color: Color::BLACK,
            show_tick_labels: true,
            tick_rotation: 0.0,
            limits: None,
            scale: AxisScale::Linear,
            color: Color::BLACK,
            line_width: 1.0,
            show_line: true,
            show_ticks: true,
            tick_length: 5.0,
            tick_width: 1.0,
            minor_ticks: false,
            grid: false,
            grid_color: Color::rgb(220, 220, 220),
            grid_style: GridStyle::Solid,
            inverted: false,
            log_base: 10.0,
            scientific_notation: false,
            tick_values: None,
            tick_labels: None,
        }
    }

    /// Set label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set label color.
    pub fn with_label_color(mut self, color: Color) -> Self {
        self.label_color = color;
        self
    }

    /// Set label size.
    pub fn with_label_size(mut self, size: f64) -> Self {
        self.label_size = size;
        self
    }

    /// Set axis limits.
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.limits = Some((min, max));
        self
    }

    /// Set scale.
    pub fn with_scale(mut self, scale: AxisScale) -> Self {
        self.scale = scale;
        self
    }

    /// Set log scale.
    pub fn with_log(mut self) -> Self {
        self.scale = AxisScale::Log;
        self
    }

    /// Set tick rotation.
    pub fn with_tick_rotation(mut self, rotation: f64) -> Self {
        self.tick_rotation = rotation;
        self
    }

    /// Set custom tick values.
    pub fn with_ticks(mut self, values: Vec<f64>) -> Self {
        self.tick_values = Some(values);
        self
    }

    /// Set custom tick labels.
    pub fn with_tick_labels(mut self, labels: Vec<String>) -> Self {
        self.tick_labels = Some(labels);
        self
    }

    /// Enable grid.
    pub fn with_grid(mut self) -> Self {
        self.grid = true;
        self
    }

    /// Set grid color.
    pub fn with_grid_color(mut self, color: Color) -> Self {
        self.grid_color = color;
        self
    }

    /// Invert axis.
    pub fn with_inverted(mut self) -> Self {
        self.inverted = true;
        self
    }

    /// Generate nice tick values.
    pub fn nice_ticks(min: f64, max: f64, max_ticks: usize) -> Vec<f64> {
        if min == max {
            return vec![min];
        }

        let range = max - min;
        let rough_step = range / (max_ticks - 1) as f64;

        // Find nice step size
        let magnitude = 10.0_f64.powf(rough_step.log10().floor());
        let residual = rough_step / magnitude;

        let nice_step = if residual <= 1.5 {
            1.0 * magnitude
        } else if residual <= 3.0 {
            2.0 * magnitude
        } else if residual <= 7.0 {
            5.0 * magnitude
        } else {
            10.0 * magnitude
        };

        let nice_min = (min / nice_step).floor() * nice_step;
        let nice_max = (max / nice_step).ceil() * nice_step;

        let mut ticks = Vec::new();
        let mut tick = nice_min;
        while tick <= nice_max + nice_step * 0.5 {
            ticks.push(tick);
            tick += nice_step;
        }

        ticks
    }

    /// Format tick value.
    pub fn format_tick(value: f64) -> String {
        if value.abs() < 1e-10 {
            "0".to_string()
        } else if value.abs() >= 1e6 || value.abs() < 1e-3 {
            format!("{:.2e}", value)
        } else if value == (value as i64) as f64 {
            format!("{}", value as i64)
        } else {
            format!("{:.2}", value)
        }
    }

    /// Format tick value with rotation.
    pub fn format_tick_svg(&self, value: f64, x: f64, y: f64) -> String {
        let text = Self::format_tick(value);
        let rotation = if self.tick_rotation != 0.0 {
            format!(" transform=\"rotate({}, {}, {})\"", self.tick_rotation, x, y)
        } else {
            String::new()
        };

        format!(
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"{}\" fill=\"{}\"{}>{}</text>",
            x, y, self.tick_size, self.tick_color.to_hex(), rotation, text
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_config_compile() {
        let _ = AxisConfig::new()
            .with_label("X Axis")
            .with_limits(0.0, 10.0)
            .with_log()
            .with_tick_rotation(45.0)
            .with_grid();
    }

    #[test]
    fn nice_ticks() {
        let ticks = AxisConfig::nice_ticks(0.0, 10.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks[0] <= 0.0);
        assert!(*ticks.last().unwrap() >= 10.0);
    }

    #[test]
    fn format_tick() {
        assert_eq!(AxisConfig::format_tick(0.0), "0");
        assert_eq!(AxisConfig::format_tick(1.0), "1");
        assert_eq!(AxisConfig::format_tick(1.5), "1.50");
        assert!(AxisConfig::format_tick(1e6).contains("e"));
    }
}
