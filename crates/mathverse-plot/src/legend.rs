//! Flexible legend positioning and layout.

use crate::style::Color;

/// Legend position relative to the plot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LegendPosition {
    /// Top-right corner (default).
    UpperRight,
    /// Top-left corner.
    UpperLeft,
    /// Bottom-right corner.
    LowerRight,
    /// Bottom-left corner.
    LowerLeft,
    /// Center of the plot.
    Center,
    /// Outside the plot on the right.
    OutsideRight,
    /// Outside the plot on the left.
    OutsideLeft,
    /// Outside the plot at the top.
    OutsideTop,
    /// Outside the plot at the bottom.
    OutsideBottom,
    /// Custom position (x, y) in pixels from top-left.
    Custom(f64, f64),
}

impl Default for LegendPosition {
    fn default() -> Self {
        Self::UpperRight
    }
}

/// Legend layout direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendLayout {
    /// Vertical list (default).
    Vertical,
    /// Horizontal row.
    Horizontal,
    /// Auto: horizontal if few items, vertical if many.
    Auto,
}

impl Default for LegendLayout {
    fn default() -> Self {
        Self::Auto
    }
}

/// Configuration for legend rendering.
#[derive(Debug, Clone)]
pub struct LegendConfig {
    /// Position of the legend.
    pub position: LegendPosition,
    /// Layout direction.
    pub layout: LegendLayout,
    /// Number of columns when layout is Horizontal or Auto.
    pub columns: usize,
    /// Font size for legend labels.
    pub font_size: f64,
    /// Background color (None = transparent).
    pub background: Option<Color>,
    /// Border color (None = no border).
    pub border: Option<Color>,
    /// Padding inside the legend box (pixels).
    pub padding: f64,
    /// Spacing between items (pixels).
    pub item_spacing: f64,
    /// Spacing between columns (pixels).
    pub column_spacing: f64,
    /// Maximum width for text labels (truncates with ellipsis).
    pub max_text_width: Option<f64>,
}

impl Default for LegendConfig {
    fn default() -> Self {
        Self {
            position: LegendPosition::UpperRight,
            layout: LegendLayout::Auto,
            columns: 3,
            font_size: 11.0,
            background: Some(Color::rgba(255, 255, 255, 230)),
            border: Some(Color::rgb(0xCC, 0xCC, 0xCC)),
            padding: 8.0,
            item_spacing: 5.0,
            column_spacing: 15.0,
            max_text_width: None,
        }
    }
}

impl LegendConfig {
    /// Create a new legend config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the position.
    pub fn with_position(mut self, pos: LegendPosition) -> Self {
        self.position = pos;
        self
    }

    /// Set the layout direction.
    pub fn with_layout(mut self, layout: LegendLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Set the number of columns.
    pub fn with_columns(mut self, cols: usize) -> Self {
        self.columns = cols.max(1);
        self
    }

    /// Set the font size.
    pub fn with_font_size(mut self, size: f64) -> Self {
        self.font_size = size;
        self
    }

    /// Set the background color.
    pub fn with_background(mut self, color: Color) -> Self {
        self.background = Some(color);
        self
    }

    /// Set the border color.
    pub fn with_border(mut self, color: Color) -> Self {
        self.border = Some(color);
        self
    }

    /// No background (transparent).
    pub fn without_background(mut self) -> Self {
        self.background = None;
        self
    }

    /// No border.
    pub fn without_border(mut self) -> Self {
        self.border = None;
        self
    }
}

/// A legend item with name and color.
#[derive(Debug, Clone)]
pub struct LegendItem {
    /// Series name.
    pub name: String,
    /// Color swatch.
    pub color: Color,
    /// Optional secondary color (for gradient swatches).
    pub color2: Option<Color>,
}

impl LegendItem {
    /// Create a new legend item.
    pub fn new(name: impl Into<String>, color: Color) -> Self {
        Self {
            name: name.into(),
            color,
            color2: None,
        }
    }

    /// Create a gradient legend item.
    pub fn gradient(name: impl Into<String>, color1: Color, color2: Color) -> Self {
        Self {
            name: name.into(),
            color: color1,
            color2: Some(color2),
        }
    }
}

/// Compute the position of the legend box within the plot area.
pub fn legend_position(
    config: &LegendConfig,
    plot_width: f64,
    plot_height: f64,
    legend_width: f64,
    legend_height: f64,
    padding: f64,
) -> (f64, f64) {
    let margin = 10.0;
    match config.position {
        LegendPosition::UpperRight => (
            padding + plot_width - legend_width - margin,
            padding + margin,
        ),
        LegendPosition::UpperLeft => (padding + margin, padding + margin),
        LegendPosition::LowerRight => (
            padding + plot_width - legend_width - margin,
            padding + plot_height - legend_height - margin,
        ),
        LegendPosition::LowerLeft => (
            padding + margin,
            padding + plot_height - legend_height - margin,
        ),
        LegendPosition::Center => (
            padding + (plot_width - legend_width) / 2.0,
            padding + (plot_height - legend_height) / 2.0,
        ),
        LegendPosition::OutsideRight => (padding + plot_width + margin, padding),
        LegendPosition::OutsideLeft => (
            padding - legend_width - margin,
            padding,
        ),
        LegendPosition::OutsideTop => (
            padding + (plot_width - legend_width) / 2.0,
            padding - legend_height - margin,
        ),
        LegendPosition::OutsideBottom => (
            padding + (plot_width - legend_width) / 2.0,
            padding + plot_height + margin,
        ),
        LegendPosition::Custom(x, y) => (x, y),
    }
}

/// Estimate the size of the legend box.
pub fn estimate_legend_size(
    items: &[LegendItem],
    config: &LegendConfig,
) -> (f64, f64) {
    let is_horizontal = config.layout == LegendLayout::Horizontal
        || (config.layout == LegendLayout::Auto && items.len() <= 4);

    let item_width = 80.0; // swatch + text + spacing
    let item_height = config.font_size + config.item_spacing;

    if is_horizontal {
        let cols = config.columns.min(items.len());
        let rows = (items.len() + cols - 1) / cols;
        let width = cols as f64 * item_width + (cols - 1) as f64 * config.column_spacing;
        let height = rows as f64 * item_height;
        (
            width + 2.0 * config.padding,
            height + 2.0 * config.padding,
        )
    } else {
        let width = items
            .iter()
            .map(|item| item.name.len() as f64 * config.font_size * 0.6 + 20.0)
            .fold(0.0_f64, f64::max);
        let height = items.len() as f64 * item_height;
        (
            width + 2.0 * config.padding,
            height + 2.0 * config.padding,
        )
    }
}

/// Render a legend as SVG.
pub fn render_legend(
    items: &[LegendItem],
    x: f64,
    y: f64,
    config: &LegendConfig,
) -> String {
    let (w, h) = estimate_legend_size(items, config);
    let mut svg = String::new();

    // Background
    if let Some(bg) = &config.background {
        svg.push_str(&format!(
            r#"  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{}" rx="4"/>"#,
            bg.to_hex()
        ));
        svg.push('\n');
    }

    // Border
    if let Some(border) = &config.border {
        svg.push_str(&format!(
            r#"  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{}" stroke-width="1" rx="4"/>"#,
            border.to_hex()
        ));
        svg.push('\n');
    }

    let is_horizontal = config.layout == LegendLayout::Horizontal
        || (config.layout == LegendLayout::Auto && items.len() <= 4);

    let item_height = config.font_size + config.item_spacing;
    let swatch_size = config.font_size * 0.8;

    if is_horizontal {
        let cols = config.columns.min(items.len());
        let item_w = (w - 2.0 * config.padding - (cols - 1) as f64 * config.column_spacing) / cols as f64;

        for (i, item) in items.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let ix = x + config.padding + col as f64 * (item_w + config.column_spacing);
            let iy = y + config.padding + row as f64 * item_height;

            // Color swatch
            svg.push_str(&format!(
                r#"  <rect x="{ix}" y="{iy}" width="{swatch_size}" height="{swatch_size}" fill="{}" rx="2"/>"#,
                item.color.to_hex()
            ));
            svg.push('\n');

            // Label
            let text_x = ix + swatch_size + 4.0;
            let text_y = iy + swatch_size * 0.8;
            svg.push_str(&format!(
                r#"  <text x="{text_x}" y="{text_y}" font-size="{}">{}</text>"#,
                config.font_size, item.name
            ));
            svg.push('\n');
        }
    } else {
        for (i, item) in items.iter().enumerate() {
            let iy = y + config.padding + i as f64 * item_height;
            let ix = x + config.padding;

            // Color swatch
            svg.push_str(&format!(
                r#"  <rect x="{ix}" y="{iy}" width="{swatch_size}" height="{swatch_size}" fill="{}" rx="2"/>"#,
                item.color.to_hex()
            ));
            svg.push('\n');

            // Label
            let text_x = ix + swatch_size + 4.0;
            let text_y = iy + swatch_size * 0.8;
            svg.push_str(&format!(
                r#"  <text x="{text_x}" y="{text_y}" font-size="{}">{}</text>"#,
                config.font_size, item.name
            ));
            svg.push('\n');
        }
    }

    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legend_items() {
        let items = vec![
            LegendItem::new("Series A", Color::RED),
            LegendItem::new("Series B", Color::BLUE),
        ];
        let config = LegendConfig::new();
        let (w, h) = estimate_legend_size(&items, &config);
        assert!(w > 0.0);
        assert!(h > 0.0);
    }

    #[test]
    fn legend_renders_svg() {
        let items = vec![
            LegendItem::new("A", Color::RED),
            LegendItem::new("B", Color::BLUE),
        ];
        let config = LegendConfig::new();
        let svg = render_legend(&items, 100.0, 100.0, &config);
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<text"));
    }

    #[test]
    fn legend_position_upper_right() {
        let config = LegendConfig::new().with_position(LegendPosition::UpperRight);
        let (x, y) = legend_position(&config, 800.0, 600.0, 100.0, 50.0, 50.0);
        assert!(x > 500.0);
        assert!(y < 100.0);
    }

    #[test]
    fn legend_position_custom() {
        let config = LegendConfig::new().with_position(LegendPosition::Custom(200.0, 300.0));
        let (x, y) = legend_position(&config, 800.0, 600.0, 100.0, 50.0, 50.0);
        assert_eq!(x, 200.0);
        assert_eq!(y, 300.0);
    }
}
