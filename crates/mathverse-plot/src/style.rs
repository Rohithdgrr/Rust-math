//! Plot styling options

/// Color representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Color {
    /// RGB color
    Rgb(u8, u8, u8),
    /// RGBA color
    Rgba(u8, u8, u8, u8),
    /// Named color
    Named(&'static str),
}

impl Color {
    /// Create a new RGB color
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Color::Rgb(r, g, b)
    }

    /// Create a new RGBA color
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color::Rgba(r, g, b, a)
    }

    /// Get the hex string representation
    pub fn to_hex(&self) -> String {
        match self {
            Color::Rgb(r, g, b) => format!("#{:02x}{:02x}{:02x}", r, g, b),
            Color::Rgba(r, g, b, a) => format!("#{:02x}{:02x}{:02x}{:02x}", r, g, b, a),
            Color::Named(name) => name.to_string(),
        }
    }
}

/// Common colors
impl Color {
    pub const BLACK: Color = Color::Named("black");
    pub const WHITE: Color = Color::Named("white");
    pub const RED: Color = Color::Named("red");
    pub const GREEN: Color = Color::Named("green");
    pub const BLUE: Color = Color::Named("blue");
    pub const YELLOW: Color = Color::Named("yellow");
    pub const CYAN: Color = Color::Named("cyan");
    pub const MAGENTA: Color = Color::Named("magenta");
    pub const GRAY: Color = Color::Named("gray");
    pub const ORANGE: Color = Color::Named("orange");
    pub const PURPLE: Color = Color::Named("purple");
    pub const BROWN: Color = Color::Named("brown");
}

/// Line style
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineStyle {
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Dash-dot line
    DashDot,
}

/// Marker style for data points
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarkerStyle {
    /// Circle marker
    Circle,
    /// Square marker
    Square,
    /// Triangle marker
    Triangle,
    /// Cross marker
    Cross,
    /// Plus marker
    Plus,
    /// Diamond marker
    Diamond,
    /// No marker
    None,
}

/// Plot style configuration
#[derive(Debug, Clone)]
pub struct PlotStyle {
    /// Line color
    pub line_color: Color,
    /// Line width
    pub line_width: f64,
    /// Line style
    pub line_style: LineStyle,
    /// Marker color
    pub marker_color: Color,
    /// Marker size
    pub marker_size: f64,
    /// Marker style
    pub marker_style: MarkerStyle,
    /// Fill color (for area plots)
    pub fill_color: Option<Color>,
    /// Background color
    pub background_color: Color,
    /// Grid color
    pub grid_color: Color,
    /// Text color
    pub text_color: Color,
}

impl Default for PlotStyle {
    fn default() -> Self {
        PlotStyle {
            line_color: Color::BLUE,
            line_width: 2.0,
            line_style: LineStyle::Solid,
            marker_color: Color::RED,
            marker_size: 5.0,
            marker_style: MarkerStyle::Circle,
            fill_color: None,
            background_color: Color::WHITE,
            grid_color: Color::GRAY,
            text_color: Color::BLACK,
        }
    }
}

impl PlotStyle {
    /// Create a new plot style with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the line color
    pub fn with_line_color(mut self, color: Color) -> Self {
        self.line_color = color;
        self
    }

    /// Set the line width
    pub fn with_line_width(mut self, width: f64) -> Self {
        self.line_width = width;
        self
    }

    /// Set the line style
    pub fn with_line_style(mut self, style: LineStyle) -> Self {
        self.line_style = style;
        self
    }

    /// Set the marker color
    pub fn with_marker_color(mut self, color: Color) -> Self {
        self.marker_color = color;
        self
    }

    /// Set the marker size
    pub fn with_marker_size(mut self, size: f64) -> Self {
        self.marker_size = size;
        self
    }

    /// Set the marker style
    pub fn with_marker_style(mut self, style: MarkerStyle) -> Self {
        self.marker_style = style;
        self
    }

    /// Set the fill color
    pub fn with_fill_color(mut self, color: Color) -> Self {
        self.fill_color = Some(color);
        self
    }

    /// Set the background color
    pub fn with_background_color(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Set the grid color
    pub fn with_grid_color(mut self, color: Color) -> Self {
        self.grid_color = color;
        self
    }

    /// Set the text color
    pub fn with_text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_to_hex() {
        assert_eq!(Color::rgb(255, 0, 0).to_hex(), "#ff0000");
        assert_eq!(Color::rgb(0, 255, 0).to_hex(), "#00ff00");
        assert_eq!(Color::rgb(0, 0, 255).to_hex(), "#0000ff");
    }

    #[test]
    fn test_plot_style_builder() {
        let style = PlotStyle::new()
            .with_line_color(Color::RED)
            .with_line_width(3.0)
            .with_marker_style(MarkerStyle::Square);

        assert_eq!(style.line_color, Color::RED);
        assert_eq!(style.line_width, 3.0);
        assert_eq!(style.marker_style, MarkerStyle::Square);
    }
}
