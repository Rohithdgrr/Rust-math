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
            Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
            Color::Rgba(r, g, b, a) => format!("#{r:02x}{g:02x}{b:02x}{a:02x}"),
            Color::Named(_name) => {
                let (r, g, b) = self.to_rgb();
                format!("#{r:02x}{g:02x}{b:02x}")
            }
        }
    }

    /// Get the RGB components, resolving named colors to concrete values.
    /// Non-RGB variants fall back to a deterministic grey derived from the
    /// variant so raster backends never panic.
    #[must_use]
    pub fn to_rgb(&self) -> (u8, u8, u8) {
        match self {
            Color::Rgb(r, g, b) => (*r, *g, *b),
            Color::Rgba(r, g, b, _) => (*r, *g, *b),
            Color::Named(name) => match *name {
                "black" => (0, 0, 0),
                "white" => (255, 255, 255),
                "red" => (255, 0, 0),
                "green" => (0, 128, 0),
                "blue" => (0, 0, 255),
                "yellow" => (255, 255, 0),
                "cyan" => (0, 255, 255),
                "magenta" => (255, 0, 255),
                "gray" | "grey" => (128, 128, 128),
                "orange" => (255, 165, 0),
                "purple" => (128, 0, 128),
                "brown" => (165, 42, 42),
                "navy" => (0, 0, 128),
                "teal" => (0, 128, 128),
                "olive" => (128, 128, 0),
                "maroon" => (128, 0, 0),
                "silver" => (192, 192, 192),
                "lime" => (0, 255, 0),
                "fuchsia" => (255, 0, 255),
                "aqua" => (0, 255, 255),
                other => {
                    let h = other
                        .as_bytes()
                        .iter()
                        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(*b as u64));
                    let v = (h & 0xFF) as u8;
                    (v, v, v)
                }
            },
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

/// Predefined style presets inspired by popular plotting libraries.
impl PlotStyle {
    /// Seaborn-muted palette: soft blue line, light gray grid, white background.
    pub fn seaborn() -> Self {
        Self::default()
            .with_line_color(Color::Rgb(31, 119, 180))
            .with_grid_color(Color::Rgb(220, 220, 220))
            .with_background_color(Color::WHITE)
    }

    /// Seaborn "darkgrid": muted palette + visible grid.
    pub fn seaborn_darkgrid() -> Self {
        Self::seaborn()
            .with_background_color(Color::Rgb(234, 234, 242))
            .with_grid_color(Color::Rgb(255, 255, 255))
    }

    /// FiveThirtyEight: bold colors, gray background.
    pub fn fivethirtyeight() -> Self {
        Self::default()
            .with_line_color(Color::Rgb(0, 113, 188))
            .with_background_color(Color::Rgb(255, 255, 255))
            .with_grid_color(Color::Rgb(204, 204, 204))
    }

    /// Minimal: no grid, thin lines, neutral colors.
    pub fn minimal() -> Self {
        Self::default()
            .with_line_color(Color::Rgb(68, 68, 68))
            .with_line_width(1.0)
            .with_background_color(Color::WHITE)
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

    #[test]
    fn style_presets_compile_and_differ() {
        let s1 = PlotStyle::seaborn();
        let s2 = PlotStyle::seaborn_darkgrid();
        let s3 = PlotStyle::fivethirtyeight();
        let s4 = PlotStyle::minimal();
        // Seaborn uses blue; minimal uses dark gray
        assert_ne!(s1.line_color, s4.line_color);
        // Darkgrid has gray background; seaborn has white
        assert_ne!(s2.background_color, s1.background_color);
        // All have line_width default
        assert_eq!(s3.line_width, 2.0);
    }
}
