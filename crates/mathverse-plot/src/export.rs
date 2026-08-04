//! Export options for SVG output.

use crate::style::Color;

/// Export configuration for SVG.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Background color.
    pub background: Color,
    /// Show background.
    pub show_background: bool,
    /// Margin (pixels).
    pub margin: Margin,
    /// Title.
    pub title: String,
    /// Footer text.
    pub footer: String,
    /// Show title.
    pub show_title: bool,
    /// Show footer.
    pub show_footer: bool,
    /// SVG version.
    pub svg_version: String,
    /// Embed fonts.
    pub embed_fonts: bool,
    /// Optimize SVG.
    pub optimize: bool,
    /// Precision for decimal places.
    pub precision: usize,
    /// Custom CSS.
    pub custom_css: String,
    /// Metadata.
    pub metadata: Metadata,
    /// Watermark.
    pub watermark: Option<Watermark>,
}

/// Margin configuration.
#[derive(Debug, Clone)]
pub struct Margin {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Margin {
    /// Create uniform margin.
    pub fn uniform(value: f64) -> Self {
        Self {
            top: value,
            right: value,
            bottom: value,
            left: value,
        }
    }

    /// Create asymmetric margin.
    pub fn new(top: f64, right: f64, bottom: f64, left: f64) -> Self {
        Self { top, right, bottom, left }
    }

    /// Total width consumed.
    pub fn total_width(&self) -> f64 {
        self.left + self.right
    }

    /// Total height consumed.
    pub fn total_height(&self) -> f64 {
        self.top + self.bottom
    }
}

impl Default for Margin {
    fn default() -> Self {
        Self::uniform(20.0)
    }
}

/// SVG metadata.
#[derive(Debug, Clone)]
pub struct Metadata {
    /// Document title.
    pub title: String,
    /// Description.
    pub description: String,
    /// Author.
    pub author: String,
    /// Keywords.
    pub keywords: Vec<String>,
    /// Custom properties.
    pub properties: std::collections::HashMap<String, String>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            title: String::new(),
            description: String::new(),
            author: String::new(),
            keywords: vec![],
            properties: std::collections::HashMap::new(),
        }
    }
}

/// Watermark configuration.
#[derive(Debug, Clone)]
pub struct Watermark {
    /// Watermark text.
    pub text: String,
    /// Text color.
    pub color: Color,
    /// Font size.
    pub font_size: f64,
    /// Opacity.
    pub opacity: f64,
    /// Rotation (degrees).
    pub rotation: f64,
    /// Position.
    pub position: WatermarkPosition,
}

/// Watermark position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WatermarkPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
    Tiled,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ExportConfig {
    /// Create a new export config.
    pub fn new() -> Self {
        Self {
            width: 600,
            height: 400,
            background: Color::WHITE,
            show_background: true,
            margin: Margin::default(),
            title: String::new(),
            footer: String::new(),
            show_title: false,
            show_footer: false,
            svg_version: "1.1".into(),
            embed_fonts: true,
            optimize: false,
            precision: 2,
            custom_css: String::new(),
            metadata: Metadata::default(),
            watermark: None,
        }
    }

    /// Set dimensions.
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set background.
    pub fn with_background(mut self, color: Color) -> Self {
        self.background = color;
        self.show_background = true;
        self
    }

    /// Set transparent background.
    pub fn transparent(mut self) -> Self {
        self.show_background = false;
        self
    }

    /// Set margin.
    pub fn with_margin(mut self, margin: Margin) -> Self {
        self.margin = margin;
        self
    }

    /// Set title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self.show_title = true;
        self
    }

    /// Set footer.
    pub fn with_footer(mut self, footer: impl Into<String>) -> Self {
        self.footer = footer.into();
        self.show_footer = true;
        self
    }

    /// Add watermark.
    pub fn with_watermark(mut self, text: impl Into<String>) -> Self {
        self.watermark = Some(Watermark {
            text: text.into(),
            color: Color::rgb(200, 200, 200),
            font_size: 24.0,
            opacity: 0.3,
            rotation: -30.0,
            position: WatermarkPosition::Center,
        });
        self
    }

    /// Set custom CSS.
    pub fn with_css(mut self, css: impl Into<String>) -> Self {
        self.custom_css = css.into();
        self
    }

    /// Generate SVG header.
    pub fn svg_header(&self) -> String {
        let mut svg = String::new();

        // XML declaration
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

        // SVG opening tag
        svg.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"");
        svg.push_str(&self.svg_version);
        svg.push_str("\" width=\"");
        svg.push_str(&self.width.to_string());
        svg.push_str("\" height=\"");
        svg.push_str(&self.height.to_string());
        svg.push_str("\" viewBox=\"0 0 ");
        svg.push_str(&self.width.to_string());
        svg.push_str(" ");
        svg.push_str(&self.height.to_string());
        svg.push_str("\">\n");

        // Metadata
        if !self.metadata.title.is_empty() {
            svg.push_str("  <title>");
            svg.push_str(&self.metadata.title);
            svg.push_str("</title>\n");
        }
        if !self.metadata.description.is_empty() {
            svg.push_str("  <desc>");
            svg.push_str(&self.metadata.description);
            svg.push_str("</desc>\n");
        }

        // Defs
        svg.push_str("  <defs>\n");

        // Custom CSS
        if !self.custom_css.is_empty() {
            svg.push_str("    <style>\n");
            svg.push_str(&self.custom_css);
            svg.push_str("\n    </style>\n");
        }

        // Watermark pattern
        if let Some(ref wm) = self.watermark {
            if wm.position == WatermarkPosition::Tiled {
                svg.push_str("    <pattern id=\"watermark\" width=\"200\" height=\"200\" patternUnits=\"userSpaceOnUse\" patternTransform=\"rotate(");
                svg.push_str(&wm.rotation.to_string());
                svg.push_str(")\">\n");
                svg.push_str("      <text x=\"100\" y=\"100\" text-anchor=\"middle\" font-size=\"");
                svg.push_str(&wm.font_size.to_string());
                svg.push_str("\" fill=\"");
                svg.push_str(&wm.color.to_hex());
                svg.push_str("\" opacity=\"");
                svg.push_str(&wm.opacity.to_string());
                svg.push_str("\">");
                svg.push_str(&wm.text);
                svg.push_str("</text>\n");
                svg.push_str("    </pattern>\n");
            }
        }

        svg.push_str("  </defs>\n");

        // Background
        if self.show_background {
            svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"");
            svg.push_str(&self.background.to_hex());
            svg.push_str("\"/>\n");
        }

        svg
    }

    /// Generate SVG footer.
    pub fn svg_footer(&self) -> String {
        let mut svg = String::new();

        // Watermark
        if let Some(ref wm) = self.watermark {
            match wm.position {
                WatermarkPosition::Tiled => {
                    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"url(#watermark)\"/>\n");
                }
                WatermarkPosition::Center => {
                    let cx = self.width as f64 / 2.0;
                    let cy = self.height as f64 / 2.0;
                    svg.push_str("  <text x=\"");
                    svg.push_str(&cx.to_string());
                    svg.push_str("\" y=\"");
                    svg.push_str(&cy.to_string());
                    svg.push_str("\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"");
                    svg.push_str(&wm.font_size.to_string());
                    svg.push_str("\" fill=\"");
                    svg.push_str(&wm.color.to_hex());
                    svg.push_str("\" opacity=\"");
                    svg.push_str(&wm.opacity.to_string());
                    svg.push_str("\" transform=\"rotate(");
                    svg.push_str(&wm.rotation.to_string());
                    svg.push_str(", ");
                    svg.push_str(&cx.to_string());
                    svg.push_str(", ");
                    svg.push_str(&cy.to_string());
                    svg.push_str(")\">");
                    svg.push_str(&wm.text);
                    svg.push_str("</text>\n");
                }
                WatermarkPosition::TopLeft => {
                    svg.push_str("  <text x=\"20\" y=\"30\" font-size=\"");
                    svg.push_str(&wm.font_size.to_string());
                    svg.push_str("\" fill=\"");
                    svg.push_str(&wm.color.to_hex());
                    svg.push_str("\" opacity=\"");
                    svg.push_str(&wm.opacity.to_string());
                    svg.push_str("\">");
                    svg.push_str(&wm.text);
                    svg.push_str("</text>\n");
                }
                WatermarkPosition::TopRight => {
                    svg.push_str("  <text x=\"");
                    svg.push_str(&(self.width as f64 - 20.0).to_string());
                    svg.push_str("\" y=\"30\" text-anchor=\"end\" font-size=\"");
                    svg.push_str(&wm.font_size.to_string());
                    svg.push_str("\" fill=\"");
                    svg.push_str(&wm.color.to_hex());
                    svg.push_str("\" opacity=\"");
                    svg.push_str(&wm.opacity.to_string());
                    svg.push_str("\">");
                    svg.push_str(&wm.text);
                    svg.push_str("</text>\n");
                }
                WatermarkPosition::BottomLeft => {
                    svg.push_str("  <text x=\"20\" y=\"");
                    svg.push_str(&(self.height as f64 - 20.0).to_string());
                    svg.push_str("\" font-size=\"");
                    svg.push_str(&wm.font_size.to_string());
                    svg.push_str("\" fill=\"");
                    svg.push_str(&wm.color.to_hex());
                    svg.push_str("\" opacity=\"");
                    svg.push_str(&wm.opacity.to_string());
                    svg.push_str("\">");
                    svg.push_str(&wm.text);
                    svg.push_str("</text>\n");
                }
                WatermarkPosition::BottomRight => {
                    svg.push_str("  <text x=\"");
                    svg.push_str(&(self.width as f64 - 20.0).to_string());
                    svg.push_str("\" y=\"");
                    svg.push_str(&(self.height as f64 - 20.0).to_string());
                    svg.push_str("\" text-anchor=\"end\" font-size=\"");
                    svg.push_str(&wm.font_size.to_string());
                    svg.push_str("\" fill=\"");
                    svg.push_str(&wm.color.to_hex());
                    svg.push_str("\" opacity=\"");
                    svg.push_str(&wm.opacity.to_string());
                    svg.push_str("\">");
                    svg.push_str(&wm.text);
                    svg.push_str("</text>\n");
                }
            }
        }

        // Title
        if self.show_title && !self.title.is_empty() {
            svg.push_str("  <text x=\"");
            svg.push_str(&(self.width as f64 / 2.0).to_string());
            svg.push_str("\" y=\"25\" text-anchor=\"middle\" font-size=\"16\" font-weight=\"bold\">");
            svg.push_str(&self.title);
            svg.push_str("</text>\n");
        }

        // Footer
        if self.show_footer && !self.footer.is_empty() {
            svg.push_str("  <text x=\"");
            svg.push_str(&(self.width as f64 / 2.0).to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&(self.height as f64 - 10.0).to_string());
            svg.push_str("\" text-anchor=\"middle\" font-size=\"10\" fill=\"gray\">");
            svg.push_str(&self.footer);
            svg.push_str("</text>\n");
        }

        // Closing tag
        svg.push_str("</svg>");

        svg
    }

    /// Get plot area dimensions.
    pub fn plot_area(&self) -> (f64, f64, f64, f64) {
        let x = self.margin.left;
        let y = self.margin.top;
        let w = self.width as f64 - self.margin.total_width();
        let h = self.height as f64 - self.margin.total_height();
        (x, y, w, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_config_compile() {
        let _ = ExportConfig::new()
            .with_dimensions(800, 600)
            .with_background(Color::WHITE)
            .with_title("My Chart")
            .with_footer("Generated by mathverse-plot")
            .with_watermark("DRAFT");
    }

    #[test]
    fn margin_calculation() {
        let m = Margin::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(m.total_width(), 60.0);
        assert_eq!(m.total_height(), 40.0);
    }

    #[test]
    fn svg_header_generation() {
        let config = ExportConfig::new().with_dimensions(600, 400);
        let header = config.svg_header();
        assert!(header.contains("600"));
        assert!(header.contains("400"));
    }

    #[test]
    fn plot_area_calculation() {
        let config = ExportConfig::new()
            .with_dimensions(800, 600)
            .with_margin(Margin::uniform(50.0));
        let (x, y, w, h) = config.plot_area();
        assert_eq!(x, 50.0);
        assert_eq!(y, 50.0);
        assert_eq!(w, 700.0);
        assert_eq!(h, 500.0);
    }
}
