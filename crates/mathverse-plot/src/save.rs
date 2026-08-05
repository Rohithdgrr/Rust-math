//! Unified export API for saving plots in multiple formats.
//!
//! Provides a single `save()` call that can output SVG, PNG, PDF, HTML,
//! and animated formats in one pass.
//!
//! # Example
//!
//! ```rust,ignore
//! use mathverse_plot::save::{PlotSaver, FormatSet, OutputFormat};
//!
//! // Save as all formats
//! PlotSaver::new(&svg_content)
//!     .save("output/plot", FormatSet::all())?;
//!
//! // Save as web-optimized set
//! PlotSaver::new(&svg_content)
//!     .save("output/plot", FormatSet::web())?;
//! ```

use std::path::{Path, PathBuf};

/// Output format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Scalable Vector Graphics.
    Svg,
    /// Portable Network Graphics (raster).
    Png,
    /// Portable Document Format (vector).
    Pdf,
    /// HyperText Markup Language (interactive).
    Html,
    /// Animated SVG.
    AnimatedSvg,
    /// Terminal ASCII art.
    Terminal,
}

impl OutputFormat {
    /// File extension for this format.
    pub fn extension(&self) -> &str {
        match self {
            OutputFormat::Svg => "svg",
            OutputFormat::Png => "png",
            OutputFormat::Pdf => "pdf",
            OutputFormat::Html => "html",
            OutputFormat::AnimatedSvg => "svg",
            OutputFormat::Terminal => "txt",
        }
    }

    /// MIME type for this format.
    pub fn mime_type(&self) -> &str {
        match self {
            OutputFormat::Svg => "image/svg+xml",
            OutputFormat::Png => "image/png",
            OutputFormat::Pdf => "application/pdf",
            OutputFormat::Html => "text/html",
            OutputFormat::AnimatedSvg => "image/svg+xml",
            OutputFormat::Terminal => "text/plain",
        }
    }
}

/// A set of output formats.
#[derive(Debug, Clone)]
pub struct FormatSet {
    /// Formats to export.
    formats: Vec<OutputFormat>,
    /// DPI for raster formats.
    dpi: u32,
    /// Scale factor for raster formats.
    scale: f64,
}

impl FormatSet {
/// Create with PNG only (default).
pub fn new() -> Self {
Self::png()
}

/// Create with specific formats. When no formats given, defaults to PNG.
pub fn with(formats: Vec<OutputFormat>) -> Self {
Self { formats, ..Self::default_set() }
}

    /// PNG-only format set at 96 DPI.
    pub fn png() -> Self {
        Self::with(vec![OutputFormat::Png])
    }

    /// All formats (SVG + PNG + PDF + HTML).
    pub fn all() -> Self {
        Self::with(vec![
            OutputFormat::Svg,
            OutputFormat::Png,
            OutputFormat::Pdf,
            OutputFormat::Html,
        ])
    }

    /// Web-optimized set (SVG + HTML).
    pub fn web() -> Self {
        Self::with(vec![OutputFormat::Svg, OutputFormat::Html])
    }

    /// Print-optimized set (PDF + high-res PNG).
    pub fn print() -> Self {
        Self::with(vec![OutputFormat::Pdf, OutputFormat::Png])
            .with_dpi(300)
    }

    /// Presentation set (SVG + PNG at 150 DPI).
    pub fn presentation() -> Self {
        Self::with(vec![OutputFormat::Svg, OutputFormat::Png])
            .with_dpi(150)
    }

    /// Screen set (SVG only).
    pub fn screen() -> Self {
        Self::with(vec![OutputFormat::Svg])
    }

    /// Add a format.
    pub fn add(mut self, format: OutputFormat) -> Self {
        if !self.formats.contains(&format) {
            self.formats.push(format);
        }
        self
    }

    /// Set DPI for raster formats.
    pub fn with_dpi(mut self, dpi: u32) -> Self {
        self.dpi = dpi;
        self
    }

    /// Set scale factor.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Get all formats.
    pub fn formats(&self) -> &[OutputFormat] {
        &self.formats
    }

    /// Get DPI.
    pub fn dpi(&self) -> u32 {
        self.dpi
    }

    /// Get scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Png
    }
}

impl Default for FormatSet {
    fn default() -> Self {
        Self::png()
    }
}

impl FormatSet {
    /// Internal default set (used by With builder).
    fn default_set() -> Self {
        Self {
            formats: Vec::new(),
            dpi: 96,
            scale: 1.0,
        }
    }
}

/// Export result for a single format.
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// The format that was exported.
    pub format: OutputFormat,
    /// Output file path.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
    /// Whether the export succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

impl ExportResult {
    /// Create a successful result.
    pub fn success(format: OutputFormat, path: PathBuf, size: u64) -> Self {
        Self {
            format,
            path,
            size,
            success: true,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failed(format: OutputFormat, path: PathBuf, error: String) -> Self {
        Self {
            format,
            path,
            size: 0,
            success: false,
            error: Some(error),
        }
    }
}

/// Unified plot saver.
pub struct PlotSaver {
    /// SVG content to export.
    svg_content: String,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Title (for HTML).
    title: String,
    /// Description (for HTML).
    description: String,
    /// Plot data for PDF/advanced backends.
    plot_data: Option<crate::backend::PlotData>,
}

impl PlotSaver {
    /// Create a new saver with SVG content.
    pub fn new(svg_content: &str) -> Self {
        Self {
            svg_content: svg_content.to_string(),
            width: 800,
            height: 600,
            title: String::new(),
            description: String::new(),
            plot_data: None,
        }
    }

    /// Create from PlotData for full backend support (PDF, etc.).
    pub fn from_plot_data(data: crate::backend::PlotData) -> Self {
        let width = data.config.width;
        let height = data.config.height;
        Self {
            svg_content: String::new(),
            width,
            height,
            title: data.config.title.clone(),
            description: String::new(),
            plot_data: Some(data),
        }
    }

    /// Create from dimensions (generates placeholder SVG).
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        let svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\
             <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\
             </svg>",
            width, height, width, height
        );
        Self {
            svg_content: svg,
            width,
            height,
            title: String::new(),
            description: String::new(),
            plot_data: None,
        }
    }

    /// Set dimensions.
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set plot data for PDF/advanced backend support.
    pub fn with_plot_data(mut self, data: crate::backend::PlotData) -> Self {
        self.plot_data = Some(data);
        self
    }

    /// Set title (for HTML output).
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set description (for HTML output).
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Save to all formats in the set.
    pub fn save(&self, base_path: &str, formats: &FormatSet) -> Vec<ExportResult> {
        let mut results = Vec::new();

        for &format in formats.formats() {
            let path = Path::new(base_path).with_extension(format.extension());
            let result = self.export_format(format, &path, formats);
            results.push(result);
        }

        results
    }

    /// Save to a single format.
    pub fn save_as(&self, path: &str, format: OutputFormat, formats: &FormatSet) -> ExportResult {
        let path = Path::new(path).with_extension(format.extension());
        self.export_format(format, &path, formats)
    }

    /// Export to a specific format.
    fn export_format(&self, format: OutputFormat, path: &Path, _formats: &FormatSet) -> ExportResult {
        let content = match format {
            OutputFormat::Svg => self.generate_svg(),
            OutputFormat::Png => {
                // PNG requires the png feature
                #[cfg(feature = "png")]
                {
                    match self.generate_png(formats.dpi(), formats.scale()) {
                        Ok(bytes) => return ExportResult::success(
                            format,
                            path.to_path_buf(),
                            bytes.len() as u64,
                        ),
                        Err(e) => return ExportResult::failed(
                            format,
                            path.to_path_buf(),
                            e,
                        ),
                    }
                }
                #[cfg(not(feature = "png"))]
                {
                    return ExportResult::failed(
                        format,
                        path.to_path_buf(),
                        "PNG feature not enabled".into(),
                    );
                }
            }
            OutputFormat::Pdf => {
                #[cfg(feature = "pdf")]
                {
                    match self.generate_pdf() {
                        Ok(bytes) => return ExportResult::success(
                            format,
                            path.to_path_buf(),
                            bytes.len() as u64,
                        ),
                        Err(e) => return ExportResult::failed(
                            format,
                            path.to_path_buf(),
                            e,
                        ),
                    }
                }
                #[cfg(not(feature = "pdf"))]
                {
                    return ExportResult::failed(
                        format,
                        path.to_path_buf(),
                        "PDF feature not enabled".into(),
                    );
                }
            }
            OutputFormat::Html => self.generate_html(),
            OutputFormat::AnimatedSvg => self.generate_svg(), // Same as SVG for static
            OutputFormat::Terminal => self.generate_terminal(),
        };

        match std::fs::write(path, &content) {
            Ok(()) => ExportResult::success(
                format,
                path.to_path_buf(),
                content.len() as u64,
            ),
            Err(e) => ExportResult::failed(
                format,
                path.to_path_buf(),
                e.to_string(),
            ),
        }
    }

    /// Generate SVG content.
    fn generate_svg(&self) -> String {
        self.svg_content.clone()
    }

    /// Generate PNG (requires png feature).
    #[cfg(feature = "png")]
    fn generate_png(&self, dpi: u32, scale: f64) -> Result<Vec<u8>, String> {
        let width = (self.width as f64 * scale) as u32;
        let height = (self.height as f64 * scale) as u32;

        let mut opts = usvg::Options::default();
        opts.resources_dir = std::env::current_dir().ok();
        opts.dpi = dpi as f32;

        let tree = usvg::Tree::from_data(
            self.svg_content.as_bytes(),
            &opts,
        )
        .map_err(|e| format!("usvg parse: {e}"))?;

        let mut pixmap = tiny_skia::Pixmap::new(width, height)
            .ok_or_else(|| "Failed to create pixmap".to_string())?;
        pixmap.fill(tiny_skia::Color::WHITE);

        let transform = tiny_skia::Transform::from_scale(
            width as f32 / tree.size().width(),
            height as f32 / tree.size().height(),
        );

        resvg::render(&tree, transform, &mut pixmap.as_mut());

        pixmap
            .encode_png()
            .map_err(|e| format!("png encode: {e}"))
    }

    /// Generate PDF (requires pdf feature).
    #[cfg(feature = "pdf")]
    fn generate_pdf(&self) -> Result<Vec<u8>, String> {
        if let Some(ref data) = self.plot_data {
            let width_mm = self.width as f32 * 0.264583;
            let height_mm = self.height as f32 * 0.264583;
            let backend = crate::pdf_backend::PdfBackend::new(width_mm, height_mm);
            backend.render(data).map_err(|e| e.to_string())
        } else {
            Err("PDF export requires plot data; use from_plot_data() or with_plot_data()".into())
        }
    }

    /// Generate HTML wrapper.
    fn generate_html(&self) -> String {
        let title = if self.title.is_empty() {
            "Plot".to_string()
        } else {
            self.title.clone()
        };

        let description = if self.description.is_empty() {
            format!("{}x{} plot", self.width, self.height)
        } else {
            self.description.clone()
        };

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: #f5f5f5;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .plot-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            max-width: 100%;
        }}
        .plot-title {{
            text-align: center;
            margin-bottom: 10px;
            color: #333;
        }}
        svg {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="plot-container">
        <h1 class="plot-title">{title}</h1>
        <p style="text-align: center; color: #666; font-size: 14px;">{description}</p>
        {svg}
    </div>
</body>
</html>"#,
            title = title,
            description = description,
            svg = self.svg_content
        )
    }

    /// Generate terminal ASCII art.
    fn generate_terminal(&self) -> String {
        // Simple ASCII representation
        let mut output = String::new();
        output.push_str(&format!("Plot ({}x{})\n", self.width, self.height));
        output.push_str(&"=".repeat(40));
        output.push('\n');
        output.push_str(&self.svg_content[..self.svg_content.len().min(500)]);
        if self.svg_content.len() > 500 {
            output.push_str("\n... (truncated)");
        }
        output.push('\n');
        output.push_str(&"=".repeat(40));
        output
    }

    /// Get SVG content.
    pub fn svg_content(&self) -> &str {
        &self.svg_content
    }

    /// Save to PNG by default (default format).
    pub fn save_png(&self, path: &str) -> ExportResult {
        self.save_as(path, OutputFormat::Png, &FormatSet::png())
    }

    /// Get dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Convenience function to save a plot in all formats.
pub fn save_plot(
    svg_content: &str,
    base_path: &str,
    formats: &FormatSet,
) -> Vec<ExportResult> {
    PlotSaver::new(svg_content).save(base_path, formats)
}

/// Convenience function to save a plot as a single format.
pub fn save_plot_as(
    svg_content: &str,
    path: &str,
    format: OutputFormat,
) -> ExportResult {
    PlotSaver::new(svg_content)
        .save_as(path, format, &FormatSet::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_set_creation() {
        let fs = FormatSet::new();
        assert!(fs.formats().is_empty());

        let fs = FormatSet::all();
        assert_eq!(fs.formats().len(), 4);
    }

    #[test]
    fn format_set_web() {
        let fs = FormatSet::web();
        assert!(fs.formats().contains(&OutputFormat::Svg));
        assert!(fs.formats().contains(&OutputFormat::Html));
        assert_eq!(fs.formats().len(), 2);
    }

    #[test]
    fn format_set_print() {
        let fs = FormatSet::print();
        assert!(fs.formats().contains(&OutputFormat::Pdf));
        assert!(fs.formats().contains(&OutputFormat::Png));
        assert_eq!(fs.dpi(), 300);
    }

    #[test]
    fn format_properties() {
        assert_eq!(OutputFormat::Svg.extension(), "svg");
        assert_eq!(OutputFormat::Png.extension(), "png");
        assert_eq!(OutputFormat::Svg.mime_type(), "image/svg+xml");
    }

    #[test]
    fn plot_saver_svg() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\"></svg>";
        let saver = PlotSaver::new(svg);
        assert_eq!(saver.svg_content(), svg);
    }

    #[test]
    fn save_as_svg() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\"></svg>";
        let saver = PlotSaver::new(svg);
        let result = saver.save_as("/tmp/test_plot", OutputFormat::Svg, &FormatSet::new());
        assert!(result.success);
        assert!(result.path.exists());
    }

    #[test]
    fn save_html() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\"><rect width=\"100\" height=\"100\" fill=\"blue\"/></svg>";
        let saver = PlotSaver::new(svg)
            .with_title("Test Plot")
            .with_description("A test plot");
        let result = saver.save_as("/tmp/test_plot", OutputFormat::Html, &FormatSet::new());
        assert!(result.success);
        let content = std::fs::read_to_string(&result.path).unwrap();
        assert!(content.contains("Test Plot"));
    }

    #[test]
    fn format_set_builder() {
        let fs = FormatSet::new()
            .add(OutputFormat::Svg)
            .add(OutputFormat::Png)
            .add(OutputFormat::Svg); // Duplicate should not be added
        assert_eq!(fs.formats().len(), 2);
    }
}
