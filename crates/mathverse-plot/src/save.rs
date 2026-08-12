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
    /// JPEG (lossy raster, requires the `png` feature for the rasterizer).
    Jpeg,
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
            OutputFormat::Jpeg => "jpg",
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
            OutputFormat::Jpeg => "image/jpeg",
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

/// Raster save options — the analogue of matplotlib's `savefig` kwargs.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterOptions {
    /// Crop to the tight bounding box of the drawing (removes uniform margins).
    pub tight_bbox: bool,
    /// Transparent background (PNG only; JPEG has no alpha channel).
    pub transparent: bool,
    /// Background color as `#rrggbb` when not transparent.
    pub facecolor: Option<String>,
    /// JPEG quality 1..=100 (ignored for other formats).
    pub quality: u8,
}

impl Default for RasterOptions {
    fn default() -> Self {
        Self {
            tight_bbox: false,
            transparent: false,
            facecolor: None,
            quality: 90,
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
    /// Raster save options.
    raster: RasterOptions,
}

impl PlotSaver {
    /// Create a new saver with SVG content.
    ///
    /// Accepts any string-like value (`&str`, `String`, `&String`).
    pub fn new(svg_content: impl AsRef<str>) -> Self {
        Self {
            svg_content: svg_content.as_ref().to_string(),
            width: 800,
            height: 600,
            title: String::new(),
            description: String::new(),
            plot_data: None,
            raster: RasterOptions::default(),
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
            raster: RasterOptions::default(),
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
            raster: RasterOptions::default(),
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

    /// Set raster save options (tight bbox, transparency, facecolor, quality).
    pub fn with_raster_options(mut self, raster: RasterOptions) -> Self {
        self.raster = raster;
        self
    }

    /// Builder-style: enable tight-bbox cropping for raster output.
    pub fn with_tight_bbox(mut self, tight: bool) -> Self {
        self.raster.tight_bbox = tight;
        self
    }

    /// Builder-style: make raster output transparent (PNG only).
    pub fn with_transparent(mut self, transparent: bool) -> Self {
        self.raster.transparent = transparent;
        self
    }

    /// Builder-style: set the raster background color (ignored when transparent).
    pub fn with_facecolor(mut self, facecolor: impl Into<String>) -> Self {
        self.raster.facecolor = Some(facecolor.into());
        self
    }

    /// Builder-style: set JPEG quality 1..=100.
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.raster.quality = quality.clamp(1, 100);
        self
    }

    /// Raster bytes for the current scene (PNG or JPEG), used by both file
    /// export and notebook inline embedding. Requires the `png` feature.
    #[cfg(feature = "png")]
    pub fn raster_bytes(&self, format: OutputFormat, dpi: u32, scale: f64) -> Result<Vec<u8>, String> {
        let pixmap = self.render_pixmap(dpi, scale)?;
        match format {
            OutputFormat::Jpeg => {
                let (w, h) = (pixmap.width(), pixmap.height());
                let data = pixmap.data();
                let mut rgba = vec![0u8; data.len()];
                for (dst, px) in rgba.chunks_exact_mut(4).zip(data.chunks_exact(4)) {
                    // tiny-skia stores premultiplied BGRA; unpremultiply to RGBA.
                    let (b, g, r, a) = (px[0], px[1], px[2], px[3]);
                    if a == 255 {
                        dst.copy_from_slice(&[r, g, b, 255]);
                    } else if a == 0 {
                        dst.copy_from_slice(&[0, 0, 0, 0]);
                    } else {
                        let inv = 255.0 / f32::from(a);
                        let rr = ((f32::from(r) * inv).round() as u8).min(255);
                        let gg = ((f32::from(g) * inv).round() as u8).min(255);
                        let bb = ((f32::from(b) * inv).round() as u8).min(255);
                        dst.copy_from_slice(&[rr, gg, bb, a]);
                    }
                }
                let img =
                    image::RgbaImage::from_raw(w, h, rgba).ok_or("pixmap size mismatch")?;
                let rgb = image::DynamicImage::ImageRgba8(img).to_rgb8();
                let mut out = std::io::Cursor::new(Vec::new());
                let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, self.raster.quality);
                encoder.encode(&rgb, w, h, image::ExtendedColorType::Rgb8).map_err(|e| format!("jpeg encode: {e}"))?;
                Ok(out.into_inner())
            }
            _ => pixmap
                .encode_png()
                .map_err(|e| format!("png encode: {e}")),
        }
    }

    /// Rasterize the SVG to a tiny-skia pixmap honoring facecolor/transparency.
    #[cfg(feature = "png")]
    fn render_pixmap(&self, dpi: u32, scale: f64) -> Result<tiny_skia::Pixmap, String> {
        let width = (self.width as f64 * scale) as u32;
        let height = (self.height as f64 * scale) as u32;

        let mut opts = usvg::Options::default();
        opts.resources_dir = std::env::current_dir().ok();
        opts.dpi = dpi as f32;

        let tree = usvg::Tree::from_data(self.svg_content.as_bytes(), &opts)
            .map_err(|e| format!("usvg parse: {e}"))?;

        let mut pixmap = tiny_skia::Pixmap::new(width, height)
            .ok_or_else(|| "Failed to create pixmap".to_string())?;

        let bg = if self.raster.transparent {
            tiny_skia::Color::TRANSPARENT
        } else if let Some(ref hex) = self.raster.facecolor {
            parse_hex_color(hex).unwrap_or(tiny_skia::Color::WHITE)
        } else {
            tiny_skia::Color::WHITE
        };
        pixmap.fill(bg);

        let transform = tiny_skia::Transform::from_scale(
            width as f32 / tree.size().width(),
            height as f32 / tree.size().height(),
        );
        resvg::render(&tree, transform, &mut pixmap.as_mut());

        if self.raster.tight_bbox && width > 2 && height > 2 {
            // Crop to the non-background region (matches bbox_inches='tight').
            let data = pixmap.data().to_vec();
            let bg = if self.raster.transparent {
                [0u8, 0, 0, 0]
            } else if let Some(ref hex) = self.raster.facecolor {
                hex_to_rgba(hex).unwrap_or([255, 255, 255, 255])
            } else {
                [255, 255, 255, 255]
            };
            let mut min_x = width as usize;
            let mut min_y = height as usize;
            let mut max_x = 0usize;
            let mut max_y = 0usize;
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let i = (y * width as usize + x) * 4;
                    if data[i] != bg[0] || data[i + 1] != bg[1] || data[i + 2] != bg[2] || data[i + 3] != bg[3] {
                        min_x = min_x.min(x);
                        max_x = max_x.max(x);
                        min_y = min_y.min(y);
                        max_y = max_y.max(y);
                    }
                }
            }
            if max_x >= min_x && max_y >= min_y {
                let x0 = min_x as u32;
                let y0 = min_y as u32;
                let w = (max_x - min_x + 1) as u32;
                let h = (max_y - min_y + 1) as u32;
                let _rect = tiny_skia::IntRect::from_xywh(x0 as i32, y0 as i32, w, h)
                    .ok_or("bad crop rect")?;
                let mut cropped = tiny_skia::Pixmap::new(w, h)
                    .ok_or("crop pixmap alloc failed")?;
                // Clone the source rows into the new pixmap manually.
                let dst = cropped.data_mut();
                for yy in 0..h as usize {
                    let row_src =
                        &data[(y0 as usize + yy) * width as usize * 4..(y0 as usize + yy + 1) * width as usize * 4];
                    let row_dst = &mut dst[yy * w as usize * 4..(yy + 1) * w as usize * 4];
                    row_dst.copy_from_slice(&row_src[x0 as usize * 4..(x0 as usize + w as usize) * 4]);
                }
                pixmap = cropped;
            }
        }
        Ok(pixmap)
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
        #[cfg(feature = "png")]
        let formats = _formats;
        let content = match format {
            OutputFormat::Svg => self.generate_svg(),
            OutputFormat::Png | OutputFormat::Jpeg => {
                // Raster formats require the png feature (rasterizer stack).
                #[cfg(feature = "png")]
                {
                    match self.raster_bytes(format, formats.dpi(), formats.scale()) {
                        Ok(bytes) => match std::fs::write(path, &bytes) {
                            Ok(()) => return ExportResult::success(
                                format,
                                path.to_path_buf(),
                                bytes.len() as u64,
                            ),
                            Err(e) => return ExportResult::failed(
                                format,
                                path.to_path_buf(),
                                e.to_string(),
                            ),
                        },
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
                        "raster export requires the png feature".into(),
                    );
                }
            }
            OutputFormat::Pdf => {
                #[cfg(feature = "pdf")]
                {
                    match self.generate_pdf() {
                        Ok(bytes) => match std::fs::write(path, &bytes) {
                            Ok(()) => return ExportResult::success(
                                format,
                                path.to_path_buf(),
                                bytes.len() as u64,
                            ),
                            Err(e) => return ExportResult::failed(
                                format,
                                path.to_path_buf(),
                                e.to_string(),
                            ),
                        },
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

    /// Generate PNG bytes (requires png feature). Kept for backward compat.
    #[cfg(feature = "png")]
    #[allow(dead_code)]
    fn generate_png(&self, dpi: u32, scale: f64) -> Result<Vec<u8>, String> {
        self.raster_bytes(OutputFormat::Png, dpi, scale)
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
            escape_html(&self.title)
        };

        let description = if self.description.is_empty() {
            format!("{}x{} plot", self.width, self.height)
        } else {
            escape_html(&self.description)
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

    /// Data-URI PNG for embedding in notebooks / HTML (`<img>` or markdown).
    /// This is the analogue of `%matplotlib inline` in Jupyter.
    #[cfg(feature = "png")]
    pub fn inline_png_data_uri(&self, dpi: u32, scale: f64) -> Result<String, String> {
        let bytes = self.raster_bytes(OutputFormat::Png, dpi, scale)?;
        let mut b64 = String::new();
        use std::fmt::Write;
        for chunk in bytes.chunks(48) {
            let _ = write!(b64, "{}", base64_encode(chunk));
        }
        Ok(format!("data:image/png;base64,{b64}"))
    }

    /// PNG bytes for notebook embedding (no file I/O).
    #[cfg(feature = "png")]
    pub fn to_png_bytes(&self) -> Result<Vec<u8>, String> {
        self.raster_bytes(OutputFormat::Png, 96, 1.0)
    }

    /// An HTML `<img>` tag embedding the rendered PNG — paste this into a
    /// notebook cell output for inline display.
    #[cfg(feature = "png")]
    pub fn inline_png_tag(&self) -> Result<String, String> {
        let uri = self.inline_png_data_uri(96, 1.0)?;
        Ok(format!("<img src=\"{uri}\" alt=\"plot\"/>"))
    }

    /// Save to PNG by default (default format).
    ///
    /// Returns `Ok(())` on success, or an I/O error describing the failure.
    pub fn save_png(&self, path: &str) -> std::io::Result<()> {
        let result = self.save_as(path, OutputFormat::Png, &FormatSet::png());
        if result.success {
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                result.error.unwrap_or_else(|| "PNG export failed".to_string()),
            ))
        }
    }

    /// Get dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Parse `#rrggbb` (or `rrggbb`) into a tiny-skia color. Returns None for junk.
#[cfg(feature = "png")]
fn parse_hex_color(hex: &str) -> Option<tiny_skia::Color> {
    hex_to_rgba(hex).map(|[r, g, b, a]| {
        tiny_skia::Color::from_rgba8(r, g, b, a)
    })
}

/// Parse `#rrggbb` into `[r, g, b, 255]`.
#[cfg(feature = "png")]
fn hex_to_rgba(hex: &str) -> Option<[u8; 4]> {
    let s = hex.trim().trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some([r, g, b, 255])
}

/// Minimal base64 encoder (RFC 4648) — avoids pulling a dep just for a data URI.
#[cfg(feature = "png")]
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Escape user-provided text for safe embedding in HTML output.
///
/// Prevents HTML injection / XSS when titles, descriptions, or labels
/// contain markup characters.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
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
        // `new()` defaults to a PNG-only set.
        let fs = FormatSet::new();
        assert_eq!(fs.formats(), &[OutputFormat::Png]);

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
    fn html_escapes_user_text() {
        // Titles/descriptions with markup must be escaped to prevent XSS.
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>";
        let saver = PlotSaver::new(svg)
            .with_title("<script>alert('xss')</script>")
            .with_description("a & b < c");
        let html = saver.generate_html();
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(html.contains("a &amp; b &lt; c"));
    }

    #[test]
    fn escape_html_basic() {
        assert_eq!(escape_html("<>&\"'"), "&lt;&gt;&amp;&quot;&#39;");
        assert_eq!(escape_html("plain text"), "plain text");
    }

    #[test]
    fn format_set_builder() {
        let fs = FormatSet::new()
            .add(OutputFormat::Svg)
            .add(OutputFormat::Png)
            .add(OutputFormat::Svg); // Duplicate should not be added
        assert_eq!(fs.formats().len(), 2);
    }

    #[test]
    fn jpeg_format_properties() {
        assert_eq!(OutputFormat::Jpeg.extension(), "jpg");
        assert_eq!(OutputFormat::Jpeg.mime_type(), "image/jpeg");
    }

    #[test]
    fn raster_options_defaults() {
        let o = RasterOptions::default();
        assert!(!o.tight_bbox);
        assert!(!o.transparent);
        assert_eq!(o.quality, 90);
    }

    #[test]
    fn hex_color_parsing() {
        assert_eq!(hex_to_rgba("#ff0000"), Some([255, 0, 0, 255]));
        assert_eq!(hex_to_rgba("00ff00"), Some([0, 255, 0, 255]));
        assert_eq!(hex_to_rgba("#12345"), None);
        assert_eq!(hex_to_rgba("nope"), None);
    }

    #[test]
    fn base64_roundtrip_known_vector() {
        // "Man" -> TWFu (RFC 4648 test vector)
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_encode(b"M"), "TQ==");
    }

    #[cfg(feature = "png")]
    #[test]
    fn jpeg_export_produces_magic_bytes() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
            <rect width="200" height="100" fill="#3366ff"/>
        </svg>"##;
        let saver = PlotSaver::new(svg).with_quality(85);
        let bytes = saver.raster_bytes(OutputFormat::Jpeg, 96, 1.0).unwrap();
        // JPEG SOI marker: FF D8 FF
        assert_eq!(&bytes[..3], &[0xFF, 0xD8, 0xFF]);
        // JPEG EOI marker near the end: FF D9
        assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9]);
        assert!(bytes.len() > 100);
    }

    #[cfg(feature = "png")]
    #[test]
    fn png_inline_data_uri() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50"></svg>"#;
        let uri = PlotSaver::new(svg).inline_png_data_uri(96, 1.0).unwrap();
        assert!(uri.starts_with("data:image/png;base64,"));
        assert!(uri.len() > 100);
        let tag = PlotSaver::new(svg).inline_png_tag().unwrap();
        assert!(tag.starts_with("<img src=\"data:image/png"));
    }

    #[cfg(feature = "png")]
    #[test]
    fn tight_bbox_crops_background() {
        // A mostly-white scene with a small dark mark: tight bbox must shrink
        // the output dimensions substantially.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
            <rect x="150" y="150" width="100" height="100" fill="black"/>
        </svg>"#;
        let saver = PlotSaver::new(svg).with_tight_bbox(true);
        let bytes = saver.raster_bytes(OutputFormat::Png, 96, 1.0).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert!(decoded.width() < 400, "tight bbox width {}", decoded.width());
        assert!(decoded.height() < 400);
    }
}
