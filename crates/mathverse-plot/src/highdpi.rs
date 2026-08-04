//! High-DPI export for publication-quality output.

/// Configuration for high-DPI rendering.
#[derive(Debug, Clone)]
pub struct DpiConfig {
    /// DPI (dots per inch) for the output.
    pub dpi: u32,
    /// Scale factor (1.0 = 100%, 2.0 = 200%).
    pub scale: f64,
}

impl Default for DpiConfig {
    fn default() -> Self {
        Self {
            dpi: 96, // Standard screen DPI
            scale: 1.0,
        }
    }
}

impl DpiConfig {
    /// Create a new DPI config.
    pub fn new(dpi: u32) -> Self {
        Self {
            dpi,
            scale: dpi as f64 / 96.0,
        }
    }

    /// Standard screen resolution (96 DPI).
    pub fn screen() -> Self {
        Self::new(96)
    }

    /// High resolution (150 DPI) for presentations.
    pub fn presentation() -> Self {
        Self::new(150)
    }

    /// Print quality (300 DPI).
    pub fn print() -> Self {
        Self::new(300)
    }

    /// High print quality (600 DPI).
    pub fn high_quality() -> Self {
        Self::new(600)
    }

    /// Compute scaled dimensions.
    pub fn scale_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        (
            (width as f64 * self.scale).round() as u32,
            (height as f64 * self.scale).round() as u32,
        )
    }

    /// Compute scaled padding.
    pub fn scale_padding(&self, padding: f64) -> f64 {
        padding * self.scale
    }

    /// Compute scaled font size.
    pub fn scale_font_size(&self, size: f64) -> f64 {
        size * self.scale
    }

    /// Compute scaled line width.
    pub fn scale_line_width(&self, width: f64) -> f64 {
        width * self.scale
    }
}

/// Preset DPI configurations.
pub mod presets {
    use super::DpiConfig;

    /// Standard screen (96 DPI).
    pub fn screen() -> DpiConfig {
        DpiConfig::screen()
    }

    /// Retina/HiDPI (192 DPI).
    pub fn retina() -> DpiConfig {
        DpiConfig::new(192)
    }

    /// Presentation (150 DPI).
    pub fn presentation() -> DpiConfig {
        DpiConfig::presentation()
    }

    /// Print quality (300 DPI).
    pub fn print() -> DpiConfig {
        DpiConfig::print()
    }

    /// High print quality (600 DPI).
    pub fn high_quality() -> DpiConfig {
        DpiConfig::high_quality()
    }
}

/// Generate an SVG header with proper dimensions and viewBox for high-DPI.
pub fn svg_header(width: u32, height: u32, dpi_config: &DpiConfig) -> String {
    let (scaled_w, scaled_h) = dpi_config.scale_dimensions(width, height);

    if dpi_config.scale == 1.0 {
        format!(
            r#"<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">"#,
        )
    } else {
        // Use viewBox for resolution independence
        format!(
            r#"<svg width="{scaled_w}" height="{scaled_h}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">"#,
        )
    }
}

/// Generate an SVG header with physical dimensions for print.
pub fn svg_header_physical(
    width: u32,
    height: u32,
    dpi_config: &DpiConfig,
) -> String {
    let width_in = width as f64 / dpi_config.dpi as f64;
    let height_in = height as f64 / dpi_config.dpi as f64;

    format!(
        r#"<svg width="{width_in}in" height="{height_in}in" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">"#,
    )
}

/// Generate a PNG metadata string for high-DPI.
pub fn png_metadata(dpi_config: &DpiConfig) -> PngMetadata {
    PngMetadata {
        dpi: dpi_config.dpi,
        width: 0,
        height: 0,
    }
}

/// PNG metadata for high-DPI export.
#[derive(Debug, Clone)]
pub struct PngMetadata {
    /// DPI value.
    pub dpi: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl PngMetadata {
    /// Create new PNG metadata.
    pub fn new(dpi: u32, width: u32, height: u32) -> Self {
        Self { dpi, width, height }
    }

    /// Get the physical dimensions in inches.
    pub fn physical_size_inches(&self) -> (f64, f64) {
        (
            self.width as f64 / self.dpi as f64,
            self.height as f64 / self.dpi as f64,
        )
    }

    /// Get the physical dimensions in millimeters.
    pub fn physical_size_mm(&self) -> (f64, f64) {
        let (w_in, h_in) = self.physical_size_inches();
        (w_in * 25.4, h_in * 25.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_config_screen() {
        let config = DpiConfig::screen();
        assert_eq!(config.dpi, 96);
        assert_eq!(config.scale, 1.0);
    }

    #[test]
    fn dpi_config_retina() {
        let config = DpiConfig::new(192);
        let (w, h) = config.scale_dimensions(800, 600);
        assert_eq!(w, 1600);
        assert_eq!(h, 1200);
    }

    #[test]
    fn svg_header_normal() {
        let config = DpiConfig::screen();
        let header = svg_header(800, 600, &config);
        assert!(header.contains("width=\"800\""));
        assert!(!header.contains("viewBox"));
    }

    #[test]
    fn svg_header_retina() {
        let config = DpiConfig::new(192);
        let header = svg_header(800, 600, &config);
        assert!(header.contains("width=\"1600\""));
        assert!(header.contains("viewBox=\"0 0 800 600\""));
    }

    #[test]
    fn png_metadata() {
        let meta = PngMetadata::new(300, 2400, 1800);
        let (w_in, h_in) = meta.physical_size_inches();
        assert!((w_in - 8.0).abs() < 0.01);
        assert!((h_in - 6.0).abs() < 0.01);
    }
}
