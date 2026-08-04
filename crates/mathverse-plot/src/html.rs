//! HTML plotting backend

use crate::backend::PlotData;
use crate::common::{DataSeries, PlotConfig};
use crate::svg::SvgPlot;

/// HTML plot generator (wraps SVG in HTML)
pub struct HtmlPlot {
    config: PlotConfig,
    series: Vec<DataSeries>,
}

impl HtmlPlot {
    /// Create a new HTML plot
    pub fn new(config: PlotConfig) -> Self {
        HtmlPlot {
            config,
            series: Vec::new(),
        }
    }

    /// Add a data series to the plot
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    /// Generate the HTML string
    pub fn generate(&self) -> String {
        let mut html = String::new();

        // HTML header
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html>\n");
        html.push_str("<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str("  <title>");
        html.push_str(&self.config.title);
        html.push_str("</title>\n");
        html.push_str("  <style>\n");
        html.push_str("    body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str("    .plot-container { text-align: center; }\n");
        html.push_str("    h1 { color: #333; }\n");
        html.push_str("  </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");

        // Title
        html.push_str("  <div class=\"plot-container\">\n");
        if !self.config.title.is_empty() {
            html.push_str("    <h1>");
            html.push_str(&self.config.title);
            html.push_str("</h1>\n");
        }

        // Generate SVG plot
        let mut svg_plot = SvgPlot::new(self.config.clone());
        for series in &self.series {
            svg_plot.add_series(series.clone());
        }
        let svg = svg_plot.generate();

        html.push_str("    <div>\n");
        html.push_str(&svg);
        html.push_str("    </div>\n");

        html.push_str("  </div>\n");
        html.push_str("</body>\n");
        html.push_str("</html>\n");

        html
    }
}

impl crate::backend::Backend for HtmlPlot {
    fn generate(&self, data: &PlotData) -> crate::error::PlotResult<String> {
        let svg_content = <SvgPlot as crate::backend::Backend>::generate(
            &SvgPlot::new(data.config.clone()),
            data,
        )?;
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n  <title>");
        html.push_str(&data.config.title);
        html.push_str(
            "</title>\n  <style>\n    body { font-family: Arial, sans-serif; margin: 20px; }\n",
        );
        html.push_str("    .plot-container { text-align: center; }\n    h1 { color: #333; }\n");
        html.push_str("  </style>\n</head>\n<body>\n  <div class=\"plot-container\">\n");
        if !data.config.title.is_empty() {
            html.push_str("    <h1>");
            html.push_str(&data.config.title);
            html.push_str("</h1>\n");
        }
        html.push_str("    <div>\n");
        html.push_str(&svg_content);
        html.push_str("    </div>\n  </div>\n</body>\n</html>\n");
        Ok(html)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataPoint;

    #[test]
    fn test_html_plot_creation() {
        let config = PlotConfig::new()
            .with_title("Test Plot".to_string())
            .with_dimensions(800, 600);

        let mut plot = HtmlPlot::new(config);

        let points = vec![
            DataPoint::new(1.0, 2.0),
            DataPoint::new(2.0, 4.0),
            DataPoint::new(3.0, 6.0),
        ];

        let series = DataSeries::new("Test Series".to_string(), points);
        plot.add_series(series);

        let html = plot.generate();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("Test Plot"));
    }
}
