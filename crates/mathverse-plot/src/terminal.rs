//! Terminal (ASCII) plotting backend

use crate::backend::PlotData;
use crate::common::{plot_bounds, DataSeries, PlotConfig};

/// Terminal plot generator
pub struct TerminalPlot {
    config: PlotConfig,
    series: Vec<DataSeries>,
    width: usize,
    height: usize,
}

impl TerminalPlot {
    /// Create a new terminal plot
    pub fn new(config: PlotConfig) -> Self {
        TerminalPlot {
            config,
            series: Vec::new(),
            width: 80,
            height: 24,
        }
    }

    /// Set the terminal dimensions
    pub fn with_dimensions(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Add a data series to the plot
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    /// Generate the ASCII plot string
    pub fn generate(&self) -> String {
        if self.series.is_empty() {
            return String::from("No data to plot\n");
        }

        // Calculate data ranges
        let (x_min, x_max, y_min, y_max) = self.calculate_ranges();

        // Create plot grid
        let mut grid = vec![vec![' '; self.width]; self.height];

        // Plot data points
        for series in &self.series {
            for point in &series.points {
                let x = ((point.x - x_min) / (x_max - x_min) * (self.width - 2) as f64) as usize;
                let y = ((point.y - y_min) / (y_max - y_min) * (self.height - 2) as f64) as usize;

                if x < self.width && y < self.height {
                    grid[self.height - 1 - y][x] = '*';
                }
            }
        }

        // Draw axes
        for i in 0..self.height {
            grid[i][0] = '|';
        }
        for j in 0..self.width {
            grid[self.height - 1][j] = '-';
        }
        grid[self.height - 1][0] = '+';

        // Convert grid to string
        let mut output = String::new();

        // Title
        if !self.config.title.is_empty() {
            output.push_str(&self.config.title);
            output.push('\n');
        }

        // Y-axis label
        if !self.config.y_label.is_empty() {
            output.push_str(&self.config.y_label);
            output.push('\n');
        }

        // Plot
        for row in &grid {
            output.push_str(&row.iter().collect::<String>());
            output.push('\n');
        }

        // X-axis label
        if !self.config.x_label.is_empty() {
            output.push_str(&self.config.x_label);
            output.push('\n');
        }

        // Legend
        if self.config.show_legend {
            output.push_str("\nLegend:\n");
            for series in &self.series {
                output.push_str(&format!("  * - {}\n", series.name));
            }
        }

        output
    }

    /// Padded bounds over all series; falls back to `0..1` when empty.
    fn calculate_ranges(&self) -> (f64, f64, f64, f64) {
        let (x, y) = plot_bounds(&self.series);
        let x = x.pad(0.05);
        let y = y.pad(0.05);
        (x.min, x.max, y.min, y.max)
    }
}

impl Default for TerminalPlot {
    fn default() -> Self {
        Self::new(PlotConfig::default())
    }
}

impl crate::backend::Backend for TerminalPlot {
    fn generate(&self, data: &PlotData) -> crate::error::PlotResult<String> {
        let mut tp =
            TerminalPlot::new(data.config.clone()).with_dimensions(self.width, self.height);
        for s in &data.series {
            tp.add_series(s.clone());
        }
        Ok(tp.generate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataPoint;

    #[test]
    fn test_terminal_plot_creation() {
        let config = PlotConfig::new().with_title("Test Plot".to_string());

        let mut plot = TerminalPlot::new(config).with_dimensions(40, 20);

        let points = vec![
            DataPoint::new(1.0, 2.0),
            DataPoint::new(2.0, 4.0),
            DataPoint::new(3.0, 6.0),
        ];

        let series = DataSeries::new("Test Series".to_string(), points);
        plot.add_series(series);

        let ascii = plot.generate();
        assert!(ascii.contains("Test Plot"));
        assert!(ascii.contains("*"));
    }
}
