//! Real-time streaming plot support.
//!
//! This module provides support for real-time data streaming and live updates
//! to plots, useful for monitoring, IoT, and live data visualization.
//!
//! # Example
//!
//! ```rust,no_run
//! use mathverse_plot::streaming::{StreamingPlot, StreamConfig};
//! use std::time::Duration;
//!
//! let config = StreamConfig::new()
//!     .with_buffer_size(1000)
//!     .with_update_interval(Duration::from_millis(100));
//!
//! let mut plot = StreamingPlot::new(config);
//!
//! // In a real application, you'd feed data from a stream
//! plot.push(1.0, 2.0);
//! plot.push(2.0, 4.0);
//!
//! let html = plot.render_html();
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::common::DataPoint;
use crate::style::Color;

/// Configuration for streaming plots.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum number of data points to keep in the buffer.
    pub buffer_size: usize,
    /// Update interval for the plot.
    pub update_interval: Duration,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Line color.
    pub line_color: Color,
    /// Background color.
    pub background_color: Color,
    /// Line width.
    pub line_width: f32,
    /// Enable auto-scaling.
    pub auto_scale: bool,
    /// Enable scroll (windowed view).
    pub scroll: bool,
    /// Window size for scrolling (number of visible points).
    pub window_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            update_interval: Duration::from_millis(100),
            width: 800,
            height: 400,
            line_color: Color::BLUE,
            background_color: Color::WHITE,
            line_width: 2.0,
            auto_scale: true,
            scroll: true,
            window_size: 100,
        }
    }
}

impl StreamConfig {
    /// Create a new stream config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the buffer size.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the update interval.
    pub fn with_update_interval(mut self, interval: Duration) -> Self {
        self.update_interval = interval;
        self
    }

    /// Set the canvas dimensions.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the line color.
    pub fn with_line_color(mut self, color: Color) -> Self {
        self.line_color = color;
        self
    }

    /// Set the background color.
    pub fn with_background_color(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Set the line width.
    pub fn with_line_width(mut self, width: f32) -> Self {
        self.line_width = width;
        self
    }

    /// Enable or disable auto-scaling.
    pub fn with_auto_scale(mut self, enable: bool) -> Self {
        self.auto_scale = enable;
        self
    }

    /// Enable or disable scrolling.
    pub fn with_scroll(mut self, enable: bool) -> Self {
        self.scroll = enable;
        self
    }

    /// Set the window size for scrolling.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }
}

/// A streaming plot that supports real-time data updates.
pub struct StreamingPlot {
    /// Data buffer.
    data: VecDeque<DataPoint>,
    /// Configuration.
    config: StreamConfig,
    /// Start time for timestamping.
    start_time: Instant,
    /// Total points received.
    total_points: usize,
}

impl StreamingPlot {
    /// Create a new streaming plot.
    pub fn new(config: StreamConfig) -> Self {
        Self {
            data: VecDeque::with_capacity(config.buffer_size),
            config,
            start_time: Instant::now(),
            total_points: 0,
        }
    }

    /// Push a new data point with auto-generated x (time-based).
    pub fn push(&mut self, y: f64) {
        let x = self.start_time.elapsed().as_secs_f64();
        self.push_xy(x, y);
    }

    /// Push a new data point with explicit x and y values.
    pub fn push_xy(&mut self, x: f64, y: f64) {
        if self.data.len() >= self.config.buffer_size {
            self.data.pop_front();
        }
        self.data.push_back(DataPoint::new(x, y));
        self.total_points += 1;
    }

    /// Push multiple data points.
    pub fn push_batch(&mut self, points: &[(f64, f64)]) {
        for &(x, y) in points {
            self.push_xy(x, y);
        }
    }

    /// Get the current data as a vector.
    pub fn data(&self) -> Vec<DataPoint> {
        self.data.iter().cloned().collect()
    }

    /// Get the visible window of data.
    pub fn visible_data(&self) -> Vec<DataPoint> {
        if self.config.scroll {
            let window_start = self.data.len().saturating_sub(self.config.window_size);
            self.data.iter().skip(window_start).cloned().collect()
        } else {
            self.data.iter().cloned().collect()
        }
    }

    /// Get the number of data points.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the total number of points received.
    pub fn total_points(&self) -> usize {
        self.total_points
    }

    /// Get the data bounds.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        if self.data.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for point in &self.data {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }

        (min_x, max_x, min_y, max_y)
    }

    /// Render the current state to a standalone HTML file with live updates.
    pub fn render_html(&self) -> String {
        let data_json = self.points_to_json();
        let bg_color = self.config.background_color.to_hex();
        let line_color = self.config.line_color.to_hex();

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Streaming Plot - {} Points</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .stats {{
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        canvas {{
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .controls {{
            margin-top: 15px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }}
        button {{
            padding: 8px 16px;
            margin-right: 10px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
        }}
        button:hover {{
            background: #0056b3;
        }}
        .status {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            margin-left: 10px;
        }}
        .status-live {{
            background: #28a745;
            color: white;
        }}
        .status-paused {{
            background: #ffc107;
            color: black;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-Time Streaming Plot <span class="status status-live" id="status">LIVE</span></h1>
        <div class="stats" id="stats">Loading...</div>
        <canvas id="canvas" width="{}" height="{}"></canvas>
        <div class="controls">
            <button onclick="toggleStreaming()">Pause/Resume</button>
            <button onclick="clearData()">Clear Data</button>
            <button onclick="exportCSV()">Export CSV</button>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const statsEl = document.getElementById('stats');
        const statusEl = document.getElementById('status');

        // Data
        let data = {};
        let isStreaming = true;
        let updateIntervalId = null;

        // Configuration
        const config = {{
            bufferSize: {},
            windowSize: {},
            autoScale: true,
            scroll: {}
        }};

        // State
        let offsetX = 0;
        let scale = 1.0;

        // Render function
        function render() {{
            ctx.fillStyle = '{}';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            if (data.length === 0) {{
                ctx.fillStyle = '#999';
                ctx.font = '14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('Waiting for data...', canvas.width / 2, canvas.height / 2);
                return;
            }}

            // Calculate visible window
            let visibleData = data;
            if (config.scroll) {{
                const windowStart = Math.max(0, data.length - config.windowSize);
                visibleData = data.slice(windowStart);
            }}

            // Calculate bounds
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            for (const point of visibleData) {{
                minX = Math.min(minX, point[0]);
                maxX = Math.max(maxX, point[0]);
                minY = Math.min(minY, point[1]);
                maxY = Math.max(maxY, point[1]);
            }}

            // Add padding
            const padding = 40;
            const plotWidth = canvas.width - 2 * padding;
            const plotHeight = canvas.height - 2 * padding;

            // Draw axes
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padding, padding);
            ctx.lineTo(padding, canvas.height - padding);
            ctx.lineTo(canvas.width - padding, canvas.height - padding);
            ctx.stroke();

            // Draw grid
            ctx.strokeStyle = '#eee';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 5; i++) {{
                const x = padding + (plotWidth * i) / 5;
                const y = padding + (plotHeight * i) / 5;
                ctx.beginPath();
                ctx.moveTo(x, padding);
                ctx.lineTo(x, canvas.height - padding);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(canvas.width - padding, y);
                ctx.stroke();
            }}

            // Draw data line
            if (visibleData.length > 1) {{
                ctx.strokeStyle = '{}';
                ctx.lineWidth = {};
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();

                for (let i = 0; i < visibleData.length; i++) {{
                    const x = padding + ((visibleData[i][0] - minX) / (maxX - minX || 1)) * plotWidth;
                    const y = canvas.height - padding - ((visibleData[i][1] - minY) / (maxY - minY || 1)) * plotHeight;

                    if (i === 0) {{
                        ctx.moveTo(x, y);
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }}

                ctx.stroke();
            }}

            // Update stats
            statsEl.textContent = `Total: ${{data.length.toLocaleString()}} points | Visible: ${{visibleData.length}} | Range: [${{minY.toFixed(2)}}, ${{maxY.toFixed(2)}}]`;
        }}

        // Simulate incoming data
        function addData() {{
            if (!isStreaming) return;

            const x = data.length;
            const y = Math.sin(x * 0.1) * 50 + Math.random() * 20;

            data.push([x, y]);

            if (data.length > config.bufferSize) {{
                data.shift();
            }}

            render();
        }}

        // Control functions
        function toggleStreaming() {{
            isStreaming = !isStreaming;
            statusEl.textContent = isStreaming ? 'LIVE' : 'PAUSED';
            statusEl.className = `status status-${{isStreaming ? 'live' : 'paused'}}`;
        }}

        function clearData() {{
            data = [];
            render();
        }}

        function exportCSV() {{
            let csv = 'x,y\\n';
            for (const point of data) {{
                csv += `${{point[0]}},${{point[1]}}\\n`;
            }}

            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'streaming_data.csv';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Initial render
        render();

        // Start streaming simulation
        updateIntervalId = setInterval(addData, {});
    </script>
</body>
</html>"#,
            self.total_points,
            self.config.width,
            self.config.height,
            data_json,
            self.config.buffer_size,
            self.config.window_size,
            self.config.scroll,
            bg_color,
            line_color,
            self.config.line_width,
            self.config.update_interval.as_millis()
        )
    }

    /// Convert points to JSON array.
    fn points_to_json(&self) -> String {
        let mut json = String::from("[");
        for (i, point) in self.data.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("[{},{}]", point.x, point.y));
        }
        json.push(']');
        json
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_plot_creation() {
        let config = StreamConfig::new();
        let plot = StreamingPlot::new(config);
        assert!(plot.is_empty());
        assert_eq!(plot.len(), 0);
    }

    #[test]
    fn streaming_plot_push() {
        let config = StreamConfig::new().with_buffer_size(5);
        let mut plot = StreamingPlot::new(config);

        for i in 0..10 {
            plot.push(i as f64);
        }

        assert_eq!(plot.len(), 5);
        assert_eq!(plot.total_points(), 10);
    }

    #[test]
    fn streaming_plot_bounds() {
        let config = StreamConfig::new();
        let mut plot = StreamingPlot::new(config);

        plot.push_xy(0.0, 0.0);
        plot.push_xy(10.0, 20.0);

        let (min_x, max_x, min_y, max_y) = plot.bounds();
        assert_eq!(min_x, 0.0);
        assert_eq!(max_x, 10.0);
        assert_eq!(min_y, 0.0);
        assert_eq!(max_y, 20.0);
    }

    #[test]
    fn streaming_plot_html() {
        let config = StreamConfig::new();
        let mut plot = StreamingPlot::new(config);

        plot.push(1.0);
        plot.push(2.0);

        let html = plot.render_html();
        assert!(html.contains("<html"));
        assert!(html.contains("Streaming Plot"));
    }

    #[test]
    fn stream_config_builder() {
        let config = StreamConfig::new()
            .with_buffer_size(500)
            .with_size(1024, 768)
            .with_line_color(Color::RED)
            .with_line_width(3.0)
            .with_auto_scale(false)
            .with_scroll(false)
            .with_window_size(50);

        assert_eq!(config.buffer_size, 500);
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert!(!config.auto_scale);
        assert!(!config.scroll);
        assert_eq!(config.window_size, 50);
    }
}
