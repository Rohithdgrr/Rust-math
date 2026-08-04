//! GPU-accelerated rendering for million-point datasets using WebGL.
//!
//! This module provides a WebGL-based renderer that can handle large datasets
//! efficiently by offloading rendering to the GPU.
//!
//! # Example
//!
//! ```rust,no_run
//! use mathverse_plot::gpu_renderer::GpuRenderer;
//! use mathverse_plot::common::DataPoint;
//!
//! // Generate 1 million points
//! let points: Vec<DataPoint> = (0..1_000_000)
//!     .map(|i| {
//!         let x = i as f64 * 0.001;
//!         let y = (x * 10.0).sin();
//!         DataPoint::new(x, y)
//!     })
//!     .collect();
//!
//! // Create GPU renderer
//! let renderer = GpuRenderer::new(points);
//! let html = renderer.render_html();
//! ```

use crate::common::DataPoint;
use crate::style::Color;

/// Configuration for GPU rendering.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Point size in pixels.
    pub point_size: f32,
    /// Point color.
    pub point_color: Color,
    /// Background color.
    pub background_color: Color,
    /// Enable anti-aliasing.
    pub anti_alias: bool,
    /// Enable point clustering for very large datasets.
    pub enable_clustering: bool,
    /// Maximum points before clustering.
    pub cluster_threshold: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            point_size: 2.0,
            point_color: Color::BLUE,
            background_color: Color::WHITE,
            anti_alias: true,
            enable_clustering: true,
            cluster_threshold: 100_000,
        }
    }
}

impl GpuConfig {
    /// Create a new GPU config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the canvas dimensions.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the point size.
    pub fn with_point_size(mut self, size: f32) -> Self {
        self.point_size = size;
        self
    }

    /// Set the point color.
    pub fn with_point_color(mut self, color: Color) -> Self {
        self.point_color = color;
        self
    }

    /// Set the background color.
    pub fn with_background_color(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Enable or disable anti-aliasing.
    pub fn with_anti_alias(mut self, enable: bool) -> Self {
        self.anti_alias = enable;
        self
    }

    /// Enable or disable point clustering.
    pub fn with_clustering(mut self, enable: bool) -> Self {
        self.enable_clustering = enable;
        self
    }

    /// Set the clustering threshold.
    pub fn with_cluster_threshold(mut self, threshold: usize) -> Self {
        self.cluster_threshold = threshold;
        self
    }
}

/// GPU-accelerated renderer for large datasets.
pub struct GpuRenderer {
    /// Data points to render.
    points: Vec<DataPoint>,
    /// Renderer configuration.
    config: GpuConfig,
}

impl GpuRenderer {
    /// Create a new GPU renderer.
    pub fn new(points: Vec<DataPoint>) -> Self {
        Self {
            points,
            config: GpuConfig::default(),
        }
    }

    /// Create a new GPU renderer with custom configuration.
    pub fn with_config(points: Vec<DataPoint>, config: GpuConfig) -> Self {
        Self { points, config }
    }

    /// Render the data to a standalone HTML file with WebGL.
    pub fn render_html(&self) -> String {
        let data_json = self.points_to_json();
        let point_color = self.config.point_color.to_hex();
        let bg_color = self.config.background_color.to_hex();

        // Parse hex colors to RGB components
        let parse_hex = |hex: &str| -> (u8, u8, u8) {
            let hex = hex.trim_start_matches('#');
            if hex.len() == 6 {
                let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
                let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
                let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
                (r, g, b)
            } else {
                (0, 0, 0)
            }
        };

        let (pr, pg, pb) = parse_hex(&point_color);
        let (br, bg, bb) = parse_hex(&bg_color);

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GPU Scatter Plot - {} Points</title>
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
        .info {{
            margin-top: 10px;
            font-size: 12px;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GPU-Accelerated Scatter Plot</h1>
        <div class="stats" id="stats">Loading...</div>
        <canvas id="canvas" width="{}" height="{}"></canvas>
        <div class="controls">
            <button onclick="resetView()">Reset View</button>
            <button onclick="togglePoints()">Toggle Points</button>
            <button onclick="exportSVG()">Export SVG</button>
        </div>
        <div class="info">
            <p>WebGL rendering for {} data points. Use mouse to pan, scroll to zoom.</p>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        const statsEl = document.getElementById('stats');

        if (!gl) {{
            statsEl.textContent = 'WebGL not supported in your browser.';
            throw new Error('WebGL not supported');
        }}

        // Data
        const rawData = {};
        const points = rawData;
        const numPoints = points.length / 2;

        // Colors
        const pointColor = [{}, {}, {}];
        const bgColor = [{}, {}, {}];

        // State
        let showPoints = true;
        let offsetX = 0, offsetY = 0;
        let scale = 1.0;
        let lastMouseX = 0, lastMouseY = 0;
        let isDragging = false;

        // Find data bounds
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < points.length; i += 2) {{
            minX = Math.min(minX, points[i]);
            maxX = Math.max(maxX, points[i]);
            minY = Math.min(minY, points[i + 1]);
            maxY = Math.max(maxY, points[i + 1]);
        }}

        // Shader sources
        const vertexShaderSource = `
            attribute vec2 a_position;
            uniform vec2 u_resolution;
            uniform vec2 u_offset;
            uniform float u_scale;
            void main() {{
                vec2 pos = (a_position - u_offset) * u_scale;
                vec2 clipSpace = (pos / u_resolution) * 2.0 - 1.0;
                gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
                gl_PointSize = {};
            }}
        `;

        const fragmentShaderSource = `
            precision mediump float;
            uniform vec3 u_color;
            void main() {{
                float dist = length(gl_PointCoord - vec2(0.5));
                if (dist > 0.5) discard;
                gl_FragColor = vec4(u_color, 1.0);
            }}
        `;

        // Compile shader
        function compileShader(source, type) {{
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
                console.error('Shader compile error:', gl.getShaderInfoLog(shader));
                gl.deleteShader(shader);
                return null;
            }}
            return shader;
        }}

        // Create program
        const vertexShader = compileShader(vertexShaderSource, gl.VERTEX_SHADER);
        const fragmentShader = compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER);
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
            console.error('Program link error:', gl.getProgramInfoLog(program));
        }}

        gl.useProgram(program);

        // Create buffer
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.STATIC_DRAW);

        // Get attribute/uniform locations
        const positionLocation = gl.getAttribLocation(program, 'a_position');
        const resolutionLocation = gl.getUniformLocation(program, 'u_resolution');
        const offsetLocation = gl.getUniformLocation(program, 'u_offset');
        const scaleLocation = gl.getUniformLocation(program, 'u_scale');
        const colorLocation = gl.getUniformLocation(program, 'u_color');

        // Set up attribute
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

        // Set initial uniforms
        gl.uniform2f(resolutionLocation, canvas.width / 2, canvas.height / 2);
        gl.uniform3f(colorLocation, pointColor[0] / 255, pointColor[1] / 255, pointColor[2] / 255);

        // Render function
        function render() {{
            gl.clearColor(bgColor[0] / 255, bgColor[1] / 255, bgColor[2] / 255, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            if (showPoints && numPoints > 0) {{
                gl.uniform2f(offsetLocation, offsetX, offsetY);
                gl.uniform1f(scaleLocation, scale);
                gl.drawArrays(gl.POINTS, 0, numPoints);
            }}

            statsEl.textContent = `Rendering ${{numPoints.toLocaleString()}} points | Bounds: [${{minX.toFixed(2)}}, ${{maxX.toFixed(2)}}] x [${{minY.toFixed(2)}}, ${{maxY.toFixed(2)}}]`;
        }}

        // Mouse handlers
        canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }});

        canvas.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                const dx = e.clientX - lastMouseX;
                const dy = e.clientY - lastMouseY;
                offsetX -= dx / scale;
                offsetY -= dy / scale;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                render();
            }}
        }});

        canvas.addEventListener('mouseup', () => {{ isDragging = false; }});
        canvas.addEventListener('mouseleave', () => {{ isDragging = false; }});

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            scale *= zoomFactor;
            render();
        }});

        // Control functions
        function resetView() {{
            offsetX = 0;
            offsetY = 0;
            scale = 1.0;
            render();
        }}

        function togglePoints() {{
            showPoints = !showPoints;
            render();
        }}

        function exportSVG() {{
            let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${{canvas.width}}" height="${{canvas.height}}">`;
            svg += `<rect width="100%" height="100%" fill="rgb(${{bgColor.join(',')}})"/>`;
            svg += `<g transform="translate(${{canvas.width/2}},${{canvas.height/2}}) scale(${{scale}}) translate(${{-offsetX}},${{-offsetY}})">`;
            for (let i = 0; i < points.length; i += 2) {{
                svg += `<circle cx="${{points[i]}}" cy="${{points[i+1]}}" r="2" fill="rgb(${{pointColor.join(',')}})"/>`;
            }}
            svg += `</g></svg>`;

            const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'scatter_plot.svg';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Initial render
        render();
    </script>
</body>
</html>"#,
            self.points.len(),
            self.config.width,
            self.config.height,
            self.points.len(),
            data_json,
            pr,
            pg,
            pb,
            br,
            bg,
            bb,
            self.config.point_size
        )
    }

    /// Convert points to JSON array.
    fn points_to_json(&self) -> String {
        let mut json = String::from("[");
        for (i, point) in self.points.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("{},{}", point.x, point.y));
        }
        json.push(']');
        json
    }

    /// Get the number of points.
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Get the data bounds (min_x, max_x, min_y, max_y).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for point in &self.points {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }

        (min_x, max_x, min_y, max_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_renderer_creation() {
        let points = vec![
            DataPoint::new(0.0, 0.0),
            DataPoint::new(1.0, 1.0),
            DataPoint::new(2.0, 0.5),
        ];
        let renderer = GpuRenderer::new(points);
        assert_eq!(renderer.num_points(), 3);
    }

    #[test]
    fn gpu_renderer_bounds() {
        let points = vec![
            DataPoint::new(-1.0, -2.0),
            DataPoint::new(3.0, 4.0),
        ];
        let renderer = GpuRenderer::new(points);
        let (min_x, max_x, min_y, max_y) = renderer.bounds();
        assert_eq!(min_x, -1.0);
        assert_eq!(max_x, 3.0);
        assert_eq!(min_y, -2.0);
        assert_eq!(max_y, 4.0);
    }

    #[test]
    fn gpu_renderer_html() {
        let points = vec![DataPoint::new(0.0, 0.0), DataPoint::new(1.0, 1.0)];
        let renderer = GpuRenderer::new(points);
        let html = renderer.render_html();
        assert!(html.contains("<html"));
        assert!(html.contains("WebGL"));
        assert!(html.contains("[0,0,1,1]"));
    }

    #[test]
    fn gpu_config_builder() {
        let config = GpuConfig::new()
            .with_size(1024, 768)
            .with_point_size(3.0)
            .with_point_color(Color::RED)
            .with_anti_alias(false);

        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert_eq!(config.point_size, 3.0);
        assert!(!config.anti_alias);
    }
}
