//! WebGL-based interactive 3D plotting.
//!
//! This module provides a WebGL-based renderer for 3D scatter plots,
//! surface plots, and other 3D visualizations with mouse interaction.
//!
//! # Example
//!
//! ```rust,no_run
//! use mathverse_plot::webgl_3d::{WebGL3D, Point3D};
//!
//! let points = vec![
//!     Point3D::new(1.0, 2.0, 3.0),
//!     Point3D::new(4.0, 5.0, 6.0),
//! ];
//!
//! let renderer = WebGL3D::new(points);
//! let html = renderer.render_html();
//! ```

use crate::style::Color;

/// A 3D point for WebGL rendering.
#[derive(Debug, Clone, Copy)]
pub struct Point3D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
    /// Optional color.
    pub color: Option<Color>,
    /// Optional size.
    pub size: Option<f32>,
}

impl Point3D {
    /// Create a new 3D point.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            color: None,
            size: None,
        }
    }

    /// Create a new 3D point with color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Create a new 3D point with size.
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = Some(size);
        self
    }
}

/// Configuration for WebGL 3D rendering.
#[derive(Debug, Clone)]
pub struct WebGL3DConfig {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Point size in pixels.
    pub point_size: f32,
    /// Background color.
    pub background_color: Color,
    /// Enable axis labels.
    pub show_axes: bool,
    /// Enable grid.
    pub show_grid: bool,
    /// Enable rotation.
    pub enable_rotation: bool,
    /// Enable zoom.
    pub enable_zoom: bool,
    /// Rotation speed.
    pub rotation_speed: f32,
}

impl Default for WebGL3DConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            point_size: 3.0,
            background_color: Color::WHITE,
            show_axes: true,
            show_grid: true,
            enable_rotation: true,
            enable_zoom: true,
            rotation_speed: 0.005,
        }
    }
}

impl WebGL3DConfig {
    /// Create a new WebGL 3D config with default values.
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

    /// Set the background color.
    pub fn with_background_color(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Enable or disable axes.
    pub fn with_axes(mut self, enable: bool) -> Self {
        self.show_axes = enable;
        self
    }

    /// Enable or disable grid.
    pub fn with_grid(mut self, enable: bool) -> Self {
        self.show_grid = enable;
        self
    }

    /// Enable or disable rotation.
    pub fn with_rotation(mut self, enable: bool) -> Self {
        self.enable_rotation = enable;
        self
    }

    /// Enable or disable zoom.
    pub fn with_zoom(mut self, enable: bool) -> Self {
        self.enable_zoom = enable;
        self
    }
}

/// WebGL 3D renderer for scatter plots.
/// A surface mesh sampled from a z-grid — the analogue of matplotlib's
/// `ax.plot_surface`. Produces triangles (two per grid cell) and a wireframe
/// segment list that callers can render or consume directly.
#[derive(Debug, Clone)]
pub struct SurfaceMesh {
    /// Row-major z values (`grid[r][c]`); rows map to y, columns to x.
    grid: Vec<Vec<f64>>,
    /// Data-space x extent `(xmin, xmax)`.
    x_range: (f64, f64),
    /// Data-space y extent `(ymin, ymax)`.
    y_range: (f64, f64),
}

impl SurfaceMesh {
    /// Build a surface from a z-grid over `[xmin, xmax] × [ymin, ymax]`.
    /// Row 0 of the grid is the `ymax` (far) edge.
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty/ragged grids or a grid too
    /// small to form a single quad (`< 2×2`).
    pub fn from_grid(
        grid: Vec<Vec<f64>>,
        x_range: (f64, f64),
        y_range: (f64, f64),
    ) -> crate::error::PlotResult<Self> {
        if grid.len() < 2 || grid[0].len() < 2 {
            return Err(crate::error::PlotError::InvalidData(
                "surface grid must be at least 2x2".into(),
            ));
        }
        let cols = grid[0].len();
        if grid.iter().any(|row| row.len() != cols) {
            return Err(crate::error::PlotError::InvalidData(
                "ragged surface grid".into(),
            ));
        }
        Ok(Self {
            grid,
            x_range,
            y_range,
        })
    }

    /// Grid dimensions `(rows, cols)`.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.grid.len(), self.grid[0].len())
    }

    /// A vertex at grid position `(r, c)`.
    #[must_use]
    pub fn vertex(&self, r: usize, c: usize) -> Point3D {
        let (rows, cols) = self.dims();
        let fx = c as f64 / (cols - 1) as f64;
        let fy = 1.0 - r as f64 / (rows - 1) as f64; // row 0 = ymax
        let x = self.x_range.0 + fx * (self.x_range.1 - self.x_range.0);
        let y = self.y_range.0 + fy * (self.y_range.1 - self.y_range.0);
        Point3D::new(x, y, self.grid[r][c])
    }

    /// All triangles (two per cell), in grid order.
    #[must_use]
    pub fn triangles(&self) -> Vec<[Point3D; 3]> {
        let (rows, cols) = self.dims();
        let mut out = Vec::with_capacity((rows - 1) * (cols - 1) * 2);
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                let a = self.vertex(r, c);
                let b = self.vertex(r, c + 1);
                let d = self.vertex(r + 1, c + 1);
                let e = self.vertex(r + 1, c);
                out.push([a, b, d]);
                out.push([a, d, e]);
            }
        }
        out
    }

    /// Wireframe segments along every grid row and column.
    #[must_use]
    pub fn wireframe_segments(&self) -> Vec<[Point3D; 2]> {
        let (rows, cols) = self.dims();
        let mut out = Vec::with_capacity((rows - 1) * cols + (cols - 1) * rows);
        for r in 0..rows {
            for c in 0..cols - 1 {
                out.push([self.vertex(r, c), self.vertex(r, c + 1)]);
            }
        }
        for c in 0..cols {
            for r in 0..rows - 1 {
                out.push([self.vertex(r, c), self.vertex(r + 1, c)]);
            }
        }
        out
    }

    /// Data bounds `(min_x, max_x, min_y, max_y, min_z, max_z)`.
    #[must_use]
    pub fn bounds(&self) -> (f64, f64, f64, f64, f64, f64) {
        let zmin = self
            .grid
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let zmax = self
            .grid
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        (
            self.x_range.0,
            self.x_range.1,
            self.y_range.0,
            self.y_range.1,
            zmin,
            zmax,
        )
    }
}

/// Render a surface mesh to a self-contained interactive HTML page. The
/// renderer projects the mesh with an orthographic camera (drag to rotate,
/// wheel to zoom), depth-sorts the triangles (painter's algorithm) and fills
/// them with a height-based viridis gradient. Plain canvas-2D, no WebGL.
#[must_use]
pub fn render_surface_html(mesh: &SurfaceMesh, config: &WebGL3DConfig) -> String {
    let triangles = mesh.triangles();
    let (min_x, max_x, min_y, max_y, min_z, max_z) = mesh.bounds();
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;
    let cz = (min_z + max_z) / 2.0;
    let rng = (max_x - min_x)
        .max(max_y - min_y)
        .max(max_z - min_z)
        .max(1.0);
    let mut tris_json = String::from("[");
    for (i, t) in triangles.iter().enumerate() {
        if i > 0 {
            tris_json.push(',');
        }
        tris_json.push_str(&format!(
            "[[{},{},{}],[{},{},{}],[{},{},{}]]",
            t[0].x, t[0].y, t[0].z, t[1].x, t[1].y, t[1].z, t[2].x, t[2].y, t[2].z
        ));
    }
    tris_json.push(']');

    format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3D Surface</title>
<style>
  body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #f5f5f5; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 20px; border-radius: 6px; }}
  canvas {{ border: 1px solid #ddd; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  .stats {{ color: #666; font-size: 13px; margin: 8px 0; }}
  button {{ padding: 6px 14px; margin-right: 8px; border: none; border-radius: 4px; background: #007bff; color: white; cursor: pointer; }}
</style>
</head>
<body>
<div class="container">
  <h1>Interactive 3D Surface</h1>
  <div class="stats">{} triangles &middot; drag to rotate, wheel to zoom</div>
  <canvas id="cv" width="{}" height="{}"></canvas>
  <div style="margin-top:10px"><button onclick="reset()">Reset View</button><button onclick="auto()">Auto-Rotate</button></div>
</div>
<script>
const TRIS = {};
const CX = {}, CY = {}, CZ = {}, RNG = {};
const W = {}, H = {};
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let rotX = -0.6, rotY = 0.6, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0, spinning = false;
function project(x, y, z) {{
  const dx = x - CX, dy = y - CY, dz = z - CZ;
  const s = Math.sin(rotY), c = Math.cos(rotY);
  const x1 = dx * c - dz * s, z1 = dx * s + dz * c;
  const s2 = Math.sin(rotX), c2 = Math.cos(rotX);
  const y1 = dy * c2 - z1 * s2, z2 = dy * s2 + z1 * c2;
  const scale = 0.45 * zoom * Math.min(W, H) / RNG;
  return [W/2 + x1 * scale, H/2 - y1 * scale, z2];
}}
function colorFor(z) {{
  const t = Math.min(1, Math.max(0, (z - CZ) / RNG + 0.5));
  const a = [68, 1, 84], b = [59, 82, 139], c = [33, 145, 140], d = [94, 201, 98], e = [253, 231, 37];
  let col;
  if (t < 0.25) col = lerp(a, b, t * 4);
  else if (t < 0.5) col = lerp(b, c, (t - 0.25) * 4);
  else if (t < 0.75) col = lerp(c, d, (t - 0.5) * 4);
  else col = lerp(d, e, (t - 0.75) * 4);
  return `rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;
}}
function lerp(a, b, t) {{ return [Math.round(a[0]+(b[0]-a[0])*t), Math.round(a[1]+(b[1]-a[1])*t), Math.round(a[2]+(b[2]-a[2])*t)]; }}
function draw() {{
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);
  const items = TRIS.map((t, i) => {{
    const p = t.map(v => project(v[0], v[1], v[2]));
    return {{ z: (p[0][2] + p[1][2] + p[2][2]) / 3, p, v: t }};
  }});
  items.sort((a, b) => b.z - a.z);
  for (const it of items) {{
    const z = (it.v[0][2] + it.v[1][2] + it.v[2][2]) / 3;
    ctx.beginPath();
    ctx.moveTo(it.p[0][0], it.p[0][1]);
    ctx.lineTo(it.p[1][0], it.p[1][1]);
    ctx.lineTo(it.p[2][0], it.p[2][1]);
    ctx.closePath();
    ctx.fillStyle = colorFor(z);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 0.6;
    ctx.stroke();
  }}
}}
function reset() {{ rotX = -0.6; rotY = 0.6; zoom = 1; draw(); }}
function auto() {{ spinning = !spinning; }}
cv.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener('mouseup', () => {{ dragging = false; }});
window.addEventListener('mousemove', e => {{
  if (dragging) {{ rotY += (e.clientX - lastX) * 0.01; rotX += (e.clientY - lastY) * 0.01; lastX = e.clientX; lastY = e.clientY; draw(); }}
}});
cv.addEventListener('wheel', e => {{ e.preventDefault(); zoom *= e.deltaY > 0 ? 0.92 : 1.08; zoom = Math.max(0.2, Math.min(6, zoom)); draw(); }});
setInterval(() => {{ if (spinning) {{ rotY += 0.01; draw(); }} }}, 30);
draw();
</script>
</body>
</html>"#,
        triangles.len(),
        config.width,
        config.height,
        tris_json,
        cx,
        cy,
        cz,
        rng,
        config.width,
        config.height
    )
}

pub struct WebGL3D {
    /// 3D points to render.
    points: Vec<Point3D>,
    /// Renderer configuration.
    config: WebGL3DConfig,
}

impl WebGL3D {
    /// Create a new WebGL 3D renderer.
    pub fn new(points: Vec<Point3D>) -> Self {
        Self {
            points,
            config: WebGL3DConfig::default(),
        }
    }

    /// Create a new WebGL 3D renderer with custom configuration.
    pub fn with_config(points: Vec<Point3D>, config: WebGL3DConfig) -> Self {
        Self { points, config }
    }

    /// Render the data to a standalone HTML file with WebGL.
    pub fn render_html(&self) -> String {
        let data_json = self.points_to_json();
        let _bg_color = self.config.background_color.to_hex();

        // Compute data bounds and center
        let (min_x, max_x, min_y, max_y, min_z, max_z) = self.bounds();
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;
        let center_z = (min_z + max_z) / 2.0;
        let range_x = (max_x - min_x).abs().max(1.0);
        let range_y = (max_y - min_y).abs().max(1.0);
        let range_z = (max_z - min_z).abs().max(1.0);
        let max_range = range_x.max(range_y).max(range_z);

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Scatter Plot - {} Points</title>
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
            cursor: grab;
        }}
        canvas:active {{
            cursor: grabbing;
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
        <h1>Interactive 3D Scatter Plot</h1>
        <div class="stats" id="stats">Loading...</div>
        <canvas id="canvas" width="{}" height="{}"></canvas>
        <div class="controls">
            <button onclick="resetView()">Reset View</button>
            <button onclick="toggleRotation()">Toggle Rotation</button>
            <button onclick="toggleAxes()">Toggle Axes</button>
        </div>
        <div class="info">
            <p>Click and drag to rotate. Scroll to zoom. Right-click and drag to pan.</p>
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
        const numPoints = points.length / 3;

        // State
        let rotationX = 0.5;
        let rotationY = 0.5;
        let rotationZ = 0;
        let offsetX = 0, offsetY = 0, offsetZ = 0;
        let scale = 1.0;
        let lastMouseX = 0, lastMouseY = 0;
        let isDragging = false;
        let isPanning = false;
        let autoRotate = true;
        let showAxes = true;

        // Find data bounds
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        for (let i = 0; i < points.length; i += 3) {{
            minX = Math.min(minX, points[i]);
            maxX = Math.max(maxX, points[i]);
            minY = Math.min(minY, points[i + 1]);
            maxY = Math.max(maxY, points[i + 1]);
            minZ = Math.min(minZ, points[i + 2]);
            maxZ = Math.max(maxZ, points[i + 2]);
        }}

        // Center and scale data
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;
        const rangeZ = maxZ - minZ || 1;
        const maxRange = Math.max(rangeX, rangeY, rangeZ);

        // Shader sources
        const vertexShaderSource = `
            attribute vec3 a_position;
            attribute vec3 a_color;
            uniform mat4 u_modelViewMatrix;
            uniform mat4 u_projectionMatrix;
            uniform float u_pointSize;
            varying vec3 v_color;
            void main() {{
                // Normalize and center data
                vec3 pos = (a_position - vec3({}, {}, {})) / {} * 2.0;
                vec4 mvPosition = u_modelViewMatrix * vec4(pos, 1.0);
                gl_Position = u_projectionMatrix * mvPosition;
                gl_PointSize = u_pointSize * (300.0 / -mvPosition.z);
                v_color = a_color;
            }}
        `;

        const fragmentShaderSource = `
            precision mediump float;
            varying vec3 v_color;
            void main() {{
                float dist = length(gl_PointCoord - vec2(0.5));
                if (dist > 0.5) discard;
                gl_FragColor = vec4(v_color, 1.0);
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

        // Create buffers
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        const positions = new Float32Array(points.length);
        for (let i = 0; i < points.length; i += 3) {{
            positions[i] = points[i];
            positions[i + 1] = points[i + 1];
            positions[i + 2] = points[i + 2];
        }}
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

        // Color buffer (all points same color for now)
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        const colors = new Float32Array(numPoints * 3);
        for (let i = 0; i < numPoints; i++) {{
            colors[i * 3] = 0.2;
            colors[i * 3 + 1] = 0.5;
            colors[i * 3 + 2] = 1.0;
        }}
        gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);

        // Get attribute/uniform locations
        const positionLocation = gl.getAttribLocation(program, 'a_position');
        const colorLocation = gl.getAttribLocation(program, 'a_color');
        const modelViewMatrixLocation = gl.getUniformLocation(program, 'u_modelViewMatrix');
        const projectionMatrixLocation = gl.getUniformLocation(program, 'u_projectionMatrix');
        const pointSizeLocation = gl.getUniformLocation(program, 'u_pointSize');

        // Set up attributes
        gl.enableVertexAttribArray(positionLocation);
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);

        gl.enableVertexAttribArray(colorLocation);
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);

        // Matrix utilities
        function perspectiveMatrix(fovy, aspect, near, far) {{
            const f = 1.0 / Math.tan(fovy / 2);
            const nf = 1 / (near - far);
            return [
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (far + near) * nf, -1,
                0, 0, (2 * far * near) * nf, 0
            ];
        }}

        function lookAtMatrix(eye, center, up) {{
            const z = normalize(subtract(eye, center));
            const x = normalize(cross(up, z));
            const y = cross(z, x);
            return [
                x[0], y[0], z[0], 0,
                x[1], y[1], z[1], 0,
                x[2], y[2], z[2], 0,
                -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
            ];
        }}

        function multiplyMatrix(a, b) {{
            const result = new Array(16).fill(0);
            for (let i = 0; i < 4; i++) {{
                for (let j = 0; j < 4; j++) {{
                    for (let k = 0; k < 4; k++) {{
                        result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
                    }}
                }}
            }}
            return result;
        }}

        function rotateXMatrix(angle) {{
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1];
        }}

        function rotateYMatrix(angle) {{
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1];
        }}

        function rotateZMatrix(angle) {{
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            return [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        }}

        function translateMatrix(x, y, z) {{
            return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
        }}

        function scaleMatrix(s) {{
            return [s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1];
        }}

        function normalize(v) {{
            const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 0];
        }}

        function subtract(a, b) {{
            return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        }}

        function cross(a, b) {{
            return [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            ];
        }}

        function dot(a, b) {{
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }}

        // Render function
        function render() {{
            gl.clearColor(0.95, 0.95, 0.95, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST);

            // Projection matrix
            const projection = perspectiveMatrix(
                Math.PI / 4,
                canvas.width / canvas.height,
                0.1,
                100.0
            );

            // Model-view matrix
            let modelView = translateMatrix(offsetX, offsetY, offsetZ - 3.0);
            modelView = multiplyMatrix(modelView, rotateXMatrix(rotationX));
            modelView = multiplyMatrix(modelView, rotateYMatrix(rotationY));
            modelView = multiplyMatrix(modelView, rotateZMatrix(rotationZ));
            modelView = multiplyMatrix(modelView, scaleMatrix(scale));

            // Set uniforms
            gl.uniformMatrix4fv(modelViewMatrixLocation, false, modelView);
            gl.uniformMatrix4fv(projectionMatrixLocation, false, projection);
            gl.uniform1f(pointSizeLocation, {} * scale);

            // Draw points
            gl.drawArrays(gl.POINTS, 0, numPoints);

            statsEl.textContent = `Rendering ${{numPoints.toLocaleString()}} points | Rotation: ${{(rotationX * 180 / Math.PI).toFixed(1)}}°, ${{(rotationY * 180 / Math.PI).toFixed(1)}}°`;
        }}

        // Animation loop
        function animate() {{
            if (autoRotate) {{
                rotationY += 0.01;
            }}
            render();
            requestAnimationFrame(animate);
        }}

        // Mouse handlers
        canvas.addEventListener('mousedown', (e) => {{
            if (e.button === 0) {{
                isDragging = true;
            }} else if (e.button === 2) {{
                isPanning = true;
            }}
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }});

        canvas.addEventListener('mousemove', (e) => {{
            const dx = e.clientX - lastMouseX;
            const dy = e.clientY - lastMouseY;

            if (isDragging) {{
                rotationY += dx * 0.01;
                rotationX += dy * 0.01;
            }} else if (isPanning) {{
                offsetX += dx * 0.005;
                offsetY -= dy * 0.005;
            }}

            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }});

        canvas.addEventListener('mouseup', () => {{
            isDragging = false;
            isPanning = false;
        }});

        canvas.addEventListener('mouseleave', () => {{
            isDragging = false;
            isPanning = false;
        }});

        canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            scale *= zoomFactor;
        }});

        // Control functions
        function resetView() {{
            rotationX = 0.5;
            rotationY = 0.5;
            rotationZ = 0;
            offsetX = 0;
            offsetY = 0;
            offsetZ = 0;
            scale = 1.0;
        }}

        function toggleRotation() {{
            autoRotate = !autoRotate;
        }}

        function toggleAxes() {{
            showAxes = !showAxes;
        }}

        // Start animation
        animate();
    </script>
</body>
</html>"#,
            self.points.len(),
            self.config.width,
            self.config.height,
            data_json,
            center_x,
            center_y,
            center_z,
            max_range,
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
            json.push_str(&format!("{},{},{}", point.x, point.y, point.z));
        }
        json.push(']');
        json
    }

    /// Get the number of points.
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Get the data bounds.
    pub fn bounds(&self) -> (f64, f64, f64, f64, f64, f64) {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for point in &self.points {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
            min_z = min_z.min(point.z);
            max_z = max_z.max(point.z);
        }

        (min_x, max_x, min_y, max_y, min_z, max_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_mesh_triangles_and_wireframe() {
        let grid = vec![vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0]];
        let mesh = SurfaceMesh::from_grid(grid, (0.0, 2.0), (0.0, 2.0)).unwrap();
        assert_eq!(mesh.dims(), (3, 3));
        // (rows-1)*(cols-1)*2 = 8 triangles.
        assert_eq!(mesh.triangles().len(), 8);
        // Wireframe: rows*(cols-1) + cols*(rows-1) = 3*2 + 3*2 = 12.
        assert_eq!(mesh.wireframe_segments().len(), 12);
        let (min_x, max_x, min_y, max_y, min_z, max_z) = mesh.bounds();
        assert_eq!((min_x, max_x), (0.0, 2.0));
        assert_eq!((min_y, max_y), (0.0, 2.0));
        assert_eq!((min_z, max_z), (0.0, 1.0));
    }

    #[test]
    fn surface_mesh_validates_input() {
        assert!(SurfaceMesh::from_grid(vec![vec![1.0]], (0.0, 1.0), (0.0, 1.0)).is_err());
        assert!(
            SurfaceMesh::from_grid(vec![vec![1.0, 2.0], vec![3.0]], (0.0, 1.0), (0.0, 1.0))
                .is_err()
        );
    }

    #[test]
    fn surface_mesh_vertex_mapping() {
        let grid = vec![vec![5.0, 5.0], vec![0.0, 0.0]];
        let mesh = SurfaceMesh::from_grid(grid, (0.0, 1.0), (0.0, 1.0)).unwrap();
        // Row 0 = ymax (1.0); col 0 = xmin (0.0).
        let v = mesh.vertex(0, 0);
        assert_eq!((v.x, v.y, v.z), (0.0, 1.0, 5.0));
        let v = mesh.vertex(1, 1);
        assert_eq!((v.x, v.y, v.z), (1.0, 0.0, 0.0));
    }

    #[test]
    fn surface_html_renders() {
        let grid = vec![vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0]];
        let mesh = SurfaceMesh::from_grid(grid, (0.0, 1.0), (0.0, 1.0)).unwrap();
        let html = render_surface_html(&mesh, &WebGL3DConfig::default());
        assert!(html.contains("<canvas"));
        assert!(html.contains("TRIS"));
        assert!(html.contains("8 triangles"));
    }

    #[test]
    fn webgl3d_creation() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 1.0),
        ];
        let renderer = WebGL3D::new(points);
        assert_eq!(renderer.num_points(), 2);
    }

    #[test]
    fn webgl3d_bounds() {
        let points = vec![
            Point3D::new(-1.0, -2.0, -3.0),
            Point3D::new(1.0, 2.0, 3.0),
        ];
        let renderer = WebGL3D::new(points);
        let (min_x, max_x, min_y, max_y, min_z, max_z) = renderer.bounds();
        assert_eq!(min_x, -1.0);
        assert_eq!(max_x, 1.0);
        assert_eq!(min_y, -2.0);
        assert_eq!(max_y, 2.0);
        assert_eq!(min_z, -3.0);
        assert_eq!(max_z, 3.0);
    }

    #[test]
    fn webgl3d_html() {
        let points = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)];
        let renderer = WebGL3D::new(points);
        let html = renderer.render_html();
        assert!(html.contains("<html"));
        assert!(html.contains("WebGL"));
        assert!(html.contains("[0,0,0,1,1,1]"));
    }

    #[test]
    fn webgl3d_config_builder() {
        let config = WebGL3DConfig::new()
            .with_size(1024, 768)
            .with_point_size(5.0)
            .with_background_color(Color::BLACK)
            .with_axes(false)
            .with_rotation(false);

        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert_eq!(config.point_size, 5.0);
        assert!(!config.show_axes);
        assert!(!config.enable_rotation);
    }

    #[test]
    fn point3d_builder() {
        let point = Point3D::new(1.0, 2.0, 3.0)
            .with_color(Color::RED)
            .with_size(5.0);

        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.color, Some(Color::RED));
        assert_eq!(point.size, Some(5.0));
    }
}
