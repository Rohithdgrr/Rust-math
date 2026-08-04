//! Interactive HTML/JS backend with tooltips, zoom, and pan.

use crate::backend::PlotData;
use crate::common::DataSeries;
use crate::error::PlotResult;
use crate::style::Color;

/// Configuration for the interactive HTML backend.
#[derive(Debug, Clone)]
pub struct InteractiveConfig {
    /// Enable mouse wheel zoom.
    pub zoom_enabled: bool,
    /// Enable click-and-drag pan.
    pub pan_enabled: bool,
    /// Enable tooltips on hover.
    pub tooltips_enabled: bool,
    /// Show crosshair on hover.
    pub crosshair: bool,
    /// Background color.
    pub background: Color,
    /// Grid color.
    pub grid_color: Color,
    /// Show export button.
    pub export_button: bool,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            zoom_enabled: true,
            pan_enabled: true,
            tooltips_enabled: true,
            crosshair: true,
            background: Color::WHITE,
            grid_color: Color::rgb(0xE0, 0xE0, 0xE0),
            export_button: true,
        }
    }
}

impl InteractiveConfig {
    /// Create a new interactive config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set zoom enabled.
    pub fn with_zoom(mut self, enabled: bool) -> Self {
        self.zoom_enabled = enabled;
        self
    }

    /// Set pan enabled.
    pub fn with_pan(mut self, enabled: bool) -> Self {
        self.pan_enabled = enabled;
        self
    }

    /// Set tooltips enabled.
    pub fn with_tooltips(mut self, enabled: bool) -> Self {
        self.tooltips_enabled = enabled;
        self
    }

    /// Set crosshair enabled.
    pub fn with_crosshair(mut self, enabled: bool) -> Self {
        self.crosshair = enabled;
        self
    }
}

/// Generate a self-contained interactive HTML page with JavaScript interactivity.
pub fn render_interactive_html(
    data: &PlotData,
    config: &InteractiveConfig,
) -> PlotResult<String> {
    let width = data.config.width;
    let height = data.config.height;
    let padding = data.config.padding;

    // Serialize series data as JSON for JavaScript
    let series_json = serialize_series_json(&data.series);

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ margin: 0; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }}
  .container {{ max-width: {width}px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 20px; }}
  h1 {{ text-align: center; color: #333; font-size: 20px; margin-bottom: 10px; }}
  .chart-container {{ position: relative; cursor: crosshair; }}
  svg {{ display: block; margin: 0 auto; }}
  .tooltip {{ position: absolute; background: rgba(0,0,0,0.8); color: white; padding: 8px 12px; border-radius: 4px; font-size: 12px; pointer-events: none; display: none; z-index: 100; white-space: nowrap; }}
  .crosshair {{ stroke: #999; stroke-width: 1; stroke-dasharray: 4,4; pointer-events: none; }}
  .controls {{ text-align: center; margin-top: 10px; }}
  .controls button {{ margin: 0 5px; padding: 6px 12px; border: 1px solid #ddd; background: white; border-radius: 4px; cursor: pointer; font-size: 12px; }}
  .controls button:hover {{ background: #f0f0f0; }}
  .legend {{ display: flex; justify-content: center; gap: 15px; margin-top: 10px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; }}
  .legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <div class="chart-container" id="chart">
    <svg id="svg" width="{width}" height="{height}"></svg>
    <div class="tooltip" id="tooltip"></div>
  </div>
  <div class="legend" id="legend"></div>
  <div class="controls">
    <button onclick="resetZoom()">Reset Zoom</button>
    <button onclick="exportSVG()">Export SVG</button>
  </div>
</div>

<script>
const SERIES = {series_json};
const WIDTH = {width};
const HEIGHT = {height};
const PADDING = {padding};
const PLOT_W = WIDTH - 2 * PADDING;
const PLOT_H = HEIGHT - 2 * PADDING;

// View state
let viewXMin = 0, viewXMax = 1, viewYMin = 0, viewYMax = 1;
let isDragging = false, dragStart = null;
let svgEl = document.getElementById('svg');
let tooltip = document.getElementById('tooltip');

// Compute data bounds
function getDataBounds() {{
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const s of SERIES) {{
    for (const p of s.points) {{
      if (p[0] < xMin) xMin = p[0];
      if (p[0] > xMax) xMax = p[0];
      if (p[1] < yMin) yMin = p[1];
      if (p[1] > yMax) yMax = p[1];
    }}
  }}
  const xPad = (xMax - xMin) * 0.05 || 0.5;
  const yPad = (yMax - yMin) * 0.05 || 0.5;
  return {{ xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad }};
}}

function init() {{
  const b = getDataBounds();
  viewXMin = b.xMin; viewXMax = b.xMax;
  viewYMin = b.yMin; viewYMax = b.yMax;
  render();
}}

function xPx(x) {{ return PADDING + (x - viewXMin) / (viewXMax - viewXMin) * PLOT_W; }}
function yPx(y) {{ return PADDING + PLOT_H - (y - viewYMin) / (viewYMax - viewYMin) * PLOT_H; }}
function dataX(px) {{ return viewXMin + (px - PADDING) / PLOT_W * (viewXMax - viewXMin); }}
function dataY(py) {{ return viewYMax - (py - PADDING) / PLOT_H * (viewYMax - viewYMin); }}

function render() {{
  let svg = '';

  // Background
  svg += `<rect width="${{WIDTH}}" height="${{HEIGHT}}" fill="{bg}"/>`;

  // Grid
  {grid_js}

  // Axes
  svg += `<line x1="${{PADDING}}" y1="${{PADDING+PLOT_H}}" x2="${{PADDING+PLOT_W}}" y2="${{PADDING+PLOT_H}}" stroke="black" stroke-width="2"/>`;
  svg += `<line x1="${{PADDING}}" y1="${{PADDING}}" x2="${{PADDING}}" y2="${{PADDING+PLOT_H}}" stroke="black" stroke-width="2"/>`;

  // Series
  for (const s of SERIES) {{
    const color = s.color || 'steelblue';
    // Line
    if (s.points.length > 1) {{
      let pts = s.points.map(p => `${{xPx(p[0])}},${{yPx(p[1])}}`).join(' ');
      svg += `<polyline points="${{pts}}" fill="none" stroke="${{color}}" stroke-width="2"/>`;
    }}
    // Markers
    for (const p of s.points) {{
      svg += `<circle cx="${{xPx(p[0])}}" cy="${{yPx(p[1])}}" r="4" fill="${{color}}"/>`;
    }}
  }}

  // Title
  svg += `<text x="${{WIDTH/2}}" y="30" text-anchor="middle" font-size="16" font-weight="bold">{title}</text>`;

  // Crosshairs (hidden by default)
  svg += `<line id="crosshair-x" class="crosshair" x1="0" y1="${{PADDING}}" x2="0" y2="${{PADDING+PLOT_H}}" style="display:none"/>`;
  svg += `<line id="crosshair-y" class="crosshair" x1="${{PADDING}}" y1="0" x2="${{PADDING+PLOT_W}}" y2="0" style="display:none"/>`;

  svgEl.innerHTML = svg;

  // Build legend
  let legendHtml = '';
  for (const s of SERIES) {{
    const c = s.color || 'steelblue';
    legendHtml += `<div class="legend-item"><div class="legend-color" style="background:${{c}}"></div>${{s.name}}</div>`;
  }}
  document.getElementById('legend').innerHTML = legendHtml;
}}

// Tooltip on hover
svgEl.addEventListener('mousemove', (e) => {{
  const rect = svgEl.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (mx < PADDING || mx > WIDTH - PADDING || my < PADDING || my > HEIGHT - PADDING) {{
    tooltip.style.display = 'none';
    return;
  }}

  const dx = dataX(mx);
  const dy = dataY(my);

  // Find nearest point
  let nearest = null, minDist = Infinity;
  for (const s of SERIES) {{
    for (const p of s.points) {{
      const dist = Math.sqrt((xPx(p[0]) - mx) ** 2 + (yPx(p[1]) - my) ** 2);
      if (dist < minDist) {{ minDist = dist; nearest = {{ series: s.name, x: p[0], y: p[1], color: s.color }}; }}
    }}
  }}

  if (nearest && minDist < 30) {{
    tooltip.innerHTML = `<strong>${{nearest.series}}</strong><br>x: ${{nearest.x.toFixed(4)}}<br>y: ${{nearest.y.toFixed(4)}}`;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
    tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
  }} else {{
    tooltip.style.display = 'none';
  }}

  // Crosshair
  document.getElementById('crosshair-x').setAttribute('x1', mx);
  document.getElementById('crosshair-x').setAttribute('x2', mx);
  document.getElementById('crosshair-x').style.display = '';
  document.getElementById('crosshair-y').setAttribute('y1', my);
  document.getElementById('crosshair-y').setAttribute('y2', my);
  document.getElementById('crosshair-y').style.display = '';
}});

svgEl.addEventListener('mouseleave', () => {{
  tooltip.style.display = 'none';
  document.getElementById('crosshair-x').style.display = 'none';
  document.getElementById('crosshair-y').style.display = 'none';
}});

// Zoom
svgEl.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const rect = svgEl.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const factor = e.deltaY > 0 ? 1.1 : 0.9;
  const dx = dataX(mx), dy = dataY(my);
  viewXMin = dx - (dx - viewXMin) * factor;
  viewXMax = dx + (viewXMax - dx) * factor;
  viewYMin = dy - (dy - viewYMin) * factor;
  viewYMax = dy + (viewYMax - dy) * factor;
  render();
}});

// Pan
svgEl.addEventListener('mousedown', (e) => {{
  isDragging = true;
  dragStart = {{ x: e.clientX, y: e.clientY, xMin: viewXMin, xMax: viewXMax, yMin: viewYMin, yMax: viewYMax }};
}});
window.addEventListener('mousemove', (e) => {{
  if (!isDragging) return;
  const dx = (e.clientX - dragStart.x) / PLOT_W * (dragStart.xMax - dragStart.xMin);
  const dy = (e.clientY - dragStart.y) / PLOT_H * (dragStart.yMax - dragStart.yMin);
  viewXMin = dragStart.xMin - dx;
  viewXMax = dragStart.xMax - dx;
  viewYMin = dragStart.yMin + dy;
  viewYMax = dragStart.yMax + dy;
  render();
}});
window.addEventListener('mouseup', () => {{ isDragging = false; }});

function resetZoom() {{
  const b = getDataBounds();
  viewXMin = b.xMin; viewXMax = b.xMax;
  viewYMin = b.yMin; viewYMax = b.yMax;
  render();
}}

function exportSVG() {{
  const blob = new Blob([svgEl.outerHTML], {{ type: 'image/svg+xml' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'chart.svg'; a.click();
  URL.revokeObjectURL(url);
}}

init();
</script>
</body>
</html>"#,
        title = data.config.title,
        width = width,
        height = height,
        padding = padding,
        bg = config.background.to_hex(),
        series_json = series_json,
        grid_js = generate_grid_js(),
    );

    Ok(html)
}

fn serialize_series_json(series: &[DataSeries]) -> String {
    let mut parts = Vec::new();
    for s in series {
        let points: Vec<String> = s
            .points
            .iter()
            .map(|p| format!("[{},{}]", p.x, p.y))
            .collect();
        let color = s.style.line_color.to_hex();
        parts.push(format!(
            r#"{{"name":"{}","color":"{}","points":[{}]}}"#,
            s.name.replace('"', "\\\""),
            color,
            points.join(",")
        ));
    }
    format!("[{}]", parts.join(","))
}

fn generate_grid_js() -> String {
    "// Simple grid ticks\n  for (let v = Math.ceil(viewXMin); v <= Math.floor(viewXMax); v++) {\n    svg += `<line x1=\"${xPx(v)}\" y1=\"${PADDING}\" x2=\"${xPx(v)}\" y2=\"${PADDING+PLOT_H}\" stroke=\"#eee\" stroke-width=\"1\"/>`;\n    svg += `<text x=\"${xPx(v)}\" y=\"${PADDING+PLOT_H+15}\" text-anchor=\"middle\" font-size=\"10\">${v}</text>`;\n  }\n  for (let v = Math.ceil(viewYMin); v <= Math.floor(viewYMax); v++) {\n    svg += `<line x1=\"${PADDING}\" y1=\"${yPx(v)}\" x2=\"${PADDING+PLOT_W}\" y2=\"${yPx(v)}\" stroke=\"#eee\" stroke-width=\"1\"/>`;\n    svg += `<text x=\"${PADDING-8}\" y=\"${yPx(v)+4}\" text-anchor=\"end\" font-size=\"10\">${v}</text>`;\n  }\n    ".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{DataPoint, DataSeries, PlotConfig};
    use crate::style::PlotStyle;

    #[test]
    fn interactive_html_renders() {
        let config = PlotConfig::new().with_title("Test".to_string());
        let points = vec![DataPoint::new(0.0, 0.0), DataPoint::new(1.0, 1.0)];
        let series = DataSeries::new("line".to_string(), points);
        let data = PlotData {
            config,
            series: vec![series],
            bars: vec![],
            boxes: vec![],
            error_bars: vec![],
            heatmaps: vec![],
        };
        let html = render_interactive_html(&data, &InteractiveConfig::new()).unwrap();
        assert!(html.contains("<svg"));
        assert!(html.contains("mousemove"));
    }
}
