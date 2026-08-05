# mathverse-plot Architecture

Plotting like matplotlib, built purely on the MathVerse ecosystem.

## Repository Layout

Single crate `crates/mathverse-plot` inside the MathVerse workspace. Backends
stay behind Cargo features until a heavy dependency (raster, PDF, WASM) earns
its own crate. This crate is intentionally one crate: the workspace convention
is one flat `mathverse-*` crate per domain, and SVG/HTML/terminal already coexist
here cleanly.

```
crates/mathverse-plot/src/
  lib.rs          re-exports + crate docs
  common.rs       DataPoint, DataSeries, PlotConfig, xml_escape, downsample_lttb
  style.rs        Color, LineStyle, MarkerStyle, PlotStyle
  svg.rs          SvgPlot      -- SVG vector output
  html.rs         HtmlPlot     -- embeds SvgPlot into a page
  terminal.rs     TerminalPlot -- ASCII output
  backend.rs      Backend trait + PlotData snapshot
  axes.rs         Range, Scale, axis_kernel, nice tick selection
  color.rs        Colormaps (Viridis, Plasma, Inferno, Magma, Cividis)
  figure.rs       Figure + Axes multi-axes layout
  error.rs        PlotError enum (InvalidData, Math, Io)
  theme.rs        9 themes with 6+ color palettes
  legend.rs       Flexible legend positioning
  animation.rs    Multi-frame SVG animation
  export.rs       Unified export API (SVG, PNG, PDF, HTML)
```

Additional modules behind Cargo features:

| Feature | Module | Dependency |
|---------|--------|-----------|
| `png` | `png_backend.rs` | `tiny-skia`, `resvg`, `usvg` |
| `pdf` | `pdf_backend.rs` | `printpdf` |
| `interactive` | `interactive.rs` | `eframe` |
| `canvas` | `canvas.rs` | `wasm-bindgen`, `web-sys` |

Specialized domain modules (gated on workspace sibling crates):

| Module | Dependency | Purpose |
|--------|-----------|---------|
| `candlestick.rs` | `mathverse-finance` | OHLC candlestick charts |
| `ml_plots.rs` | `mathverse-machine-learning` | Confusion matrix, ROC curve |
| `graph_layout.rs` | `mathverse-graph` | Network/graph visualization |
| `spectrogram.rs` | `mathverse-transforms`, `mathverse-signal` | FFT-based spectrogram |
| `surface.rs` | `mathverse-graphics` | 3D wireframe surface |
| `complex_plane.rs` | `mathverse-complex` | Argand diagram, domain coloring |
| `pdf_overlay.rs` | `mathverse-probability` | Theoretical PDF overlays |

## Dependency Rules (mandatory)

Every mathematical computation ships from a `mathverse-*` crate. Standard
library and other crates provide only IO, temporals, and rendering glue.

| Concern | Use | Never hand-roll |
|---|---|---|
| Numeric traits, precision, constants | `mathverse-core` | `eq` on bare f64 |
| Points / polygons / bounding boxes | `mathverse-geometry`, `mathverse-vector` | custom `Vec<f64>` pairs |
| Transforms, projection, fitting | `mathverse-matrix`, `mathverse-linear-algebra`, `mathverse-graphics` | hand math |
| Descriptive stats, binning, CI, KDE | `mathverse-statistics` | ad-hoc loops |
| Theoretical PDF/CDF overlays | `mathverse-probability` | hardcoded curves |
| Smoothing / interpolation / roots | `mathverse-numerical` | hand-rolled splines |
| FFT / signal plots | `mathverse-transforms`, `mathverse-signal` | custom DFT |
| Heatmap data / image display | `mathverse-image` | raw pixel loops |
| Complex-plane / Argand plots | `mathverse-complex` | manual re/im juggling |
| Candlesticks / time-series | `mathverse-finance` | duplicated indicators |
| Network / graph layout | `mathverse-graph` | manual node edges |

Rule of thumb: if a number is computed, it comes from mathverse. Rendering glue
(e.g. SVG element string assembly) is ours.

## Backend Abstraction

The `Backend` trait in `backend.rs` defines a single method:

```rust
pub trait Backend {
    fn generate(&self, data: &PlotData) -> PlotResult<String>;
}
```

`SvgPlot` implements `Backend` directly. `PngBackend` and `PdfBackend`
implement `Backend` behind feature flags. `TerminalPlot` implements `Backend`
for ASCII output. The `PlotData` snapshot decouples data preparation from
rendering so any backend can consume the same pre-computed data.

## Rendering Pipeline

```
data -> [chart type] -> mathverse computation (stats, calculus, ...)
     -> Axes (scale + nice ticks) -> pixel coordinates
     -> render primitives (lines, rects, markers, text, fill)
     -> Backend (Svg / Html / Terminal / Png / Pdf)
```

Two transforms matter:

1. **Scale**: value -> "ticks/kernel" space. Linear identity, Log = ln,
   etc. Always paired with exact inverse for axis labels.
2. **Affine**: kernel value -> pixel. Folded into 2x3 matrix to avoid
   recomputing range bounds per point.

## Error Handling

One error type (`PlotError`) in `mathverse-plot` (thiserror), with `From`
conversions so mathverse errors surface unchanged:

```rust
pub enum PlotError {
    InvalidData(String),
    Math(#[from] mathverse_core::MathError),
    Io(#[from] std::io::Error),
}
```

Rules: empty data renders an empty plot (never panics); empty/constant data
guard ranges; invalid data yields a typed error.

## Performance Strategy

- Zero-copy data views for large series (`&[f64]`, no per-point `DataPoint`
  allocation in hot loops).
- Reusable pixel buffers on the plot struct to avoid per-frame `Vec` churn.
- LTTB downsampling (`downsample_lttb`) for series exceeding target point count.
- Rendering never blocks on mathverse compute; charts are precomputed, then
  drawn.