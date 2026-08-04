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
  common.rs       DataPoint, DataSeries, PlotConfig
  style.rs        Color, LineStyle, MarkerStyle, ChartStyle
  svg.rs          SvgPlot      -- SVG vector output
  html.rs         HtmlPlot     -- embeds SvgPlot into a page
  terminal.rs     TerminalPlot -- ASCII output
```

Planned modules (next phases):

```
  axes.rs         Axes: data-space <-> pixel-space transform + tick selection
  scales.rs       Scale trait: Linear, Log, SymLog, Sqrt
  charts/         one file per chart type (line, scatter, histogram, ...)
  color.rs        ColorSpace conversions + perceptual colormaps
  figure.rs       Figure: multi-axes layout + title/legend aggregation
  backend.rs      Backend trait (extracted when a 2nd real backend lands)
```

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
| Tangent lines, area fills, meshes | `mathverse-calculus` | per-chart integrals |
| Smoothing / interpolation / roots | `mathverse-numerical` | hand-rolled splines |
| FFT / signal plots | `mathverse-transforms`, `mathverse-signal` | custom DFT |
| Scientific overlays (gamma, erf, bessel) | `mathverse-special` | approximations |
| Heatmap data / image display | `mathverse-image` | raw pixel loops |
| Complex-plane / Argand plots | `mathverse-complex` | manual re/im juggling |
| Candlesticks / time-series | `mathverse-finance` | duplicated indicators |
| Network / graph layout | `mathverse-graph` | manual node edges |
| Typed axis ranges | `mathverse-units` | bare f64 labels |
| LaTeX-ish labels | `mathverse-symbolic`, `mathverse-algebra` | hand parsers |

Rule of thumb: if a number is computed, it comes from mathverse. Rendering glue
(e.g. SVG element string assembly) is ours.

## Backend Abstraction

Backends share one `Backend` trait contract (size, draw_line / draw_rect /
draw_circle / draw_path / draw_text / draw_image, save). Until a second real
backend exists (raster/PDF), the trait is an internal module, not public API:
premature abstraction is avoided and `SvgPlot` remains the single concrete
renderer. HTML wraps SVG. Terminal renders to a `Vec<char>` grid then prints.

## Rendering Pipeline

```
data -> [chart type] -> mathverse computation (stats, calculus, ...)
     -> Axes (scale + nice ticks) -> pixel coordinates
     -> render primitives (lines, rects, markers, text, fill)
     -> Backend (Svg / Html / Terminal)
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
    Math(mathverse_core::MathError),
    InvalidRange(String),   // zero-width axis, empty data where shape required
    Backend(String),
    Io(std::io::Error),
}
```

Rules: empty data renders an empty plot (never panics); empty/constant data
guard ranges; invalid data yields a typed error.

## Performance Strategy

- Zero-copy data views for large series (`&[f64]`, no per-point `DataPoint`
  allocation in hot loops).
- Reusable pixel buffers on the plot struct to avoid per-frame `Vec` churn.
- Binned charts optionally use `rayon` behind feature `parallel`, delegating
  the histogram kernel to `mathverse-statistics`.
- Rendering never blocks on mathverse compute; charts are precomputed, then
  drawn.