# mathverse-plot: How It Works

## Core Data Model

Everything is data + config + backend:

```
DataPoint { x: f64, y: f64 }                 -- single point
DataSeries { name, points, style }           -- one styled series
PlotConfig { title, labels, size, toggles }  -- plot-level options
SvgPlot / HtmlPlot / TerminalPlot            -- renderers
```

A plot is created by constructing a `PlotConfig`, adding `DataSeries`, and
calling `generate()` on the chosen backend. Histograms are built from raw
samples with `Histogram::bin` (binning rules come from `mathverse-statistics`).

## Coordinate Transform

Each backend maps data space to output space in two stages:

```
data  --(scale)-->  kernel space  --(translate+scale)-->  pixels
```

- Data range: min/max over all series via `plot_bounds`, padded by 5%
  (`Range::pad`). Padding happens in kernel space, where ticks are uniform.
- Kernel space is `Scale::transform` applied to the data range; ticks are
  generated in kernel space with `Scale::ticks` and mapped back to data-space
  labels with the inverse transform (`axis_kernel` in `axes.rs`).
- `axis_kernel` falls back to a linear identity mapping when a non-linear
  scale is degenerate on the data (e.g. `Log` on non-positive values), so
  rendering never emits NaN.
- X pixel = padding + (x - x_min) / (x_max - x_min) * plot_width
- Y pixel = padding + plot_height - (y - y_min) / (y_max - y_min) * plot_height
  (Y inverted: SVG/terminal origin is top-left / top-down).

The same transform serves the grid (ticks), axes, tick labels, and series.

## Rendering per Backend

### SVG (`svg.rs`)

String-assembled vector output, layer order:

1. Background rect (series style or white)
2. Grid (lines at nice ticks, `grid_color`, 0.5 opacity)
3. Axes (2px black lines)
4. Ticks: marks + labels at nice-tick positions (kernel-space, data labels)
5. Data series: `<polyline>` for the line, marker elements per point
6. Box plots: whiskers + caps, quartile box, median line, outlier circles
7. Error bars: vertical whisker + caps + center marker
8. Title, X label, rotated Y label
9. Legend box (top-right, sized by series + box entries)

Histograms/boxplots/KDE/heatmaps are built from `mathverse-statistics`
(binning rules, quartiles, mean_ci, kernel density). Colormaps (Viridis,
Plasma) and value normalization live in `color.rs`; both are tested against
matplotlib reference stops.

### HTML (`html.rs`)

Minimal wrapper: `<!DOCTYPE html>` + inline CSS + embedded `SvgPlot::generate()`
output. No JS; plot is static SVG.

### Terminal (`terminal.rs`)

1. Compute ranges (same logic as SVG, guarded against zero-width ranges)
2. Fill `Vec<Vec<char>>` grid (default 80x24)
3. Plot each point as `*` (Y inverted: `grid[height-1-y][x]`)
4. Draw `|` axis and `-` axis with `+` origin
5. Print title, Y label, grid, X label, legend

## Style System

`ChartStyle` (per series) and `PlotConfig` (per plot) are separate: series
style controls color/width/markers, config controls title/labels/grid/legend.
Colors render via `to_hex()` (`#rrggbb` / `#rrggbbaa` / named).

## Current Limitations (known, intentional)

- Grid and tick labels are generated from nice-tick computation; tick count
  is configurable via `PlotConfig.tick_count` (default 6).
- `Log`/`SymLog`/`Sqrt` scales are implemented; they fall back to linear on
  non-positive data rather than erroring.
- HTML embeds SVG textually; large plots produce large HTML.
- Mathverse integration current: `mathverse-statistics` supplies all binning
  rules + KDE; `mathverse-core` supplies `MathError` (`PlotError::Math`);
  `mathverse-probability` supplies PDF overlays.
- PNG backend is feature-gated (`png`); rasterises lines, bars, boxes, error
  bars, heatmaps, and scatter dots via `tiny-skia`. Does not render text or
  axis labels yet.
- PDF backend is feature-gated (`pdf`); renders lines, bars, box plots, error
  bars, and heatmaps via `printpdf`. Circles approximated as 32-segment
  polygons; text labels not yet rendered.
- Interactive backend is feature-gated (`interactive`); an egui/eframe window
  (`examples/interactive.rs`) renders the same `PlotData` scene with drag-to-pan
  and scroll-to-zoom. `Color::to_rgb` resolves named colors for the raster
  backends.
- Downsampling: `downsample_lttb` (Largest-Triangle-Three-Buckets) in
  `common.rs` preserves endpoints; available for large datasets.
- Style presets: `PlotStyle::seaborn()`, `seaborn_darkgrid()`,
  `fivethirtyeight()`, `minimal()`.

## Architecture

- **`backend.rs`**: `Backend` trait + `PlotData` snapshot (decouples renderers
  from `SvgPlot` internals). `SvgPlot`, `TerminalPlot`, `HtmlPlot`,
  `PngBackend`, and `PdfBackend` all implement `Backend`.
- **`figure.rs`**: `Figure` (rows×cols grid of `Axes`) with aggregated legend
  deduplication. `Axes` wraps a `PlotConfig` + series and provides range
  computation. `Figure` is layout-only; rendering delegates to a `Backend`.

## Performance Notes

- `DataPoint` is `Copy`; series are `Vec<DataPoint>`.
- SVG generation is O(n) per series with string building; 100k+ points can
  use `downsample_lttb` (Largest-Triangle-Three-Buckets) in `common.rs`.
- Terminal plot is O(points) with O(width*height) grid allocation.
