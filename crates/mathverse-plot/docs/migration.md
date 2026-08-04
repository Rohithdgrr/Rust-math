# Migration Guide

Guides users of `mathverse-plot` 0.1.x to the current API. The core `DataPoint`
/ `DataSeries` / `PlotConfig` builder flow is unchanged; this document covers
the additions and the small renames that landed since the initial release.

## Summary of changes

| Area | 0.1.0 | Current |
|---|---|---|
| Backends | `SvgPlot`, `HtmlPlot`, `TerminalPlot` | + `Backend` trait, `PngBackend`, `PdfBackend`, `InteractivePlot` |
| Chart types | line/scatter/bar | + `Histogram`, `BoxStats`, `ErrorBar`, `HeatmapData`, `Candlestick`, `PolarData` |
| Scales | linear only | `Scale::{Linear, Log, SymLog, Sqrt}` |
| Colors | `Color` with `to_hex()` | + `Color::to_rgb()`, colormaps (`viridis`, `plasma`), `Normalization` |
| Histograms | manual binning | `Histogram::bin` with `BinningMethod` from `mathverse-statistics` |
| Figures | single axes | `Figure` (rows × cols) with aggregated legend |

## Migration: rendering a plot

**0.1.0** — renderers took the plot directly:

```rust
let mut plot = SvgPlot::new(config);
plot.add_series(series);
plot.generate(); // String
```

**Current** — rendering goes through the `Backend` trait against a `PlotData`
snapshot, so every backend shares one scene description:

```rust
let mut plot = SvgPlot::new(config);
plot.add_series(series);
let snapshot = plot.snapshot(); // PlotData

// SVG
let svg: String = plot.generate();

// PNG (feature "png")
use mathverse_plot::Backend;
let uri = PngBackend::new(800, 600).generate(&snapshot)?;

// Terminal
TerminalPlot::new(config).with_dimensions(80, 24).generate(); // legacy path still works
```

No action is required if you only ever called `plot.generate()` on an
`SvgPlot`; that path is preserved.

## Migration: histograms

**0.1.0** — you picked bin width yourself.

**Current** — binning rules live in `mathverse-statistics`; ask for the rule by
name instead of guessing:

```rust
use mathverse_plot::{Histogram, BinningMethod};

let h = Histogram::new(samples).with_binning(BinningMethod::Scott).bin()?;
```

`BinningMethod::{Sturges, Scott, FreedmanDiaconis, Sqrt, Auto, Kde}` are
available; `Auto` selects on data size.

## Migration: axes and scales

Ticks are now generated with "nice" 1/2/5 × 10^n steps in a kernel space, and
log/symlog/sqrt scales are available on the config:

```rust
let config = PlotConfig::new()
    .with_x_scale(Scale::Log)
    .with_tick_count(8);
```

`Scale::Log` and `Scale::Sqrt` fall back to linear on non-positive data rather
than panicking; degenerate input is always a typed `PlotError` or an empty
plot, never a panic.

## Migration: colors

If you matched on `Color` variants directly, prefer the accessors:

```rust
let color = Color::Named("forestgreen");
let (r, g, b) = color.to_rgb(); // (34, 139, 34)
let hex = color.to_hex();       // "forestgreen"
```

`to_rgb()` resolves the CSS named palette for raster/PDF/interactive backends;
`to_hex()` keeps the previous behaviour.

## Migration: figures

New `Figure` supports subplots with shared labels and a deduplicated legend:

```rust
let mut figure = Figure::new(1, 2)
    .with_shared_x_label("x")
    .with_shared_y_label("y");

let axes = Axes::new(PlotConfig::new().with_title("A"));
figure.set_axes(0, 0, axes)?;
```

Rendering is delegated to a `Backend`; combine with `Backend`-based rendering
for multi-panel output.

## Feature flags

Enable only what you need:

```toml
[dependencies]
mathverse-plot = { version = "0.1", features = ["png", "pdf", "interactive"] }
```

| Feature | Provides | Pulls in |
|---|---|---|
| `png` | `PngBackend` (tiny-skia rasteriser) | `tiny-skia` |
| `pdf` | `PdfBackend` (printpdf) | `printpdf` |
| `interactive` | `InteractivePlot`, `run()` (egui/eframe) | `eframe` |
| (none) | `SvgPlot`, `HtmlPlot`, `TerminalPlot`, all chart types | — |

`mathverse-prelude` re-exports `mathverse_plot` behind its `plot` feature, so
`use mathverse_prelude::*;` brings in the whole crate.

## Behaviour changes

- **Empty data**: `Histogram`, `BoxStats::compute`, `ErrorBar::ci`, and
  `HeatmapData::new` return `Err(PlotError::InvalidData(..))` instead of
  panicking or emitting a blank SVG.
- **Non-finite data**: points containing `NaN`/infinity are skipped during
  range computation; `BoxStats` and `ErrorBar` reject them with a typed error.
- **Empty candlestick series**: renders an empty SVG rather than a blank
  canvas.

## Still not provided (planned)

- WASM canvas backend (feature `canvas`)
- Benchmark suite vs plotters/matplotlib (see `benches/`)
- 50+ worked examples (currently 10 under `examples/`)
