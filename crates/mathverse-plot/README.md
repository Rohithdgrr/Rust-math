# mathverse-plot

[![Crates.io](https://img.shields.io/crates/v/mathverse-plot.svg)](https://crates.io/crates/mathverse-plot)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

Plotting with SVG, HTML, and terminal output backends for the MathVerse ecosystem.

## Features

- **SVG backend** — production-quality vector plots with grid, axes, legend
- **HTML backend** — self-contained HTML pages wrapping SVG plots
- **Terminal backend** — ASCII art plots for CLI output
- **Style system** — configurable colors, line styles, markers, fill
- **Multi-series** — overlay multiple data series on one plot

## Module Overview

| Module | Description |
|---|---|
| `common` | `DataPoint`, `DataSeries`, `PlotConfig` — shared data types |
| `style` | `Color`, `LineStyle`, `MarkerStyle`, `PlotStyle` — visual configuration |
| `svg` | `SvgPlot` — SVG vector output |
| `html` | `HtmlPlot` — self-contained HTML output |
| `terminal` | `TerminalPlot` — ASCII art terminal output |

## Installation

```toml
[dependencies]
mathverse-plot = { path = "../mathverse-plot" }
```

## Quick Start

```rust
use mathverse_prelude::*;

fn main() {
    let config = PlotConfig::new()
        .with_title("Sine Wave")
        .with_x_label("x")
        .with_y_label("sin(x)")
        .with_dimensions(800, 400);

    let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
    let points: Vec<DataPoint> = xs.iter().zip(ys.iter()).map(|(x, y)| DataPoint::new(*x, *y)).collect();
    let series = DataSeries::new("sin(x)", points);

    let mut svg_plot = SvgPlot::new(config.clone());
    svg_plot.add_series(series);
    let svg = svg_plot.generate();
    println!("SVG length: {} bytes", svg.len());

    let mut term_plot = TerminalPlot::new(config).with_dimensions(80, 24);
    term_plot.add_series(DataSeries::new("sin(x)", points));
    println!("{}", term_plot.generate());
}
```

Expected output (terminal, truncated):

```
    1.0 |           ****
        |        **     **
    0.0 |----**-----------**----
        |  **               **
   -1.0 | *                   *
        +------------------------
          0    1    2    3    4
```

## Per-Module Reference

### `common` — Data Types

| Type | Description |
|---|---|
| `DataPoint` | Single `(x, y)` point |
| `DataSeries` | Named collection of `DataPoint`s with style |
| `PlotConfig` | Title, axis labels, dimensions, grid/legend toggles |

### `style` — Visual Configuration

| Type | Description |
|---|---|
| `Color` | `Rgb(r,g,b)`, `Rgba(r,g,b,a)`, or `Named(&str)` |
| `LineStyle` | `Solid`, `Dashed`, `Dotted`, `DashDot` |
| `MarkerStyle` | `Circle`, `Square`, `Triangle`, `Cross`, `Plus`, `Diamond`, `None` |
| `PlotStyle` | Full style config with builder methods |

Built-in colors: `BLACK`, `WHITE`, `RED`, `GREEN`, `BLUE`, `YELLOW`, `CYAN`, `MAGENTA`, `GRAY`, `ORANGE`, `PURPLE`, `BROWN`.

### `svg` — SVG Backend

| Method | Description |
|---|---|
| `SvgPlot::new(config)` | Create with plot configuration |
| `.add_series(series)` | Add a data series |
| `.generate()` → `String` | Render to SVG markup |

### `html` — HTML Backend

| Method | Description |
|---|---|
| `HtmlPlot::new(config)` | Create with plot configuration |
| `.add_series(series)` | Add a data series |
| `.generate()` → `String` | Render to self-contained HTML |

### `terminal` — Terminal Backend

| Method | Description |
|---|---|
| `TerminalPlot::new(config)` | Create (default 80×24) |
| `.with_dimensions(w, h)` | Set terminal size |
| `.add_series(series)` | Add a data series |
| `.generate()` → `String` | Render to ASCII art |

## Future Scope

- Bar charts, pie charts, heatmaps
- Polar plots
- Interactive HTML with JavaScript zoom/pan
- LaTeX/TikZ export
- Animation support

## License

MIT OR Apache-2.0
