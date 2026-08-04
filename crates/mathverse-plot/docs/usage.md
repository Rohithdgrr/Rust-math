# mathverse-plot Usage

## Quick Start

```toml
[dependencies]
mathverse-plot = { path = "../../crates/mathverse-plot" }
```

```rust
use mathverse_plot::{Color, DataPoint, DataSeries, LineStyle, PlotConfig, SvgPlot};

fn main() {
    let config = PlotConfig::new()
        .with_title("Sine Wave")
        .with_x_label("x (radians)")
        .with_y_label("sin(x)")
        .with_dimensions(800, 400);

    let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
    let points: Vec<DataPoint> =
        xs.iter().zip(ys.iter()).map(|(x, y)| DataPoint::new(*x, *y)).collect();

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("sin(x)".into(), points));
    std::fs::write("sine.svg", plot.generate()).unwrap();
}
```

## Backends

| Backend | Type | When to use |
|---|---|---|
| `SvgPlot` | `generate() -> String` | Vector output for docs, web, print |
| `HtmlPlot` | `generate() -> String` | Self-contained page (embeds SVG) |
| `TerminalPlot` | `generate() -> String` | CLI tools, logs, quick checks |

All three share the same data types (`DataPoint`, `DataSeries`, `PlotConfig`),
so a plot defined once can be rendered to every backend.

## Styling

```rust
use mathverse_plot::{Color, MarkerStyle, PlotStyle};

let style = PlotStyle::default()
    .with_line_color(Color::RED)
    .with_line_style(LineStyle::Dashed)
    .with_line_width(2.0)
    .with_marker_style(MarkerStyle::Circle)
    .with_marker_size(4.0);
```

Line styles: `Solid`, `Dashed`, `Dotted`, `DashDot`.
Markers: `Circle`, `Square`, `Triangle`, `Cross`, `Plus`, `Diamond`, `None`.
Named colors: `BLACK` `WHITE` `RED` `GREEN` `BLUE` `YELLOW` `CYAN` `MAGENTA`
`GRAY` `ORANGE` `PURPLE` `BROWN`, plus `Color::rgb(r,g,b)` / `Color::rgba(...)`.

## Multi-Series with MathVerse Statistics

Compute is always delegated to mathverse crates:

```rust
use mathverse_plot::{DataPoint, DataSeries, PlotConfig, SvgPlot};
use mathverse_statistics::descriptive::{mean, stddev};

let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let ys = vec![2.1, 3.9, 6.2, 7.8, 10.3];

let m = mean(&ys);
let s = stddev(&ys);

let data: Vec<DataPoint> = xs.iter().zip(&ys).map(|(x, y)| DataPoint::new(*x, *y)).collect();
let mean_line: Vec<DataPoint> = xs.iter().map(|x| DataPoint::new(*x, m)).collect();

let mut plot = SvgPlot::new(PlotConfig::new().with_title("Fit vs Mean"));
plot.add_series(DataSeries::new("data".into(), data));
plot.add_series(DataSeries::new(format!("mean = {m:.2}"), mean_line));
// stddev bands, confidence intervals: same pattern with mathverse-statistics
std::fs::write("stats.svg", plot.generate()).unwrap();
```

## matplotlib Migration Cheat Sheet

| matplotlib | mathverse-plot |
|---|---|
| `plt.plot(x, y)` | `add_series(DataSeries::new(.., points))` |
| `plt.xlabel("t")` | `PlotConfig::with_x_label("t")` |
| `plt.title("T")` | `PlotConfig::with_title("T")` |
| `plt.grid()` | `PlotConfig::with_grid(true)` |
| `plt.legend()` | `PlotConfig::with_legend(true)` |
| `color='r', linestyle='--'` | `PlotStyle` builder |
| `plt.savefig("f.svg")` | `std::fs::write("f.svg", plot.generate())` |
| `plt.show()` | `println!("{}", plot.generate())` |

## Error Handling

`generate()` never panics on empty or constant data. Empty series produce an
empty plot with a grid. Mathverse errors surface as typed `PlotError`
variants, never panics.

## Conventions

- Prefer `DataSeries` + `PlotConfig` over ad-hoc render calls: every chart is
  just data + config + a backend.
- `f64` throughout: match the mathverse ecosystem convention.
- Check `docs/usage.md` siblings for internals: `architecture.md` (design),
  `working.md` (how it works), `workflow.md` (contributing), `phasewiseplan.md`
  (roadmap).
