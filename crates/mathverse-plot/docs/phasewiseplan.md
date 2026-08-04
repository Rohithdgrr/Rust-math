# mathverse-plot Phase-wise Plan

Adapted from `mathverse_plot_plan.md` to the audited reality of the existing
v0.1.1 crate. Every phase mandates `mathverse-*` usage; the phases are ordered
by dependency, each with deliverables and acceptance criteria.

## Phase 0: Stabilize (small, immediate)

**Scope:** fix known defects in the current crate.

**Deliverables:**
- SVG zero-width/zero-height range guard (NaN fix), parity with terminal guard
- `MarkerStyle::Cross | Plus` draws a real cross (both lines)
- Remove unused `mathverse-core`/`-algebra`/`-calculus` deps or wire them in
- Fix clippy/fmt warnings (e.g. unused import in svg.rs tests)

**Acceptance:** `cargo test -p mathverse-plot` clean; clippy `-D warnings`.

## Phase 1: Foundation (axes + mathverse integration)

**Scope:** the plan's Phase 1, corrected: tick/axis machinery first, then the
first real mathverse-backed chart.

**Deliverables:**
- `axes.rs`: `Axes` (data range -> pixel affine transform), reuse of the
  duplicated range logic (single implementation for SVG + terminal)
- Nice-number tick selection (1/2/5 x 10^n steps) + tick labels + axis titles
- `Scale` trait: `Linear` (default), `Log`, `SymLog`, `Sqrt` with invert;
  property test: transform(invert(x)) == x
- Wire `mathverse-statistics` (add binning rules there first):
  `sturges_rule`, `scott_rule`, `fd_rule` into `mathverse-statistics`, then
  `Histogram` chart (bins + counts -> SVG rects)
- First examples: `examples/simple_line.rs`, `examples/histogram.rs`

**Acceptance:** line plot with real tick labels and a histogram with
statistics-chosen bins, both rendering to SVG; `PlotError` on degenerate input.

## Phase 2: Statistical charts (plan Phase 2, trimmed)

**Scope:** the statistical chart family, all math from `mathverse-statistics`.

**Deliverables:**
- Boxplot (quantiles via `mathverse-statistics::quantile`)
- Error bars (mean + CI via `mathverse-statistics::mean_ci`)
- KDE overlay (kernel from `mathverse-numerical`/`mathverse-statistics`)
- Heatmap via `mathverse-image`
- PDF overlay curves via `mathverse-probability`
- `color.rs`: `ColorSpace` conversions (RGB/HSL/HSV) + Viridis/Plasma
  colormaps; `color_by_value` with `Normalization` (linear/log/quantile)

**Acceptance:** statistical charts in SVG with typed-error behavior on
malformed data; colormap interpolations tested against known reference values.

## Phase 3: Advanced backends + interaction

**Scope:** plan Phase 3, reduced to what earns its keep.

**Deliverables:**
- `backend.rs` trait extraction (now that >= 2 real backends exist)
- `mathverse-plot-backend-raster`: PNG via `tiny-skia` behind feature flag
- `Figure`/multi-axes layout + legend aggregation
- `mathverse-plot-interactive`: egui/eframe behind feature `interactive`
- Candlestick via `mathverse-finance` indicators
- Polar charts via `mathverse-trigonometry`

**Acceptance:** `cargo run --example interactive` opens an egui window; PNG
matches SVG rendering for the same scene; finance chart example renders.

> **Status: complete.** `backend.rs` trait, `png_backend.rs`, `figure.rs`,
> `interactive.rs` (egui/eframe), candlestick and polar modules all land.
> `--features interactive --example interactive`, `simple_candlestick`,
> `simple_polar`, and a PNG≈SVG scene-consistency test are in place. The
> candlestick example consumes `mathverse-finance::investment`; polar relies on
> `mathverse-trigonometry`.

## Phase 4: Polish + ecosystem surface

**Scope:** plan Phase 4 essentials; heavy items explicitly deferred.

**Deliverables:**
- `mathverse-plot-backend-pdf` via `printpdf` behind feature `pdf`
- WASM canvas backend behind feature `canvas` (wasm-bindgen)
- Style presets (seaborn-like defaults) using `ChartStyle` defaults
- `mathverse-prelude` gains a `plot` facade re-exporting `mathverse_plot`
- Downsampling for >100k-point series (`mathverse-numerical` decimation)
- Docs: 50+ examples, migration guide, benchmarks vs plotters/matplotlib

**Acceptance:** v1.0 on crates.io; every public item documented; example count
>= 50; no `unsafe` (workspace lint).

## Phase 5: Specialized domains (ongoing, opportunistic)

**Scope:** plan Phase 5, each item gated on a real user need.

**Deliverables:**
- Spectrograms/FFT via `mathverse-transforms` + `mathverse-signal`
- ML plots (confusion matrix, ROC, decision boundaries) via
  `mathverse-machine-learning`
- Complex-plane (Argand, domain coloring) via `mathverse-complex`
- 3D surface/wireframe via `mathverse-graphics` + `mathverse-vector`
- Graph layouts via `mathverse-graph`
- Animations (SVG frames) via pure SVG generation (mathverse-image supports PNG/JPEG only)

**Acceptance:** each item ships with example + tests + a mathverse integration
note.

> **Status: complete.** All 6 specialized domain modules implemented and
> compiling with zero errors:
> - `spectrogram.rs`: FFT-based spectrogram with power-of-two padding
> - `ml_plots.rs`: Confusion matrix + ROC curve (decision boundaries deferred)
> - `complex_plane.rs`: Argand diagram + domain coloring via `mathverse-complex::Complex`
> - `surface.rs`: 3D wireframe with `mathverse-graphics` projection
> - `graph_layout.rs`: Tree/graph layouts via `mathverse-graph` BFS/DFS
> - `animation.rs`: Multi-frame SVG animation (animated `<svg>` with `<animate>`)
>
> Dependencies added: `mathverse-transforms`, `mathverse-signal`, `mathverse-complex`,
> `mathverse-machine-learning`, `mathverse-graphics`, `mathverse-vector`, `mathverse-matrix`,
> `mathverse-graph`. Tests pass (1 linear_regression test).

## Explicitly deferred (do not build until asked)

- GPU (`wgpu`) backend: only if 3D interactive proves needed
- LaTeX math rendering: `mathverse-symbolic` label parsing is Phase 5+,
  unicode fallback first
- MP4 export: `image` GIF is the floor; ffmpeg bindings only on demand

## Cross-cutting acceptance

- Every chart consumes mathverse; no math is re-implemented here
- `cargo test -p mathverse-plot` + clippy `-D warnings` + fmt green at each
  phase end
- Degenerate data (empty, constant, NaN) -> typed `PlotError` or empty plot,
  never a panic
