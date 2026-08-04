# mathverse-plot Workflow

## Development Loop

```
pick a chart/feature -> check mathverse-* for the math -> implement -> test -> clippy+fmt -> example
```

Rule: before writing any math, check the matching `mathverse-*` crate. If the
math doesn't exist there, add it to that crate first (it belongs to the
ecosystem, not to the plot crate), then consume it here.

## Adding a Chart Type

1. Create `charts/<name>.rs` (or `src/<name>.rs` while flat).
2. Data comes in as `&[f64]` / `&[DataPoint]`; never build math from scratch.
3. Map the chart's need to a mathverse crate:
   - histogram bins, KDE, CIs  -> `mathverse-statistics`
   - curve smoothing, roots    -> `mathverse-numerical`
   - PDF overlays              -> `mathverse-probability`
   - special function overlays -> `mathverse-special`
4. Implement against the backend abstraction (see `architecture.md`), not
   against `SvgPlot` directly.
5. Add a `#[test]` for the math (reference values from the mathverse crate's
   tests) and one for the rendering (assert key SVG/ASCII fragments).
6. Add an example under `examples/`.

## Adding a Backend

1. Read the `Backend` contract in `backend.rs` (or `svg.rs` until extracted).
2. Implement size, primitives (line, rect, circle, path, text), save.
3. Wrap other backends rather than duplicating layout: `HtmlPlot` wraps
   `SvgPlot`; a future PNG backend can rasterize the same SVG scene.
4. Feature-gate heavy dependencies (`tiny-skia`, `printpdf`, `wgpu`).

## MathVerse Integration Protocol

- **Extend, don't fork**: new math (binning rules, tick helpers) is added to
  `mathverse-statistics` / `mathverse-numerical` etc., then re-used here.
- **Version bump discipline**: bump the mathverse crate before depending on
  the new function here.
- **No cycles**: `mathverse-plot` depends on mathverse crates; mathverse
  crates never depend on `mathverse-plot`.

## Testing

- Unit tests in each module: math correctness (reference values), rendering
  (SVG/ASCII fragment assertions), edge cases (empty, constant, single point,
  NaN).
- `cargo test -p mathverse-plot`
- `cargo clippy -p mathverse-plot -- -D warnings` and `cargo fmt --check`
  before pushing (workspace lints: pedantic clippy, `unsafe_code = forbid`).

## Example-Driven Development

Every chart/feature ships with a runnable example. Examples are the contract:
they show the intended user experience (matplotlib-like builder flow) and act
as visual regression fixtures.

## Release Checklist

1. Examples run; tests pass; clippy/fmt clean.
2. Docs updated (this folder + README + doc comments).
3. `Cargo.toml` version bump.
4. Cross-check the mathverse crate versions pinned are the ones actually used.
