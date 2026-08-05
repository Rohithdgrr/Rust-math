# mathverse-plot — Fix Plan

Prioritized list of issues from the dual audit (6.2/10 + 7.3/10) and direct code inspection.

## P0 — Do First (blocks production use)

| ID | Fix | Status |
|----|-----|--------|
| P0-1 | Fix README quick-start compile errors (`impl Into<String>` + `.clone()`) | ✅ Done |
| P0-2 | XML-escape SVG text content (`<`, `>`, `&`, `"`) | ✅ Done |
| P0-3 | Fix violin plot quartiles (linear interpolation + 1.5×IQR whiskers) | ✅ Done |
| P0-4 | Remove global clippy suppressions from `lib.rs` | ✅ Done |
| P0-5 | Fix `Color::Named::to_hex()` to return hex codes | ✅ Done |
| P0-6 | Reconcile `docs/architecture.md` with reality | ✅ Done |

## P1 — Next milestone (usability & correctness)

| ID | Fix | Status |
|----|-----|--------|
| P1-1 | Extract shared range computation into `common.rs` | ✅ Done |
| P1-2 | Change `Backend::generate` return type to `PlotOutput` enum | ⏳ Deferred (trait-breaking API change) |
| P1-3 | Add `--all-features` to CI for `mathverse-plot` | ⏳ No CI config in local clone |
| P1-4 | Guard `png_backend::render` against degenerate dimensions | ✅ Done |
| P1-5 | Differentiate `MarkerStyle::Cross` vs `Plus` in `svg.rs` | ✅ Done |

## P2 — Short-term improvements (Month 1)

| ID | Fix | Status |
|----|-----|--------|
| P2-1 | Add `impl Into<String>` to all `DataSeries`/`BoxData` constructors | ✅ Done (side effect of P0-1) |
| P2-2 | Add font metrics or char-width lookup for legend sizing | ⏳ Skipped (0.6× approximation is acceptable) |
| P2-3 | Use shared base64 encoding utility | ✅ Done |
| P2-4 | Make `mathverse-finance`/`ml`/`graph` optional deps | ⏳ Pending |
| P2-5 | Add SVG `<title>` tooltips | ⏳ Pending |
| P2-6 | Add `ndarray` interop feature flag | ⏳ Pending |

## Execution order

```
Week 1:  P0-1, P0-4, P0-6    ✅ Done
Week 2:  P0-2, P0-5           ✅ Done
Week 3:  P0-3                 ✅ Done
Week 4:  P1-1, P1-4, P1-5    ✅ Done
Month 2: P1-2, P1-3           ⏳ Deferred / blocked
Month 3: P2-1 through P2-6    P2-1 ✅, P2-3 ✅, rest pending
```

## The single highest-impact change

P0-3 (violin quartiles) — a correctness bug that produces wrong statistical output, fixable in ~15 lines.
The second highest is P0-2 (XML escaping) — a security/data-integrity issue affecting every SVG output.