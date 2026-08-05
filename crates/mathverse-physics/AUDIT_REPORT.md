# MathVerse Physics — Audit Report

**Scope:** `crates/mathverse-physics` (v0.1.1)
**Date:** 2026-08-05
**Method:** Source review, `cargo build`, `cargo test`, `cargo clippy`, and a scratch-crate compile of the README example.

## Build health

| Check | Result |
|-------|--------|
| `cargo build -p mathverse-physics` | ✅ compiles |
| `cargo test -p mathverse-physics` | ✅ 13/13 pass |
| `cargo clippy -p mathverse-physics` | ⚠️ 4 warnings (blocking under CI `-D warnings`) |
| README quick-start example | ❌ does not compile |

---

## 1. BUG: README example does not compile (High)

`README.md` quick-start uses `mechanics::G_0`:

```rust
let v = mechanics::final_velocity(0.0, mechanics::G_0, t);
```

But in `src/mechanics.rs:3` the import is **private**:

```rust
use crate::constants::G_0;   // not `pub use`
```

`G_0` is therefore **not** a public path of the `mechanics` module → `E0603` (private constant). Verified with a scratch `cargo check`.

**Also wrong:** the claimed output `Impact velocity: 9.90 m/s` does not match the code. `final_velocity(0.0, 9.80665, sqrt(10)) ≈ 31.0 m/s`. The `t = sqrt(10)` does not even correspond to the comment's "free-fall from 100 m" (that needs `t≈4.51 s`, `v≈44.3 m/s`).

**Fix:** `pub use crate::constants::G_0;` in `mechanics.rs` (or reference `crate::G_0` / `constants::G_0`), and correct the example's arithmetic so the documented output is real. This is a user-facing correctness bug.

## 2. BUG: unused `K_B` import + hard-coded gas constant (Medium)

`src/thermodynamics.rs:3` imports `crate::constants::K_B` but never uses it — it hard-codes `8.314_462_618` in two places (`ideal_gas_pressure` line 14, `ideal_gas_temperature` line 27) instead of deriving the molar gas constant:

```rust
pub const R: f64 = N_A * K_B;   // R = 8.314462618... exactly
```

The proper fix removes the duplicate magic number **and** clears the `unused_imports` warning:

- Add `R` to `constants.rs` (it is CODATA-standard and currently missing — README implies 17 constants).
- In `thermodynamics.rs` replace `8.314_462_618` with `crate::constants::R`.
- Drop the aged `use crate::constants::K_B;` import (or keep `K_B` if `R` lives in `thermodynamics`).

## 3. BUG: silent `NaN`/`inf` on out-of-domain input (High, systemic)

Every function returns plain `f64`. Many silently return `NaN`/`inf` when the mathematical domain is violated. Nothing documents this; `f64` NaN propagates invisibly downstream. Grep-confirmed high-risk cases:

| Function | Location | Silent failure |
|----------|----------|----------------|
| `velocity_from_displacement` | mechanics.rs:40 | `sqrt(v0²+2ad)` → `NaN` when `v0²+2ad < 0` |
| `power` | mechanics.rs:114 | `w/0` → `inf` |
| `centripetal_force` | mechanics.rs:127 | `r=0` → `inf` |
| `gravitational_force` | mechanics.rs:140 | `r=0` → `inf` |
| `pendulum_period` | mechanics.rs:152 | `l<0`→`NaN`; `g≤0`→`inf`/`NaN` |
| `angular_velocity` | mechanics.rs:176 | `r=0` → `inf` |
| `ideal_gas_pressure` / `_temperature` | thermo.rs:14,27 | `v=0` / `n=0` → `inf` |
| `carnot_efficiency` | thermo.rs:78 | `T_cold>T_hot`→negative; `T_hot=0`→`NaN` |
| `entropy_change` | thermo.rs:90 | `t=0` → `inf` |
| `heat_conduction` | thermo.rs:117 | `d=0` → `inf` |
| `snells_law` | waves.rs:166 | total internal reflection → `NaN` (`asin > 1`) |
| `critical_angle` | waves.rs:178 | `n2>n1` → `NaN` |
| `lens_focal_length` | waves.rs:191 | `n=1` or `r1=r2` → `inf` |
| `thin_lens_equation` | waves.rs:203 | `d_o=f` → `inf` |
| `single_slit_diffraction` | waves.rs:228 | `mλ/a > 1` → `NaN` |
| `double_slit_interference` | waves.rs:241 | `mλ/d > 1` → `NaN` |
| `doppler_source_moving` | waves.rs:92 | `v_source → -v_wave` → `inf` |
| `string_wave_speed` | waves.rs:129 | `linear_density=0` → `inf` |

**Recommendation (choices, cheapest first):**
1. **Minimal:** add a `# Panics` / `# Returns` doc section stating the domain for each function, and prettify nothing. Smallest diff, still leaves silent NaN.
2. **Better:** return `Option<f64>` from the domain-sensitive functions (`snells_law`, `critical_angle`, `single_slit_diffraction`, `double_slit_interference`, `carnot_efficiency`, `velocity_from_displacement`, `thin_lens_equation`, `lens_focal_length`). Breaking change; bump to 0.2.
3. **Robust:** introduce a thin error surface — the workspace already has `mathverse-core::error::MathResult` (currently an **unused** dependency, see §4) — so `Option`/`Result` is available without new deps.

At minimum, do not leave `snells_law`/`critical_angle`/diffraction returning `NaN` for the most common real inputs (`mλ/d` *normally exceeds 1* after a few orders — a silent, surprising bug for any caller).

## 4. BUG: all 6 declared dependencies are unused (Medium)

`Cargo.toml` declares:

```
mathverse-core, mathverse-calculus, mathverse-algebra,
mathverse-trigonometry, mathverse-vector, mathverse-units
```

Grep across `src/**/*.rs` shows **zero** references to any `mathverse_*` crate — only `std::f64::consts` is used. The dependencies are dead weight: slower builds, larger dependency graph, more rebuild churn, and they mask the fact that none of the shared infrastructure is actually being used.

**Fix:** delete them (or, if refactoring, use them — e.g. `mathverse-core::MathResult` for §3 and `mathverse-vector`/`mathverse-units` wherever vector/unit-typed physics would help). Verify with `cargo machete` if installed.

## 5. CI will fail under `--workspace ... -D warnings` (High, workspace-wide)

The CI `clippy` job runs `cargo clippy --workspace --all-targets -- -D warnings`. Two blockers:

1. **Workspace lint typo:** `Cargo.toml:68` sets `clippy::empty_line_after_doc_comment` (singular) but the correct lint is `empty_line_after_doc_comments` (plural). Clippy emits `E0602 unknown lint` *for every crate* in the workspace → promoted to a hard error by `-D warnings`. Rename it.
2. **Unrelated but blocking:** `mathverse-optimization` fails to compile under clippy (5 errors seen in `gradient.rs` / `constrained.rs` / `unconstrained.rs` / `line_search.rs` / `combinatorial.rs`). This makes the whole-workspace clippy gate red regardless of physics.

Physics-crate specific clippy findings to fix once build is green:
- `unused import: crate::constants::K_B` — thermodynamics.rs:3 (see §2).
- `doc_markdown` — lib.rs:1: `MathVerse` should be `` `MathVerse` ``.
- `cast_lossless` — waves.rs:229,242: `m as f64` → `f64::from(m)`.

## 6. Test coverage is very low (Medium)

Only **13 tests** cover **~60 public functions** (~20%). Entirely untested:

- **mechanics:** `velocity_from_displacement`, `potential_energy`, `work`, `power`, `centripetal_force`, `gravitational_force`, `pendulum_period`, `spring_force`, `angular_velocity`, both moments of inertia.
- **thermodynamics:** all ideal-gas fns, `internal_energy_change`, `work_isobaric`, `heat_isobaric`, `entropy_change`, `linear_expansion`, `heat_conduction`, `heat_radiation`, `specific_heat`, both Fahrenheit conversions.
- **electromagnetism:** everything except `resistor_power` and `capacitor_energy`.
- **waves:** everything except `wave_speed`, `period`, `speed_of_sound_air`.

No test asserts behavior at domain boundaries (the §3 bugs). At minimum add one test per family that exercises a normal case **and** a boundary/NaN case so the silence becomes a failure.

## 7. Minor issues

- **`lib.rs` doc mismatch:** the crate doc claims a "Modern physics" section but no such module exists. Update the doc, or note it as roadmap.
- **Doc-panic sections missing:** no function documents `# Panics` / domain constraints (tied to §3).
- **Dependency version drift:** `mathverse-trigonometry = "0.2.0"` while other path deps are `"0.1.1"`; verify intended (path overrides make it moot locally but matters for published metadata).
- **`speed_of_sound_air`** uses the linear approximation `331 + 0.6T` — fine for teaching, but worth a doc note that it's approximate and unphysical below ~0 °C / extreme temperatures.

---

## Recommended priority order

1. Fix §1 (public-facing broken example + wrong output).
2. Fix §5.1 workspace lint typo (blocks all CI; affects every crate).
3. Fix §2 (± §4) — removes the only compile warning, centralizes the gas constant.
4. Decide on §3 NaN policy (doc ≥ `Option` ≥ `MathResult`); at least fix `snells_law`/diffraction/TIR which fail on ordinary inputs.
5. Add §6 boundary tests as you touch each function.
6. Clean unused deps (§4) — can be done in the same PR as §3 if adopting `MathResult`.