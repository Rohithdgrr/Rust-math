# MathVerse Units

[![Crates.io](https://img.shields.io/crates/v/mathverse-units.svg)](https://crates.io/crates/mathverse-units)
[![docs.rs](https://docs.rs/mathverse-units/badge.svg)](https://docs.rs/mathverse-units)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Compile-time dimensional analysis and unit conversion for the MathVerse ecosystem.

---

## Features

- **Type-safe dimensions** — 7 SI base dimensions encoded at the type level via `typenum`
- **Dimensional arithmetic** — `Add`, `Sub`, `Mul<f64>`, `Div<f64>` preserve dimension correctness
- **SI base units** — Meter, Kilogram, Second, Ampere, Kelvin, Mole, Candela
- **SI derived units** — Newton, Joule, Watt, Pascal, Hertz
- **14 dimension types** — Length, Mass, Time, Velocity, Force, Energy, Power, etc.
- **Conversion factors** — length, mass, time, temperature, energy, pressure conversions
- **Temperature functions** — Celsius ↔ Kelvin ↔ Fahrenheit

## Module Overview

| Module | Items | Description |
|---|---|---|
| `dimensions` | 14 dimension structs + `Dimension` trait | Type-level dimension encoding |
| `quantity` | `Quantity<D, U>` struct + ops | Generic physical quantity |
| `si` | 12 unit marker types + `SiUnit` trait | SI base & derived units |
| `conversions` | 20+ constants + 6 functions | Conversion factors & temperature |

## Installation

```toml
[dependencies]
mathverse-units = "0.1"
```

## Quick Start

```rust
use mathverse_units::*;

fn main() {
    // Type-safe: Length / Time = Velocity
    let distance: Quantity<LengthDim, Meter> = Quantity::new(100.0);
    let time: Quantity<TimeDim, Second> = Quantity::new(9.58);
    // let speed = distance / time;  // Would produce Quantity<VelocityDim, ...>

    // Temperature conversions
    let f = conversions::celsius_to_fahrenheit(100.0);
    println!("100°C = {f}°F");

    let k = conversions::fahrenheit_to_kelvin(32.0);
    println!("32°F = {k} K");

    // Conversion factors
    let inches = 1.0 * conversions::METER_TO_INCH;
    println!("1 meter = {inches:.4} inches");
}
```

Expected output:

```
100°C = 212°F
32°F = 273.15 K
1 meter = 39.3701 inches
```

## Per-Module Reference

### `dimensions` — Type-Level Dimensions

Each dimension struct implements `Dimension` with 7 associated `typenum` integers:

| Struct | L | M | T | I | Th | N | J | Meaning |
|---|---|---|---|---|---|---|---|---|
| `Dimensionless` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | No dimension |
| `LengthDim` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | Length [L] |
| `MassDim` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | Mass [M] |
| `TimeDim` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | Time [T] |
| `VelocityDim` | 1 | 0 | −1 | 0 | 0 | 0 | 0 | L/T |
| `AccelerationDim` | 1 | 0 | −2 | 0 | 0 | 0 | 0 | L/T² |
| `ForceDim` | 1 | 1 | −2 | 0 | 0 | 0 | 0 | ML/T² |
| `EnergyDim` | 2 | 1 | −2 | 0 | 0 | 0 | 0 | ML²/T² |
| `PowerDim` | 2 | 1 | −3 | 0 | 0 | 0 | 0 | ML²/T³ |
| `FrequencyDim` | 0 | 0 | −1 | 0 | 0 | 0 | 0 | 1/T |
| `AreaDim` | 2 | 0 | 0 | 0 | 0 | 0 | 0 | L² |
| `VolumeDim` | 3 | 0 | 0 | 0 | 0 | 0 | 0 | L³ |
| `PressureDim` | −1 | 1 | −2 | 0 | 0 | 0 | 0 | M/(LT²) |
| `DensityDim` | −3 | 1 | 0 | 0 | 0 | 0 | 0 | M/L³ |

### `quantity` — Physical Quantity

```rust
pub struct Quantity<D: Dimension, U: SiUnit> {
    pub value: f64,
}
```

| Method | Description |
|---|---|
| `Quantity::new(value)` | Create from scalar |
| `.value()` | Get numeric value |
| `.convert(factor)` | Convert to another unit |

| Operator | Description |
|---|---|
| `q1 + q2` | Same-dimension addition |
| `q1 - q2` | Same-dimension subtraction |
| `q * scalar` | Scale quantity |
| `q / scalar` | Inverse scale |

### `si` — SI Unit Markers

**Base units:**

| Type | Name | Symbol |
|---|---|---|
| `Meter` | meter | m |
| `Kilogram` | kilogram | kg |
| `Second` | second | s |
| `Ampere` | ampere | A |
| `Kelvin` | kelvin | K |
| `Mole` | mole | mol |
| `Candela` | candela | cd |

**Derived units:**

| Type | Name | Symbol | Meaning |
|---|---|---|---|
| `Newton` | newton | N | Force: kg·m/s² |
| `Joule` | joule | J | Energy: kg·m²/s² |
| `Watt` | watt | W | Power: kg·m²/s³ |
| `Pascal` | pascal | Pa | Pressure: kg/(m·s²) |
| `Hertz` | hertz | Hz | Frequency: 1/s |

### `conversions` — Conversion Factors

**Length (from meters):**

| Constant | Value |
|---|---|
| `METER_TO_CENTIMETER` | 100 |
| `METER_TO_MILLIMETER` | 1000 |
| `METER_TO_KILOMETER` | 0.001 |
| `METER_TO_INCH` | 39.3701 |
| `METER_TO_FOOT` | 3.28084 |
| `METER_TO_YARD` | 1.09361 |
| `METER_TO_MILE` | 6.21371e-4 |

**Mass (from kilograms):**

| Constant | Value |
|---|---|
| `KILOGRAM_TO_GRAM` | 1000 |
| `KILOGRAM_TO_MILLIGRAM` | 1e6 |
| `KILOGRAM_TO_POUND` | 2.20462 |
| `KILOGRAM_TO_OUNCE` | 35.274 |

**Time (from seconds):**

| Constant | Value |
|---|---|
| `SECOND_TO_MILLISECOND` | 1000 |
| `SECOND_TO_MICROSECOND` | 1e6 |
| `SECOND_TO_NANOSECOND` | 1e9 |
| `SECOND_TO_MINUTE` | 1/60 |
| `SECOND_TO_HOUR` | 1/3600 |
| `SECOND_TO_DAY` | 1/86400 |

**Energy (from joules):**

| Constant | Value |
|---|---|
| `JOULE_TO_CALORIE` | 0.239006 |
| `JOULE_TO_KILOWATT_HOUR` | 2.77778e-7 |
| `JOULE_TO_ELECTRONVOLT` | 6.242e18 |

**Pressure (from pascals):**

| Constant | Value |
|---|---|
| `PASCAL_TO_BAR` | 1e-5 |
| `PASCAL_TO_ATMOSPHERE` | 9.86923e-6 |
| `PASCAL_TO_MMHG` | 0.00750062 |

**Temperature functions:**

| Function | Description |
|---|---|
| `celsius_to_kelvin(c)` | K = °C + 273.15 |
| `kelvin_to_celsius(k)` | °C = K − 273.15 |
| `fahrenheit_to_celsius(f)` | °C = (°F − 32) × 5/9 |
| `celsius_to_fahrenheit(c)` | °F = °C × 9/5 + 32 |
| `fahrenheit_to_kelvin(f)` | K = (°F + 459.67) × 5/9 |
| `kelvin_to_fahrenheit(k)` | °F = K × 9/5 − 459.67 |

## Dependencies

- `mathverse-core`
- `typenum 1.17`
- `frunk 0.4`

## Future Scope

- Imperial unit set (foot, pound, Fahrenheit as base)
- CGS unit set
- Compile-time unit checking via `const generics`
- Compound derived units (N·m, kg/m³)
- Unit-aware arithmetic errors at compile time

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE) for details.
