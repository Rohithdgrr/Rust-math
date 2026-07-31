# MathVerse Units

A production-grade Rust library for compile-time dimensional analysis and type-safe unit conversions using Rust's type system.

## Features

- **Compile-Time Dimension Checking**: Catch unit errors at compile time
- **SI Unit Support**: Full support for SI base and derived units
- **Type-Safe Arithmetic**: Dimension-aware operations that prevent unit mismatches
- **Unit Conversions**: Conversion factors for common units
- **Zero-Cost Abstractions**: No runtime overhead for dimension checking

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mathverse-units = "0.1.0"
```

## Usage

### Basic Quantities

```rust
use mathverse_units::{Quantity, dimensions::LengthDim, si::Meter};

// Create a length quantity
let length: Quantity<LengthDim, Meter> = Quantity::new(5.0);
println!("Length: {} m", length.value());
```

### Arithmetic Operations

```rust
use mathverse_units::{Quantity, dimensions::LengthDim, si::Meter};

let q1: Quantity<LengthDim, Meter> = Quantity::new(5.0);
let q2: Quantity<LengthDim, Meter> = Quantity::new(3.0);

let sum = q1 + q2;
let diff = q1 - q2;
let scaled = q1 * 2.0;
let divided = q1 / 2.0;
```

### Unit Conversions

```rust
use mathverse_units::conversions::{celsius_to_kelvin, kelvin_to_celsius};

let temp_c = 25.0;
let temp_k = celsius_to_kelvin(temp_c);
let temp_c_back = kelvin_to_celsius(temp_k);
```

### Temperature Conversions

```rust
use mathverse_units::conversions::{
    celsius_to_kelvin, kelvin_to_celsius,
    fahrenheit_to_celsius, celsius_to_fahrenheit,
    fahrenheit_to_kelvin, kelvin_to_fahrenheit
};

let celsius = 25.0;
let fahrenheit = celsius_to_fahrenheit(celsius);
let kelvin = celsius_to_kelvin(celsius);
```

## Dimensions

The library provides type-level dimensions for:

- **Length** (L)
- **Mass** (M)
- **Time** (T)
- **Electric Current** (I)
- **Temperature** (Th)
- **Amount of Substance** (N)
- **Luminous Intensity** (J)

Derived dimensions include:
- Velocity (L/T)
- Acceleration (L/T²)
- Force (M·L/T²)
- Energy (M·L²/T²)
- Power (M·L²/T³)
- Frequency (1/T)
- Area (L²)
- Volume (L³)
- Pressure (M/(L·T²))
- Density (M/L³)

## SI Units

Base SI units:
- Meter (m) - length
- Kilogram (kg) - mass
- Second (s) - time
- Ampere (A) - electric current
- Kelvin (K) - temperature
- Mole (mol) - amount of substance
- Candela (cd) - luminous intensity

Derived SI units:
- Newton (N) - force
- Joule (J) - energy
- Watt (W) - power
- Pascal (Pa) - pressure
- Hertz (Hz) - frequency

## Testing

Run the test suite:

```bash
cargo test
```

Run benchmarks:

```bash
cargo bench
```

## License

This project is dual-licensed under either:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting a pull request.

## Performance

The library uses zero-cost abstractions:
- Compile-time dimension checking
- No runtime overhead for type safety
- Efficient arithmetic operations

## Roadmap

- [ ] More derived units
- [ ] Custom unit definitions
- [ ] Imperial units support
- [ ] Physical constants
- [ ] Dimensional analysis for custom types
