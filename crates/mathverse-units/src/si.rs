//! SI unit definitions and base units.


/// SI base unit marker trait
pub trait SiUnit: Copy + Default + std::fmt::Debug {
    const NAME: &'static str;
    const SYMBOL: &'static str;
}

/// Meter - base unit of length
#[derive(Debug, Copy, Clone, Default)]
pub struct Meter;

impl SiUnit for Meter {
    const NAME: &'static str = "meter";
    const SYMBOL: &'static str = "m";
}

/// Kilogram - base unit of mass
#[derive(Debug, Copy, Clone, Default)]
pub struct Kilogram;

impl SiUnit for Kilogram {
    const NAME: &'static str = "kilogram";
    const SYMBOL: &'static str = "kg";
}

/// Second - base unit of time
#[derive(Debug, Copy, Clone, Default)]
pub struct Second;

impl SiUnit for Second {
    const NAME: &'static str = "second";
    const SYMBOL: &'static str = "s";
}

/// Ampere - base unit of electric current
#[derive(Debug, Copy, Clone, Default)]
pub struct Ampere;

impl SiUnit for Ampere {
    const NAME: &'static str = "ampere";
    const SYMBOL: &'static str = "A";
}

/// Kelvin - base unit of temperature
#[derive(Debug, Copy, Clone, Default)]
pub struct Kelvin;

impl SiUnit for Kelvin {
    const NAME: &'static str = "kelvin";
    const SYMBOL: &'static str = "K";
}

/// Mole - base unit of amount of substance
#[derive(Debug, Copy, Clone, Default)]
pub struct Mole;

impl SiUnit for Mole {
    const NAME: &'static str = "mole";
    const SYMBOL: &'static str = "mol";
}

/// Candela - base unit of luminous intensity
#[derive(Debug, Copy, Clone, Default)]
pub struct Candela;

impl SiUnit for Candela {
    const NAME: &'static str = "candela";
    const SYMBOL: &'static str = "cd";
}

/// Derived SI units
#[derive(Debug, Copy, Clone, Default)]
pub struct Newton; // Force: kg·m/s²

impl SiUnit for Newton {
    const NAME: &'static str = "newton";
    const SYMBOL: &'static str = "N";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Joule; // Energy: kg·m²/s²

impl SiUnit for Joule {
    const NAME: &'static str = "joule";
    const SYMBOL: &'static str = "J";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Watt; // Power: kg·m²/s³

impl SiUnit for Watt {
    const NAME: &'static str = "watt";
    const SYMBOL: &'static str = "W";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Pascal; // Pressure: kg/(m·s²)

impl SiUnit for Pascal {
    const NAME: &'static str = "pascal";
    const SYMBOL: &'static str = "Pa";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Hertz; // Frequency: 1/s

impl SiUnit for Hertz {
    const NAME: &'static str = "hertz";
    const SYMBOL: &'static str = "Hz";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Tesla; // Magnetic flux density: kg/(A·s²) = Wb/m²

impl SiUnit for Tesla {
    const NAME: &'static str = "tesla";
    const SYMBOL: &'static str = "T";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Weber; // Magnetic flux: kg·m²/(A·s²)

impl SiUnit for Weber {
    const NAME: &'static str = "weber";
    const SYMBOL: &'static str = "Wb";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Henry; // Inductance: kg·m²/(A²·s²)

impl SiUnit for Henry {
    const NAME: &'static str = "henry";
    const SYMBOL: &'static str = "H";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Coulomb; // Electric charge: A·s

impl SiUnit for Coulomb {
    const NAME: &'static str = "coulomb";
    const SYMBOL: &'static str = "C";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Volt; // Electric potential: kg·m²/(A·s³)

impl SiUnit for Volt {
    const NAME: &'static str = "volt";
    const SYMBOL: &'static str = "V";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Ohm; // Electric resistance: kg·m²/(A²·s³)

impl SiUnit for Ohm {
    const NAME: &'static str = "ohm";
    const SYMBOL: &'static str = "Ω";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Siemens; // Electric conductance: A²·s³/(kg·m²)

impl SiUnit for Siemens {
    const NAME: &'static str = "siemens";
    const SYMBOL: &'static str = "S";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Farad; // Capacitance: A²·s⁴/(kg·m²)

impl SiUnit for Farad {
    const NAME: &'static str = "farad";
    const SYMBOL: &'static str = "F";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Gray; // Absorbed dose: m²/s²

impl SiUnit for Gray {
    const NAME: &'static str = "gray";
    const SYMBOL: &'static str = "Gy";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Sievert; // Equivalent dose: m²/s²

impl SiUnit for Sievert {
    const NAME: &'static str = "sievert";
    const SYMBOL: &'static str = "Sv";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Katal; // Catalytic activity: mol/s

impl SiUnit for Katal {
    const NAME: &'static str = "katal";
    const SYMBOL: &'static str = "kat";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Lux; // Illuminance: cd·sr/m²

impl SiUnit for Lux {
    const NAME: &'static str = "lux";
    const SYMBOL: &'static str = "lx";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Becquerel; // Radioactivity: 1/s

impl SiUnit for Becquerel {
    const NAME: &'static str = "becquerel";
    const SYMBOL: &'static str = "Bq";
}

/// SI scaling prefixes (powers of 10).
///
/// ```rust
/// use mathverse_units::prefix::{Prefix, prefix_factor};
/// assert_eq!(prefix_factor(Prefix::Kilo), 1e3);
/// assert_eq!(prefix_factor(Prefix::Milli), 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prefix {
    Yotta,
    Zetta,
    Exa,
    Peta,
    Tera,
    Giga,
    Mega,
    Kilo,
    Hecto,
    Deka,
    Deci,
    Centi,
    Milli,
    Micro,
    Nano,
    Pico,
    Femto,
    Atto,
    Zepto,
    Yocto,
}

impl Prefix {
    /// Decimal scaling factor for this prefix.
    pub fn factor(self) -> f64 {
        match self {
            Prefix::Yotta => 1e24,
            Prefix::Zetta => 1e21,
            Prefix::Exa => 1e18,
            Prefix::Peta => 1e15,
            Prefix::Tera => 1e12,
            Prefix::Giga => 1e9,
            Prefix::Mega => 1e6,
            Prefix::Kilo => 1e3,
            Prefix::Hecto => 1e2,
            Prefix::Deka => 1e1,
            Prefix::Deci => 1e-1,
            Prefix::Centi => 1e-2,
            Prefix::Milli => 1e-3,
            Prefix::Micro => 1e-6,
            Prefix::Nano => 1e-9,
            Prefix::Pico => 1e-12,
            Prefix::Femto => 1e-15,
            Prefix::Atto => 1e-18,
            Prefix::Zepto => 1e-21,
            Prefix::Yocto => 1e-24,
        }
    }

    /// Construct a Prefix from its base-10 exponent. Returns `None` for
    /// exponents outside the SI prefix range (±24).
    pub fn from_exponent(exp: i8) -> Option<Self> {
        match exp {
            24 => Some(Prefix::Yotta),
            21 => Some(Prefix::Zetta),
            18 => Some(Prefix::Exa),
            15 => Some(Prefix::Peta),
            12 => Some(Prefix::Tera),
            9 => Some(Prefix::Giga),
            6 => Some(Prefix::Mega),
            3 => Some(Prefix::Kilo),
            2 => Some(Prefix::Hecto),
            1 => Some(Prefix::Deka),
            -1 => Some(Prefix::Deci),
            -2 => Some(Prefix::Centi),
            -3 => Some(Prefix::Milli),
            -6 => Some(Prefix::Micro),
            -9 => Some(Prefix::Nano),
            -12 => Some(Prefix::Pico),
            -15 => Some(Prefix::Femto),
            -18 => Some(Prefix::Atto),
            -21 => Some(Prefix::Zepto),
            -24 => Some(Prefix::Yocto),
            _ => None,
        }
    }
}

/// Return the decimal factor for a `Prefix` (convenience free function).
pub fn prefix_factor(p: Prefix) -> f64 {
    p.factor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_si_units() {
        let m = Meter;
        assert_eq!(Meter::NAME, "meter");
        assert_eq!(Meter::SYMBOL, "m");
    }

    #[test]
    fn test_prefix_factors() {
        assert_eq!(Prefix::Kilo.factor(), 1e3);
        assert_eq!(Prefix::Milli.factor(), 1e-3);
        assert_eq!(Prefix::Micro.factor(), 1e-6);
        assert_eq!(Prefix::Nano.factor(), 1e-9);
        assert_eq!(Prefix::Mega.factor(), 1e6);
        assert_eq!(Prefix::Yotta.factor(), 1e24);
        assert_eq!(Prefix::Yocto.factor(), 1e-24);
    }

    #[test]
    fn test_prefix_from_exponent() {
        assert_eq!(Prefix::from_exponent(3), Some(Prefix::Kilo));
        assert_eq!(Prefix::from_exponent(-3), Some(Prefix::Milli));
        assert_eq!(Prefix::from_exponent(5), None);
    }

    #[test]
    fn test_prefix_factor_fn() {
        assert_eq!(prefix_factor(Prefix::Kilo), 1e3);
    }
}
