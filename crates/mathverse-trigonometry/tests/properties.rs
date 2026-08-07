//! Integration tests closing the audit's testing gaps:
//! property tests (sin²+cos²=1, asin∘sin), f32 coverage, extended Chebyshev,
//! batched NaN handling, sinc near zero, and cot/tan_half consistency.

use mathverse_trigonometry::*;

/// Deterministic LCG so the tests are reproducible with no RNG dependency.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn in_unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.in_unit()
    }
}

const N: usize = 1000;

#[test]
fn pythagorean_identity_f64() {
    let mut rng = Lcg(42);
    for _ in 0..N {
        let x = rng.range(-100.0, 100.0);
        let s = sin(x);
        let c = cos(x);
        assert!((s * s + c * c - 1.0).abs() < 1e-9, "x={x}");
    }
}

#[test]
fn pythagorean_identity_f32() {
    let mut rng = Lcg(7);
    for _ in 0..N {
        let x = rng.range(-100.0, 100.0) as f32;
        let s = sin(x);
        let c = cos(x);
        assert!((s * s + c * c - 1.0).abs() < 1e-5, "x={x}");
    }
}

#[test]
fn asin_sin_inverse() {
    let mut rng = Lcg(99);
    for _ in 0..N {
        let x = rng.range(-core::f64::consts::FRAC_PI_2, core::f64::consts::FRAC_PI_2);
        assert!((asin(sin(x)) - x).abs() < 1e-9, "x={x}");
    }
}

#[test]
fn atan2_polar_roundtrip() {
    let mut rng = Lcg(1234);
    for _ in 0..N {
        let x = rng.range(-1e6, 1e6);
        let y = rng.range(-1e6, 1e6);
        let (r, theta) = cartesian_to_polar(x, y);
        let (xx, yy) = polar_to_cartesian(r, theta);
        assert!((xx - x).abs() < 1e-6 * r.max(1.0), "x={x} y={y}");
        assert!((yy - y).abs() < 1e-6 * r.max(1.0), "x={x} y={y}");
    }
}

#[test]
fn chebyshev_cos_relation_extended() {
    // Tₙ(cos θ) = cos(nθ) for n up to 20.
    let mut rng = Lcg(5);
    for n in 0..=20u32 {
        for _ in 0..50 {
            let theta = rng.range(0.0, core::f64::consts::PI);
            let expected = (n as f64 * theta).cos();
            let actual = chebyshev_first(n, theta.cos());
            assert!((actual - expected).abs() < 1e-8, "n={n} theta={theta}");
        }
    }
}

#[test]
fn f32_coverage_no_nan() {
    // Every public math function accepts f32 without returning NaN on in-domain input.
    let x = 0.7f32;
    let y = 1.3f32;
    let ok: [f32; 40] = [
        sin(x), cos(x), tan(x), cot(x), sec(x), csc(x),
        sinh(x), cosh(x), tanh(x), coth(x), sech(x), csch(x),
        asin(x), acos(x), atan(x), atan2(y, x), acot(x),
        asinh(x), acosh(2.0), atanh(x), acoth(2.0), asech(x), acsch(x),
        sin_deg(30.0), cos_deg(60.0), tan_deg(45.0), cot_deg(30.0),
        sind(30.0), cosd(60.0), tand(45.0),
        asin_deg(x), acos_deg(x), atan_deg(x), atan2_deg(y, x), acot_deg(x),
        sinpi(x), cospi(x),
        sinc(x), sinc_unnorm(x), versine(x),
    ];
    for v in ok {
        assert!(!v.is_nan() && v.is_finite(), "got {v}");
    }

    let ok2: [f32; 12] = [
        coversine(x), haversine(x), exsecant(x), excosecant(x),
        gudermannian(x), gudermannian_inv(x), gudermannian_alt(x),
        chebyshev_first(3, x), chebyshev_second(3, x),
        sin_power(3, x), cos_power(3, x),
        wrap_angle(-4.5),
    ];
    for v in ok2 {
        assert!(!v.is_nan() && v.is_finite(), "got {v}");
    }

    assert!(map_sin(&[x, y], &mut [0.0f32; 2]));
    assert!(sum_sin(&[x, y]).is_finite());
}

#[test]
fn sinc_small_f32_no_nan() {
    // f32 sinc at tiny arguments must not become NaN (audit P0).
    for e in -25..=-8 {
        let x = 10f32.powi(e);
        let v = sinc(x);
        assert!(v.is_finite() && !v.is_nan(), "sinc(10^{e}) = {v}");
    }
    assert!((sinc(0.0f32) - 1.0).abs() < 1e-6);
    assert!((sinc(1e-10f32) - 1.0).abs() < 1e-3);
}

#[test]
fn batched_nan_passthrough() {
    // NaN inputs propagate to NaN outputs (documented behavior), never panic.
    let xs = [0.0f64, f64::NAN, 1.0];
    let mut out = [0.0; 3];
    assert!(map_sin(&xs, &mut out));
    assert!(out[1].is_nan());
    assert!(sum_sin(&xs).is_nan());
}

#[test]
fn cot_and_tan_half_consistency() {
    // cot matches 1/tan (the two definitions) near the asymptote and elsewhere.
    for x in [0.1f64, 1.0, core::f64::consts::FRAC_PI_2 - 1e-9, core::f64::consts::FRAC_PI_2] {
        assert!((cot(x) - 1.0 / tan(x)).abs() < 1e-9 * cot(x).abs().max(1.0), "x={x}");
    }
    // tan_half matches the direct half-angle tan for well-scaled inputs.
    for x in [0.1f64, 0.5, 1.0, 2.5, -1.7, 6.0] {
        assert!((tan_half(x) - (x / 2.0).tan()).abs() < 1e-9, "x={x}");
    }
}
