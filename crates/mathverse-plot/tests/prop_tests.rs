//! Property-style tests: deterministic loops over thousands of inputs.
//! No `proptest` dependency — these use a fixed LCG so failures reproduce.

use mathverse_plot::axes::Scale;
use mathverse_plot::boxplot::BoxStats;
use mathverse_plot::style::Color;

/// Deterministic 32-bit LCG (Numerical Recipes constants).
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    fn f64(&mut self) -> f64 {
        self.next() as f64 / u32::MAX as f64
    }
}

#[test]
fn rgb_hex_roundtrip() {
    let mut rng = Lcg(0x1234_5678);
    for _ in 0..2000 {
        let (r, g, b) = (rng.next() as u8, rng.next() as u8, rng.next() as u8);
        let c = Color::rgb(r, g, b);
        let hex = c.to_hex();
        assert_eq!(hex.len(), 7, "hex length for {r},{g},{b}: {hex}");
        assert!(hex.starts_with('#'));
        assert_eq!(
            hex,
            format!("#{r:02x}{g:02x}{b:02x}"),
            "hex mismatch for {r},{g},{b}"
        );
        // Inverse: parse back and compare.
        let (r2, g2, b2) = c.to_rgb();
        assert_eq!((r2, g2, b2), (r, g, b), "to_rgb round-trip failed");
    }
}

#[test]
fn rgba_hex_roundtrip() {
    let mut rng = Lcg(0xdead_beef);
    for _ in 0..500 {
        let (r, g, b, a) = (
            rng.next() as u8,
            rng.next() as u8,
            rng.next() as u8,
            rng.next() as u8,
        );
        let hex = Color::rgba(r, g, b, a).to_hex();
        assert_eq!(hex.len(), 9, "rgba hex length: {hex}");
        assert_eq!(hex, format!("#{r:02x}{g:02x}{b:02x}{a:02x}"));
    }
}

#[test]
fn named_colors_always_emit_valid_hex() {
    let names = [
        "red", "green", "blue", "black", "white", "gray", "grey", "orange",
        "purple", "brown", "navy", "teal", "olive", "maroon", "silver",
        "lime", "fuchsia", "aqua", "yellow", "cyan", "magenta",
        "completely_unknown_name_xyz", "", "chartreuse",
    ];
    for name in names {
        let hex = Color::Named(name).to_hex();
        assert!(hex.starts_with('#'), "bare name leaked: {name:?} -> {hex}");
        assert_eq!(hex.len(), 7, "malformed hex for {name:?}: {hex}");
        // Must be parseable as hex digits.
        let digits = &hex[1..];
        assert!(digits.bytes().all(|b| b.is_ascii_hexdigit()), "{hex}");
    }
}

#[test]
fn quartiles_are_monotonic_and_in_range() {
    let mut rng = Lcg(0x0bad_cafe);
    for _ in 0..300 {
        let n = 1 + (rng.next() % 40) as usize;
        let mut xs: Vec<f64> = (0..n).map(|_| rng.f64() * 100.0).collect();
        xs.push(0.0); // ensure a 0..100 span
        let s = BoxStats::compute(&xs).unwrap();
        assert!(
            s.q1 <= s.median && s.median <= s.q3,
            "quartiles out of order: {:?}",
            (s.q1, s.median, s.q3)
        );
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(s.min >= lo && s.max <= hi, "whiskers outside data range");
        assert!(s.q1 >= lo && s.q3 <= hi, "quartiles outside data range");
    }
}

#[test]
fn linear_scale_roundtrip() {
    let mut rng = Lcg(0xfeed_face);
    for _ in 0..2000 {
        let x = (rng.f64() - 0.5) * 2.0 * 1e6; // -1e6..1e6
        let t = Scale::Linear.transform(x);
        let back = Scale::Linear.inverse(t);
        assert!(
            (back - x).abs() <= 1e-9 * x.abs().max(1.0),
            "linear roundtrip {x} -> {t} -> {back}"
        );
    }
}

#[test]
fn log_scale_roundtrip() {
    let mut rng = Lcg(0x5eed_5eed);
    for _ in 0..1000 {
        let x = 10f64.powf((rng.f64() - 0.5) * 12.0); // 1e-6 .. 1e6
        let t = Scale::Log.transform(x);
        let back = Scale::Log.inverse(t);
        assert!(
            (back - x).abs() <= 1e-9 * x,
            "log roundtrip {x} -> {t} -> {back}"
        );
    }
}

#[test]
fn sqrt_and_symlog_roundtrip() {
    let mut rng = Lcg(0xabcd_1234);
    for _ in 0..1000 {
        let x = rng.f64() * 1e3;
        let t = Scale::Sqrt.transform(x);
        let back = Scale::Sqrt.inverse(t);
        assert!((back - x).abs() <= 1e-9 * x.max(1.0));

        let s = (rng.f64() - 0.5) * 2.0 * 1e3;
        let t = Scale::SymLog.transform(s);
        let back = Scale::SymLog.inverse(t);
        assert!((back - s).abs() <= 1e-9 * s.abs().max(1.0));
    }
}

#[test]
fn xml_escape_identity_for_safe_text() {
    let safe = ["plain text", "sin(x)", "2024-01-01", "42.5", "a-b_c.d"];
    for s in safe {
        assert_eq!(mathverse_plot::common::xml_escape(s), s);
    }
    let dangerous = ["<", ">", "&", "\"", "'", "<script>"];
    for s in dangerous {
        let out = mathverse_plot::common::xml_escape(s);
        assert_ne!(out, s, "dangerous input passed through unchanged: {s:?}");
        assert!(!out.contains('<') && !out.contains('>'), "angle brackets leaked: {s:?} -> {out:?}");
    }
    // The canonical escape sequence must be emitted.
    assert_eq!(mathverse_plot::common::xml_escape("&"), "&amp;");
    assert_eq!(mathverse_plot::common::xml_escape("<"), "&lt;");
    assert_eq!(mathverse_plot::common::xml_escape(">"), "&gt;");
    assert_eq!(mathverse_plot::common::xml_escape("\""), "&quot;");
    assert_eq!(mathverse_plot::common::xml_escape("'"), "&apos;");
}
