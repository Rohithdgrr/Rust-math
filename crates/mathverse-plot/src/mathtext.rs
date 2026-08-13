//! Minimal mathtext — a lightweight stand-in for matplotlib's mathtext.
//!
//! Labels written as `$...$` are rendered with Greek letters, common symbols,
//! `_`/`^` sub- and superscripts, `\frac`, `\sqrt`, and `\text`. Everything
//! outside `$...$` is passed through as escaped plain text, so existing labels
//! are unaffected. Output is SVG (`<tspan>` runs) and can be embedded in any
//! `<text>` element.

use std::collections::HashMap;

/// Escape XML special characters. Delegates to the canonical
/// `crate::common::xml_escape` so the two implementations can never drift.
fn esc(s: &str) -> String {
    crate::common::xml_escape(s)
}

fn symbols() -> &'static HashMap<&'static str, &'static str> {
    use std::sync::OnceLock;
    static SYMS: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    SYMS.get_or_init(|| {
        let mut m = HashMap::new();
        // Greek lowercase.
        for (k, v) in [
            ("alpha", "\u{03B1}"), ("beta", "\u{03B2}"), ("gamma", "\u{03B3}"),
            ("delta", "\u{03B4}"), ("epsilon", "\u{03B5}"), ("zeta", "\u{03B6}"),
            ("eta", "\u{03B7}"), ("theta", "\u{03B8}"), ("iota", "\u{03B9}"),
            ("kappa", "\u{03BA}"), ("lambda", "\u{03BB}"), ("mu", "\u{03BC}"),
            ("nu", "\u{03BD}"), ("xi", "\u{03BE}"), ("omicron", "\u{03BF}"),
            ("pi", "\u{03C0}"), ("rho", "\u{03C1}"), ("sigma", "\u{03C3}"),
            ("tau", "\u{03C4}"), ("upsilon", "\u{03C5}"), ("phi", "\u{03C6}"),
            ("chi", "\u{03C7}"), ("psi", "\u{03C8}"), ("omega", "\u{03C9}"),
            ("varepsilon", "\u{03B5}"), ("varphi", "\u{03C6}"),
        ] {
            m.insert(k, v);
        }
        // Greek uppercase.
        for (k, v) in [
            ("Gamma", "\u{0393}"), ("Delta", "\u{0394}"), ("Theta", "\u{0398}"),
            ("Lambda", "\u{039B}"), ("Xi", "\u{039E}"), ("Pi", "\u{03A0}"),
            ("Sigma", "\u{03A3}"), ("Phi", "\u{03A6}"), ("Psi", "\u{03A8}"),
            ("Omega", "\u{03A9}"),
        ] {
            m.insert(k, v);
        }
        // Operators and symbols.
        for (k, v) in [
            ("times", "\u{00D7}"), ("div", "\u{00F7}"), ("pm", "\u{00B1}"),
            ("mp", "\u{2213}"), ("leq", "\u{2264}"), ("geq", "\u{2265}"),
            ("neq", "\u{2260}"), ("approx", "\u{2248}"), ("equiv", "\u{2261}"),
            ("infty", "\u{221E}"), ("cdot", "\u{00B7}"), ("sum", "\u{2211}"),
            ("prod", "\u{220F}"), ("int", "\u{222B}"), ("sqrt", "\u{221A}"),
            ("partial", "\u{2202}"), ("rightarrow", "\u{2192}"), ("to", "\u{2192}"),
            ("leftarrow", "\u{2190}"), ("uparrow", "\u{2191}"), ("downarrow", "\u{2193}"),
            ("in", "\u{2208}"), ("notin", "\u{2209}"), ("subset", "\u{2282}"),
            ("supset", "\u{2283}"), ("subseteq", "\u{2286}"), ("cup", "\u{222A}"),
            ("cap", "\u{2229}"), ("forall", "\u{2200}"), ("exists", "\u{2203}"),
            ("dots", "\u{2026}"), ("ldots", "\u{2026}"), ("langle", "\u{27E8}"),
            ("rangle", "\u{27E9}"), ("propto", "\u{221D}"), ("circ", "\u{2218}"),
            ("prime", "\u{2032}"), ("degree", "\u{00B0}"), ("ell", "\u{2113}"),
            ("hbar", "\u{210F}"), ("nabla", "\u{2207}"), ("aleph", "\u{2135}"),
            ("Re", "\u{211C}"), ("Im", "\u{2111}"),
        ] {
            m.insert(k, v);
        }
        m
    })
}

/// True if the text contains a `$...$` math span.
#[must_use]
pub fn contains_math(text: &str) -> bool {
    text.contains('$')
}

/// Whether a character opens a subscript/superscript group (i.e. `{`).
fn is_group_open(ch: char) -> bool {
    ch == '{'
}

/// Render a math-mode string (no surrounding `$`) to SVG `tspan` runs.
fn render_math(s: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    let syms = symbols();
    while i < chars.len() {
        let ch = chars[i];
        match ch {
            ' ' => {
                out.push(' ');
                i += 1;
            }
            '\\' => {
                // Command or escaped literal.
                let rest: String = chars[i + 1..].iter().collect();
                if let Some(cmd) = rest.split(|c: char| !c.is_ascii_alphabetic()).next() {
                    if cmd.is_empty() {
                        out.push('\\');
                        i += 1;
                        continue;
                    }
                    i += 1 + cmd.len();
                    match cmd {
                        "frac" => {
                            let num = take_group(&chars, &mut i);
                            let den = take_group(&chars, &mut i);
                            out.push_str(&format!(
                                r#"<tspan dy="-0.35em">{}</tspan><tspan dy="0.7em">{}</tspan>"#,
                                render_math(&num),
                                render_math(&den)
                            ));
                        }
                        "sqrt" => {
                            let body = take_group(&chars, &mut i);
                            out.push_str("\u{221A}");
                            out.push_str(&render_math(&body));
                        }
                        "text" => {
                            let body = take_group(&chars, &mut i);
                            out.push_str(&esc(&body));
                        }
                        "hat" => {
                            let body = take_group(&chars, &mut i);
                            out.push_str(&format!(
                                r#"<tspan dy="-0.3em">^</tspan><tspan dy="0.3em">{}</tspan>"#,
                                render_math(&body)
                            ));
                        }
                        _ => {
                            if let Some(g) = syms.get(cmd) {
                                out.push_str(g);
                            } else {
                                // Unknown command: drop the backslash, keep text.
                                out.push_str(&cmd);
                            }
                        }
                    }
                } else {
                    out.push('\\');
                    i += 1;
                }
            }
            '_' | '^' => {
                let (script, consumed) = take_script(&chars, i + 1);
                let shift = if ch == '_' { "sub" } else { "super" };
                out.push_str(&format!(
                    r#"<tspan baseline-shift="{shift}">{}</tspan>"#,
                    render_math(&script)
                ));
                i += 1 + consumed;
            }
            '{' | '}' => {
                i += 1;
            }
            _ => {
                out.push_str(&esc(&ch.to_string()));
                i += 1;
            }
        }
    }
    out
}

/// Consume a braced group `{...}` starting at `chars[i]` (which must be `{`),
/// advancing `i` past the closing brace. Returns the inner text.
fn take_group(chars: &[char], i: &mut usize) -> String {
    if *i >= chars.len() || chars[*i] != '{' {
        return String::new();
    }
    let mut depth = 0usize;
    let mut out = String::new();
    while *i < chars.len() {
        let c = chars[*i];
        if c == '{' {
            depth += 1;
            if depth > 1 {
                out.push(c);
            }
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                *i += 1;
                break;
            }
            out.push(c);
        } else {
            out.push(c);
        }
        *i += 1;
    }
    out
}

/// Consume the next script argument (a single char or `{...}` group) starting
/// at `chars[start]`. Returns `(text, chars_consumed)`.
fn take_script(chars: &[char], start: usize) -> (String, usize) {
    if start >= chars.len() {
        return (String::new(), 0);
    }
    if is_group_open(chars[start]) {
        let mut i = start;
        let text = take_group(chars, &mut i);
        (text, i - start)
    } else {
        (chars[start].to_string(), 1)
    }
}

/// Render arbitrary text: plain runs are XML-escaped, `$...$` runs become
/// math. Plain text without `$` is byte-identical to `xml_escape`.
#[must_use]
pub fn render(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(idx) = rest.find('$') {
        out.push_str(&esc(&rest[..idx]));
        rest = &rest[idx + 1..];
        match rest.find('$') {
            Some(end) => {
                out.push_str(&render_math(&rest[..end]));
                rest = &rest[end + 1..];
            }
            None => {
                // Unbalanced: treat the rest as literal.
                out.push('$');
                out.push_str(&esc(rest));
                rest = "";
            }
        }
    }
    out.push_str(&esc(rest));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_unchanged() {
        assert_eq!(render("Hello world"), "Hello world");
        assert_eq!(render("a < b & c"), "a &lt; b &amp; c");
    }

    #[test]
    fn greek_and_symbols() {
        let out = render(r"$\alpha + \beta \leq \infty$");
        assert!(out.contains("\u{03B1}"));
        assert!(out.contains("\u{03B2}"));
        assert!(out.contains("\u{2264}"));
        assert!(out.contains("\u{221E}"));
    }

    #[test]
    fn super_and_subscript() {
        let out = render(r"$x^2 + y_{ij}$");
        assert!(out.contains(r#"<tspan baseline-shift="super">2</tspan>"#));
        assert!(out.contains(r#"<tspan baseline-shift="sub">ij</tspan>"#));
    }

    #[test]
    fn fraction_and_sqrt() {
        let out = render(r"$\frac{a}{b} + \sqrt{x}$");
        assert!(out.contains("\u{221A}"));
        assert!(out.contains("a"));
        assert!(out.contains("b"));
    }

    #[test]
    fn mixed_plain_and_math() {
        let out = render("y = $x^2$ here");
        assert!(out.starts_with("y = "));
        assert!(out.contains("super"));
        assert!(out.ends_with(" here"));
    }

    #[test]
    fn unbalanced_dollar_is_literal() {
        assert_eq!(render("price $5"), "price $5");
    }

    #[test]
    fn contains_math_flag() {
        assert!(contains_math("$x$"));
        assert!(!contains_math("x"));
    }
}
