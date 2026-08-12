//! Font management and text layout — the analogue of matplotlib's font
//! properties (`font.family`, `font.size`) and text wrapping/alignment.
//!
//! The SVG backend emits text directly; this module provides the selection
//! (`FontConfig`) and layout (word wrap, alignment, multi-line emission)
//! helpers so labels behave like matplotlib's `Text` objects.

use crate::style::Color;

/// Font selection for a plot, mirroring matplotlib's `rcParams` font keys.
#[derive(Debug, Clone, PartialEq)]
pub struct FontConfig {
    /// CSS font-family stack (e.g. `"DejaVu Sans", sans-serif`).
    pub family: String,
    /// Base size in px.
    pub size: f64,
    /// Title size in px (falls back to `size * 1.5`).
    pub title_size: f64,
    /// Axis label size in px (falls back to `size`).
    pub label_size: f64,
    /// Tick label size in px (falls back to `size * 0.8`).
    pub tick_size: f64,
    /// Bold for titles.
    pub bold: bool,
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "Arial, sans-serif".to_string(),
            size: 14.0,
            title_size: 20.0,
            label_size: 14.0,
            tick_size: 11.0,
            bold: false,
        }
    }
}

impl FontConfig {
    /// Create with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the font-family stack.
    #[must_use]
    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = family.into();
        self
    }

    /// Set the base size in px.
    #[must_use]
    pub fn with_size(mut self, size: f64) -> Self {
        self.size = size;
        self.title_size = size * 1.5;
        self.label_size = size;
        self.tick_size = size * 0.8;
        self
    }

    /// Set sizes individually.
    #[must_use]
    pub fn with_sizes(mut self, title: f64, label: f64, tick: f64) -> Self {
        self.title_size = title;
        self.label_size = label;
        self.tick_size = tick;
        self
    }

    /// Set bold.
    #[must_use]
    pub fn with_bold(mut self, bold: bool) -> Self {
        self.bold = bold;
        self
    }
}

/// Horizontal alignment of laid-out text lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    /// Left-aligned (SVG `text-anchor: start`).
    Left,
    /// Centered (SVG `text-anchor: middle`).
    Center,
    /// Right-aligned (SVG `text-anchor: end`).
    Right,
}

impl TextAlign {
    /// SVG `text-anchor` value.
    #[must_use]
    pub fn anchor(&self) -> &'static str {
        match self {
            TextAlign::Left => "start",
            TextAlign::Center => "middle",
            TextAlign::Right => "end",
        }
    }
}

/// Wrap `text` into lines of at most `max_chars` characters, breaking on
/// spaces. Long words are hard-broken so no line exceeds the limit. This is a
/// character-count heuristic (matplotlib's `wrap` does the same).
#[must_use]
pub fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    if max_chars == 0 {
        return text.lines().map(ToString::to_string).collect();
    }
    let mut out = Vec::new();
    for line in text.lines() {
        if line.chars().count() <= max_chars {
            out.push(line.to_string());
            continue;
        }
        let mut current = String::new();
        let mut current_len = 0usize;
        for word in line.split_whitespace() {
            let wlen = word.chars().count();
            if current_len > 0 && current_len + 1 + wlen > max_chars {
                out.push(std::mem::take(&mut current));
                current_len = 0;
            }
            // A single word longer than the limit gets hard-broken.
            if wlen > max_chars {
                let mut rest = word.to_string();
                while rest.chars().count() > max_chars {
                    let take: String = rest.chars().take(max_chars).collect();
                    out.push(take);
                    rest = rest.chars().skip(max_chars).collect();
                }
                if !rest.is_empty() {
                    if !current.is_empty() {
                        out.push(std::mem::take(&mut current));
                    }
                    current = rest.clone();
                    current_len = rest.chars().count();
                }
                continue;
            }
            if current_len > 0 {
                current.push(' ');
                current_len += 1;
            }
            current.push_str(word);
            current_len += wlen;
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

/// Emit multi-line SVG `<text>`/`<tspan>` elements for `text` laid out at
/// `(x, y)` with the given alignment. Newlines and over-length lines become
/// separate `<tspan>` runs (line height `line_height`).
///
/// The returned string is already XML-escaped and math-rendered.
#[must_use]
pub fn multiline_svg_text(
    text: &str,
    x: f64,
    y: f64,
    max_chars: usize,
    align: TextAlign,
    font_family: &str,
    font_size: f64,
    fill: Color,
    line_height: f64,
) -> String {
    let lines = wrap_text(text, max_chars);
    let anchor = align.anchor();
    let mut out = String::new();
    let mut dy = y;
    for (i, line) in lines.iter().enumerate() {
        let rendered = crate::mathtext::render(line);
        let esc = crate::common::xml_escape(&rendered);
        let fill_hex = fill.to_hex();
        if i == 0 {
            out.push_str(&format!(
                r#"<text x="{x:.1}" y="{dy:.1}" text-anchor="{anchor}" font-family="{font_family}" font-size="{font_size:.1}" fill="{fill_hex}">{esc}</text>"#
            ));
        } else {
            out.push_str(&format!(
                r#"<tspan x="{x:.1}" dy="{dy_off:.1}" text-anchor="{anchor}">{esc}</tspan>"#,
                dy_off = line_height
            ));
        }
        dy += line_height;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_short_text_unchanged() {
        assert_eq!(wrap_text("hello", 80), vec!["hello"]);
    }

    #[test]
    fn wrap_breaks_on_spaces() {
        assert_eq!(wrap_text("a b c d", 3), vec!["a b", "c d"]);
    }

    #[test]
    fn wrap_hard_breaks_long_words() {
        let lines = wrap_text("supercalifragilistic", 6);
        assert!(lines.iter().all(|l| l.chars().count() <= 6));
        assert!(lines.len() >= 3);
    }

    #[test]
    fn wrap_respects_explicit_newlines() {
        assert_eq!(wrap_text("ab\ncd ef", 10), vec!["ab", "cd ef"]);
    }

    #[test]
    fn align_anchors() {
        assert_eq!(TextAlign::Left.anchor(), "start");
        assert_eq!(TextAlign::Center.anchor(), "middle");
        assert_eq!(TextAlign::Right.anchor(), "end");
    }

    #[test]
    fn font_config_sizes() {
        let f = FontConfig::new().with_size(12.0);
        assert!((f.title_size - 18.0).abs() < 1e-9);
        assert!((f.label_size - 12.0).abs() < 1e-9);
        assert!((f.tick_size - 9.6).abs() < 1e-9);
    }

    #[test]
    fn multiline_emits_tspan() {
        let out = multiline_svg_text(
            "first line\nsecond line",
            10.0,
            20.0,
            100,
            TextAlign::Center,
            "Arial",
            14.0,
            Color::BLACK,
            16.0,
        );
        assert!(out.contains("<text"));
        assert!(out.contains("<tspan"));
        assert!(out.contains("text-anchor=\"middle\""));
    }
}
