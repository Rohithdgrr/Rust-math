//! Automatic statistical annotations for plots.
//!
//! Adds significance brackets, p-values, and statistical test results
//! to box plots, violin plots, bar charts, and other categorical comparisons.
//!
//! # Example
//!
//! ```rust
//! use mathverse_plot::stats_annotate::{StatTest, StatAnnotation};
//!
//! let bracket = StatAnnotation::bracket(
//!     "Control", 0.0,
//!     "Treatment", 1.0,
//!     1.5,
//!     StatTest::TTest { p_value: 0.003 },
//! );
//! ```

use crate::style::Color;

/// Statistical test types with p-values.
#[derive(Debug, Clone)]
pub enum StatTest {
    /// Independent t-test.
    TTest { p_value: f64 },
    /// Paired t-test.
    PairedTTest { p_value: f64 },
    /// Mann-Whitney U test (non-parametric).
    MannWhitney { p_value: f64 },
    /// Wilcoxon signed-rank test.
    Wilcoxon { p_value: f64 },
    /// One-way ANOVA.
    Anova { f_statistic: f64, p_value: f64 },
    /// Kruskal-Wallis test (non-parametric ANOVA).
    KruskalWallis { p_value: f64 },
    /// Chi-squared test.
    ChiSquared { p_value: f64 },
    /// Fisher's exact test.
    FisherExact { p_value: f64 },
    /// Custom test with name and p-value.
    Custom { name: String, p_value: f64 },
}

impl StatTest {
    /// Get the p-value.
    pub fn p_value(&self) -> f64 {
        match self {
            StatTest::TTest { p_value } => *p_value,
            StatTest::PairedTTest { p_value } => *p_value,
            StatTest::MannWhitney { p_value } => *p_value,
            StatTest::Wilcoxon { p_value } => *p_value,
            StatTest::Anova { p_value, .. } => *p_value,
            StatTest::KruskalWallis { p_value } => *p_value,
            StatTest::ChiSquared { p_value } => *p_value,
            StatTest::FisherExact { p_value } => *p_value,
            StatTest::Custom { p_value, .. } => *p_value,
        }
    }

    /// Get the test name.
    pub fn name(&self) -> &str {
        match self {
            StatTest::TTest { .. } => "t-test",
            StatTest::PairedTTest { .. } => "paired t-test",
            StatTest::MannWhitney { .. } => "Mann-Whitney",
            StatTest::Wilcoxon { .. } => "Wilcoxon",
            StatTest::Anova { .. } => "ANOVA",
            StatTest::KruskalWallis { .. } => "Kruskal-Wallis",
            StatTest::ChiSquared { .. } => "chi-squared",
            StatTest::FisherExact { .. } => "Fisher exact",
            StatTest::Custom { name, .. } => name,
        }
    }

    /// Format p-value as significance string.
    pub fn format_p(&self) -> String {
        let p = self.p_value();
        if p < 0.001 {
            format!("{:.2e}", p)
        } else {
            format!("{:.3}", p)
        }
    }

    /// Get significance stars based on alpha thresholds.
    pub fn stars(&self) -> &'static str {
        let p = self.p_value();
        if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "**"
        } else if p < 0.05 {
            "*"
        } else {
            "ns"
        }
    }

    /// Test if significant at given alpha level.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value() < alpha
    }
}

/// A single significance bracket annotation.
#[derive(Debug, Clone)]
pub struct StatAnnotation {
    /// Left group label.
    pub left_label: String,
    /// Left group x position.
    pub left_x: f64,
    /// Right group label.
    pub right_label: String,
    /// Right group x position.
    pub right_x: f64,
    /// Bracket height (y position above data).
    pub bracket_y: f64,
    /// Bracket line color.
    pub color: Color,
    /// Bracket line width.
    pub line_width: f64,
    /// Bracket line style.
    pub line_style: BracketLineStyle,
    /// Statistical test.
    pub test: StatTest,
    /// Text to display (overrides auto-generated).
    pub text_override: Option<String>,
    /// Text font size.
    pub font_size: f64,
    /// Text color.
    pub text_color: Color,
    /// Show p-value (default: true).
    pub show_p: bool,
    /// Show stars (default: true).
    pub show_stars: bool,
    /// Show test name (default: false).
    pub show_test_name: bool,
    /// Significance threshold.
    pub alpha: f64,
}

/// Bracket line styles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BracketLineStyle {
    /// Simple bracket with vertical ticks.
    Bracket,
    /// Horizontal line with vertical ticks.
    Horizontal,
    /// Bracket with rounded corners.
    Rounded,
}

impl Default for StatAnnotation {
    fn default() -> Self {
        Self::new()
    }
}

impl StatAnnotation {
    /// Create a new empty annotation (use builder methods).
    pub fn new() -> Self {
        Self {
            left_label: String::new(),
            left_x: 0.0,
            right_label: String::new(),
            right_x: 1.0,
            bracket_y: 1.0,
            color: Color::BLACK,
            line_width: 1.5,
            line_style: BracketLineStyle::Bracket,
            test: StatTest::TTest { p_value: 1.0 },
            text_override: None,
            font_size: 10.0,
            text_color: Color::BLACK,
            show_p: true,
            show_stars: true,
            show_test_name: false,
            alpha: 0.05,
        }
    }

    /// Create a bracket between two groups.
    pub fn bracket(
        left_label: impl Into<String>,
        left_x: f64,
        right_label: impl Into<String>,
        right_x: f64,
        bracket_y: f64,
        test: StatTest,
    ) -> Self {
        Self {
            left_label: left_label.into(),
            left_x,
            right_label: right_label.into(),
            right_x,
            bracket_y,
            test,
            ..Self::new()
        }
    }

    /// Set bracket color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set line width.
    pub fn with_line_width(mut self, width: f64) -> Self {
        self.line_width = width;
        self
    }

    /// Set line style.
    pub fn with_line_style(mut self, style: BracketLineStyle) -> Self {
        self.line_style = style;
        self
    }

    /// Set font size.
    pub fn with_font_size(mut self, size: f64) -> Self {
        self.font_size = size;
        self
    }

    /// Set text color.
    pub fn with_text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }

    /// Set significance threshold.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Show only stars (no p-value text).
    pub fn stars_only(mut self) -> Self {
        self.show_p = false;
        self.show_stars = true;
        self.show_test_name = false;
        self
    }

    /// Show only p-value (no stars).
    pub fn p_only(mut self) -> Self {
        self.show_p = true;
        self.show_stars = false;
        self.show_test_name = false;
        self
    }

    /// Show test name + p-value.
    pub fn with_test_name(mut self) -> Self {
        self.show_test_name = true;
        self
    }

    /// Override the display text.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text_override = Some(text.into());
        self
    }

    /// Get the display text for this annotation.
    pub fn display_text(&self) -> String {
        if let Some(ref text) = self.text_override {
            return text.clone();
        }

        let mut parts = Vec::new();

        if self.show_test_name {
            parts.push(self.test.name().to_string());
        }

        if self.show_stars {
            parts.push(self.test.stars().to_string());
        }

        if self.show_p {
            parts.push(format!("p={}", self.test.format_p()));
        }

        if parts.is_empty() {
            self.test.stars().to_string()
        } else {
            parts.join(", ")
        }
    }

    /// Render the bracket as SVG.
    pub fn render_svg(&self, plot_top: f64, y_scale: f64) -> String {
        let mut svg = String::new();
        let x1 = self.left_x;
        let x2 = self.right_x;
        let y = self.bracket_y;
        let tick_len = 0.05 * (x2 - x1).abs().max(1.0);
        let color_hex = self.color.to_hex();

        match self.line_style {
            BracketLineStyle::Bracket => {
                // Horizontal line
                svg.push_str(&format!(
                    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>\n",
                    x1, y, x2, y, color_hex, self.line_width
                ));
                // Left tick
                svg.push_str(&format!(
                    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>\n",
                    x1, y, x1, y - tick_len, color_hex, self.line_width
                ));
                // Right tick
                svg.push_str(&format!(
                    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>\n",
                    x2, y, x2, y - tick_len, color_hex, self.line_width
                ));
            }
            BracketLineStyle::Horizontal => {
                // Just a horizontal line with small end caps
                svg.push_str(&format!(
                    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>\n",
                    x1 + tick_len, y, x2 - tick_len, y, color_hex, self.line_width
                ));
                svg.push_str(&format!(
                    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>\n",
                    x1, y - tick_len / 2.0, x1, y + tick_len / 2.0, color_hex, self.line_width
                ));
                svg.push_str(&format!(
                    "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>\n",
                    x2, y - tick_len / 2.0, x2, y + tick_len / 2.0, color_hex, self.line_width
                ));
            }
            BracketLineStyle::Rounded => {
                // Rounded bracket using path
                let mid_x = (x1 + x2) / 2.0;
                svg.push_str(&format!(
                    "  <path d=\"M{},{} L{},{} L{},{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n",
                    x1, y - tick_len, x1, y, mid_x, y, color_hex, self.line_width
                ));
                svg.push_str(&format!(
                    "  <path d=\"M{},{} L{},{} L{},{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n",
                    mid_x, y, x2, y, x2, y - tick_len, color_hex, self.line_width
                ));
            }
        }

        // Text label
        let text = self.display_text();
        let text_x = (x1 + x2) / 2.0;
        let text_y = y - tick_len - 2.0;
        svg.push_str(&format!(
            "  <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"{}\" fill=\"{}\">{}</text>\n",
            text_x, text_y, self.font_size, self.text_color.to_hex(), text
        ));

        svg
    }
}

/// A collection of statistical annotations for a plot.
#[derive(Debug, Clone)]
pub struct StatAnnotations {
    /// List of annotations.
    pub annotations: Vec<StatAnnotation>,
    /// Global significance threshold.
    pub alpha: f64,
    /// Global font size.
    pub font_size: f64,
    /// Global text color.
    pub text_color: Color,
}

impl Default for StatAnnotations {
    fn default() -> Self {
        Self {
            annotations: Vec::new(),
            alpha: 0.05,
            font_size: 10.0,
            text_color: Color::BLACK,
        }
    }
}

impl StatAnnotations {
    /// Create a new empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the significance threshold for all annotations.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Add a bracket annotation.
    pub fn add_bracket(mut self, annotation: StatAnnotation) -> Self {
        self.annotations.push(annotation);
        self
    }

    /// Add a bracket between two groups.
    pub fn bracket(
        mut self,
        left_label: impl Into<String>,
        left_x: f64,
        right_label: impl Into<String>,
        right_x: f64,
        bracket_y: f64,
        test: StatTest,
    ) -> Self {
        self.annotations.push(StatAnnotation::bracket(
            left_label, left_x,
            right_label, right_x,
            bracket_y,
            test,
        ));
        self
    }

    /// Auto-arrange brackets vertically to avoid overlap.
    pub fn auto_arrange(&mut self, min_spacing: f64) {
        // Sort by x-span (wider brackets go lower)
        let mut indexed: Vec<(usize, f64)> = self.annotations
            .iter()
            .enumerate()
            .map(|(i, a)| (i, (a.right_x - a.left_x).abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign y positions from top
        let mut current_y = self.annotations.iter().map(|a| a.bracket_y).fold(f64::INFINITY, f64::min);
        for (idx, _) in &indexed {
            self.annotations[*idx].bracket_y = current_y;
            current_y -= min_spacing;
        }
    }

    /// Render all annotations as SVG.
    pub fn render_svg(&self, plot_top: f64, y_scale: f64) -> String {
        let mut svg = String::new();
        for annotation in &self.annotations {
            svg.push_str(&annotation.render_svg(plot_top, y_scale));
        }
        svg
    }

    /// Get count of annotations.
    pub fn len(&self) -> usize {
        self.annotations.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.annotations.is_empty()
    }

    /// Filter annotations by significance.
    pub fn significant_only(&self) -> Vec<&StatAnnotation> {
        self.annotations
            .iter()
            .filter(|a| a.test.is_significant(self.alpha))
            .collect()
    }
}

/// Format a p-value for display (convenience function).
pub fn format_pvalue(p: f64) -> String {
    if p < 0.001 {
        format!("{:.2e}", p)
    } else if p < 0.01 {
        format!("{:.3}", p)
    } else {
        format!("{:.3}", p)
    }
}

/// Get significance stars for a p-value (convenience function).
pub fn significance_stars(p: f64) -> &'static str {
    if p < 0.001 {
        "***"
    } else if p < 0.01 {
        "**"
    } else if p < 0.05 {
        "*"
    } else {
        "ns"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_test_p_value() {
        let test = StatTest::TTest { p_value: 0.03 };
        assert_eq!(test.p_value(), 0.03);
        assert_eq!(test.stars(), "*");
        assert!(test.is_significant(0.05));
        assert!(!test.is_significant(0.01));
    }

    #[test]
    fn stat_test_format_p() {
        let test = StatTest::TTest { p_value: 0.00012 };
        assert_eq!(test.format_p(), "1.20e-4");

        let test2 = StatTest::TTest { p_value: 0.042 };
        assert_eq!(test2.format_p(), "0.042");
    }

    #[test]
    fn stat_test_stars() {
        assert_eq!(StatTest::TTest { p_value: 0.0005 }.stars(), "***");
        assert_eq!(StatTest::TTest { p_value: 0.005 }.stars(), "**");
        assert_eq!(StatTest::TTest { p_value: 0.03 }.stars(), "*");
        assert_eq!(StatTest::TTest { p_value: 0.1 }.stars(), "ns");
    }

    #[test]
    fn stat_test_name() {
        assert_eq!(StatTest::TTest { p_value: 0.0 }.name(), "t-test");
        assert_eq!(StatTest::MannWhitney { p_value: 0.0 }.name(), "Mann-Whitney");
        assert_eq!(StatTest::Anova { f_statistic: 1.0, p_value: 0.0 }.name(), "ANOVA");
    }

    #[test]
    fn bracket_render_svg() {
        let bracket = StatAnnotation::bracket(
            "A", 0.0,
            "B", 1.0,
            2.0,
            StatTest::TTest { p_value: 0.003 },
        );
        let svg = bracket.render_svg(0.0, 1.0);
        assert!(svg.contains("<line"));
        assert!(svg.contains("<text"));
        assert!(svg.contains("p=0.003"));
    }

    #[test]
    fn bracket_display_text() {
        let b = StatAnnotation::bracket(
            "A", 0.0, "B", 1.0, 2.0,
            StatTest::TTest { p_value: 0.03 },
        ).stars_only();
        assert_eq!(b.display_text(), "*");

        let b2 = StatAnnotation::bracket(
            "A", 0.0, "B", 1.0, 2.0,
            StatTest::TTest { p_value: 0.03 },
        ).p_only();
        assert_eq!(b2.display_text(), "p=0.030");

        let b3 = StatAnnotation::bracket(
            "A", 0.0, "B", 1.0, 2.0,
            StatTest::TTest { p_value: 0.03 },
        ).with_test_name();
        assert!(b3.display_text().contains("t-test"));
    }

    #[test]
    fn annotations_collection() {
        let ann = StatAnnotations::new()
            .with_alpha(0.05)
            .bracket("A", 0.0, "B", 1.0, 3.0, StatTest::TTest { p_value: 0.01 })
            .bracket("B", 1.0, "C", 2.0, 3.5, StatTest::TTest { p_value: 0.1 });

        assert_eq!(ann.len(), 2);
        assert_eq!(ann.significant_only().len(), 1);
    }

    #[test]
    fn auto_arrange_spacing() {
        let mut ann = StatAnnotations::new()
            .bracket("A", 0.0, "C", 2.0, 3.0, StatTest::TTest { p_value: 0.01 })
            .bracket("A", 0.0, "B", 1.0, 3.0, StatTest::TTest { p_value: 0.01 });
        ann.auto_arrange(0.5);
        // After arrange, y positions should differ
        let y0 = ann.annotations[0].bracket_y;
        let y1 = ann.annotations[1].bracket_y;
        assert!((y0 - y1).abs() >= 0.4);
    }

    #[test]
    fn format_pvalue_convenience() {
        assert_eq!(format_pvalue(0.0001), "1.00e-4");
        assert_eq!(format_pvalue(0.05), "0.050");
    }

    #[test]
    fn significance_stars_convenience() {
        assert_eq!(significance_stars(0.0005), "***");
        assert_eq!(significance_stars(0.04), "*");
        assert_eq!(significance_stars(0.2), "ns");
    }
}
