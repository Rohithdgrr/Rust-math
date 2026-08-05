//! Theme system for plot customization.

use crate::style::Color;

/// Predefined plot themes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Theme {
    /// Clean minimal style.
    Minimal,
    /// Seaborn-style (whitegrid).
    Seaborn,
    /// ggplot2-style.
    Ggplot,
    /// Dark background.
    Dark,
    /// High contrast for accessibility.
    HighContrast,
    /// Classic matplotlib style.
    Classic,
    /// Academic paper style.
    Academic,
    /// Presentation style (larger fonts).
    Presentation,
    /// Blueprint style.
    Blueprint,
    /// Matplotlib default (v3.x / v2.x) — replicates the classic default style:
    /// grey grid, white axes, blue(ish) tick labels, default tab10 colors.
    Matplotlib,
}

impl Default for Theme {
    fn default() -> Self {
        Self::Minimal
    }
}

/// Complete theme configuration.
#[derive(Debug, Clone)]
pub struct ThemeConfig {
    /// Theme preset.
    pub theme: Theme,
    /// Background color.
    pub background_color: Color,
    /// Plot area background.
    pub plot_background: Color,
    /// Text color.
    pub text_color: Color,
    /// Axis color.
    pub axis_color: Color,
    /// Grid color.
    pub grid_color: Color,
    /// Grid line style.
    pub grid_style: LineStyle,
    /// Font family.
    pub font_family: String,
    /// Title font size.
    pub title_size: f64,
    /// Axis label font size.
    pub label_size: f64,
    /// Tick label font size.
    pub tick_size: f64,
    /// Legend font size.
    pub legend_size: f64,
    /// Line width.
    pub line_width: f64,
    /// Border width.
    pub border_width: f64,
    /// Show grid.
    pub show_grid: bool,
    /// Show axis border.
    pub show_border: bool,
    /// Show tick marks.
    pub show_ticks: bool,
    /// Show spine.
    pub show_spine: SpineVisibility,
    /// Color palette.
    pub palette: ColorPalette,
    /// Figure face color.
    pub figure_facecolor: Color,
    /// Axes face color.
    pub axes_facecolor: Color,
    /// Axes edge color.
    pub axes_edgecolor: Color,
    /// X-axis grid.
    pub xgrid: bool,
    /// Y-axis grid.
    pub ygrid: bool,
    /// Grid axis.
    pub grid_axis: GridAxis,
}

/// Line style options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

impl LineStyle {
    /// Get SVG stroke-dasharray value.
    pub fn to_svg(&self) -> &str {
        match self {
            LineStyle::Solid => "none",
            LineStyle::Dashed => "6,4",
            LineStyle::Dotted => "2,2",
            LineStyle::DashDot => "6,2,2,2",
        }
    }
}

/// Spine visibility options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpineVisibility {
    pub top: bool,
    pub right: bool,
    pub bottom: bool,
    pub left: bool,
}

impl SpineVisibility {
    pub fn all() -> Self {
        Self { top: true, right: true, bottom: true, left: true }
    }

    pub fn none() -> Self {
        Self { top: false, right: false, bottom: false, left: false }
    }

    pub fn bottom_left() -> Self {
        Self { top: false, right: false, bottom: true, left: true }
    }
}

/// Grid axis selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridAxis {
    Both,
    X,
    Y,
    None,
}

/// Color palette for plots.
#[derive(Debug, Clone)]
pub struct ColorPalette {
    /// Primary colors.
    pub colors: Vec<Color>,
    /// Name of the palette.
    pub name: String,
}

impl ColorPalette {
    /// Create a new palette.
    pub fn new(name: impl Into<String>, colors: Vec<Color>) -> Self {
        Self { name: name.into(), colors }
    }

    /// Get color by index (cycles).
    pub fn get(&self, index: usize) -> Color {
        self.colors[index % self.colors.len()]
    }
}

/// Built-in palettes.
impl ColorPalette {
    /// Default matplotlib palette.
    pub fn default() -> Self {
        Self::new("default", vec![
            Color::BLUE, Color::ORANGE, Color::GREEN, Color::RED,
            Color::PURPLE, Color::BROWN, Color::Named("pink"), Color::GRAY,
        ])
    }

    /// Seaborn deep palette.
    pub fn seaborn_deep() -> Self {
        Self::new("seaborn_deep", vec![
            Color::rgb(0, 114, 178),   // Blue
            Color::rgb(230, 159, 0),    // Orange
            Color::rgb(86, 180, 233),   // Sky blue
            Color::rgb(0, 158, 115),    // Bluish green
            Color::rgb(240, 228, 66),   // Yellow
            Color::rgb(213, 94, 0),     // Vermillion
            Color::rgb(204, 121, 167),  // Reddish purple
        ])
    }

    /// Viridis palette.
    pub fn viridis() -> Self {
        Self::new("viridis", vec![
            Color::rgb(68, 1, 84),
            Color::rgb(72, 35, 116),
            Color::rgb(64, 67, 135),
            Color::rgb(52, 94, 141),
            Color::rgb(33, 145, 140),
            Color::rgb(94, 201, 98),
            Color::rgb(253, 231, 37),
        ])
    }

    /// Pastel palette.
    pub fn pastel() -> Self {
        Self::new("pastel", vec![
            Color::rgb(179, 205, 227),
            Color::rgb(253, 205, 227),
            Color::rgb(203, 213, 232),
            Color::rgb(179, 222, 105),
            Color::rgb(252, 205, 229),
            Color::rgb(217, 217, 217),
            Color::rgb(255, 255, 178),
        ])
    }

    /// Set1 (qualitative).
    pub fn set1() -> Self {
        Self::new("set1", vec![
            Color::rgb(228, 26, 28),
            Color::rgb(55, 126, 184),
            Color::rgb(77, 175, 74),
            Color::rgb(255, 127, 0),
            Color::rgb(152, 78, 163),
            Color::rgb(255, 255, 51),
            Color::rgb(166, 86, 40),
            Color::rgb(247, 129, 191),
        ])
    }

    /// Dark2 (qualitative).
    pub fn dark2() -> Self {
        Self::new("dark2", vec![
            Color::rgb(27, 158, 119),
            Color::rgb(217, 95, 2),
            Color::rgb(117, 112, 179),
            Color::rgb(231, 41, 138),
            Color::rgb(102, 166, 30),
            Color::rgb(230, 171, 2),
            Color::rgb(166, 118, 29),
            Color::rgb(102, 102, 102),
        ])
    }

    /// Colorblind-friendly palette.
    pub fn colorblind() -> Self {
        Self::new("colorblind", vec![
            Color::rgb(0, 114, 178),
            Color::rgb(230, 159, 0),
            Color::rgb(86, 180, 233),
            Color::rgb(0, 158, 115),
            Color::rgb(240, 228, 66),
            Color::rgb(213, 94, 0),
        ])
    }
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self::seaborn_deep()
    }
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self::new(Theme::Minimal)
    }
}

impl ThemeConfig {
    /// Create a new theme config.
    pub fn new(theme: Theme) -> Self {
        match theme {
            Theme::Minimal => Self::minimal(),
            Theme::Seaborn => Self::seaborn(),
            Theme::Ggplot => Self::ggplot(),
            Theme::Dark => Self::dark(),
            Theme::HighContrast => Self::high_contrast(),
            Theme::Classic => Self::classic(),
            Theme::Academic => Self::academic(),
            Theme::Presentation => Self::presentation(),
        Theme::Blueprint => Self::blueprint(),
        Theme::Matplotlib => Self::matplotlib(),
    }
    }

    /// Minimal theme.
    pub fn minimal() -> Self {
        Self {
            theme: Theme::Minimal,
            background_color: Color::WHITE,
            plot_background: Color::WHITE,
            text_color: Color::BLACK,
            axis_color: Color::rgb(128, 128, 128),
            grid_color: Color::rgb(230, 230, 230),
            grid_style: LineStyle::Solid,
            font_family: "Arial, sans-serif".into(),
            title_size: 16.0,
            label_size: 12.0,
            tick_size: 10.0,
            legend_size: 11.0,
            line_width: 1.5,
            border_width: 1.0,
            show_grid: false,
            show_border: true,
            show_ticks: true,
            show_spine: SpineVisibility::bottom_left(),
            palette: ColorPalette::seaborn_deep(),
            figure_facecolor: Color::WHITE,
            axes_facecolor: Color::WHITE,
            axes_edgecolor: Color::BLACK,
            xgrid: false,
            ygrid: false,
            grid_axis: GridAxis::None,
        }
    }

    /// Seaborn whitegrid theme.
    pub fn seaborn() -> Self {
        Self {
            theme: Theme::Seaborn,
            background_color: Color::WHITE,
            plot_background: Color::WHITE,
            text_color: Color::rgb(50, 50, 50),
            axis_color: Color::rgb(180, 180, 180),
            grid_color: Color::rgb(220, 220, 220),
            grid_style: LineStyle::Solid,
            font_family: "Helvetica, Arial, sans-serif".into(),
            title_size: 16.0,
            label_size: 12.0,
            tick_size: 10.0,
            legend_size: 11.0,
            line_width: 1.5,
            border_width: 1.0,
            show_grid: true,
            show_border: true,
            show_ticks: true,
            show_spine: SpineVisibility::all(),
            palette: ColorPalette::seaborn_deep(),
            figure_facecolor: Color::WHITE,
            axes_facecolor: Color::WHITE,
            axes_edgecolor: Color::rgb(180, 180, 180),
            xgrid: false,
            ygrid: true,
            grid_axis: GridAxis::Y,
        }
    }

    /// ggplot2 theme.
    pub fn ggplot() -> Self {
        Self {
            theme: Theme::Ggplot,
            background_color: Color::rgb(229, 229, 229),
            plot_background: Color::WHITE,
            text_color: Color::BLACK,
            axis_color: Color::rgb(128, 128, 128),
            grid_color: Color::WHITE,
            grid_style: LineStyle::Solid,
            font_family: "Arial, sans-serif".into(),
            title_size: 16.0,
            label_size: 12.0,
            tick_size: 10.0,
            legend_size: 11.0,
            line_width: 1.0,
            border_width: 0.5,
            show_grid: true,
            show_border: false,
            show_ticks: false,
            show_spine: SpineVisibility::none(),
            palette: ColorPalette::default(),
            figure_facecolor: Color::rgb(229, 229, 229),
            axes_facecolor: Color::WHITE,
            axes_edgecolor: Color::BLACK,
            xgrid: true,
            ygrid: true,
            grid_axis: GridAxis::Both,
        }
    }

    /// Dark theme.
    pub fn dark() -> Self {
        Self {
            theme: Theme::Dark,
            background_color: Color::rgb(30, 30, 30),
            plot_background: Color::rgb(40, 40, 40),
            text_color: Color::rgb(220, 220, 220),
            axis_color: Color::rgb(100, 100, 100),
            grid_color: Color::rgb(60, 60, 60),
            grid_style: LineStyle::Solid,
            font_family: "Consolas, monospace".into(),
            title_size: 16.0,
            label_size: 12.0,
            tick_size: 10.0,
            legend_size: 11.0,
            line_width: 1.5,
            border_width: 1.0,
            show_grid: true,
            show_border: true,
            show_ticks: true,
            show_spine: SpineVisibility::all(),
            palette: ColorPalette::new("dark", vec![
                Color::rgb(114, 158, 206),
                Color::rgb(255, 127, 14),
                Color::rgb(103, 191, 92),
                Color::rgb(227, 119, 194),
                Color::rgb(255, 215, 0),
                Color::rgb(148, 103, 189),
                Color::rgb(140, 86, 75),
            ]),
            figure_facecolor: Color::rgb(30, 30, 30),
            axes_facecolor: Color::rgb(40, 40, 40),
            axes_edgecolor: Color::rgb(100, 100, 100),
            xgrid: true,
            ygrid: true,
            grid_axis: GridAxis::Both,
        }
    }

    /// High contrast theme.
    pub fn high_contrast() -> Self {
        Self {
            theme: Theme::HighContrast,
            background_color: Color::WHITE,
            plot_background: Color::WHITE,
            text_color: Color::BLACK,
            axis_color: Color::BLACK,
            grid_color: Color::rgb(200, 200, 200),
            grid_style: LineStyle::Dashed,
            font_family: "Arial, sans-serif".into(),
            title_size: 18.0,
            label_size: 14.0,
            tick_size: 12.0,
            legend_size: 12.0,
            line_width: 2.5,
            border_width: 2.0,
            show_grid: true,
            show_border: true,
            show_ticks: true,
            show_spine: SpineVisibility::all(),
            palette: ColorPalette::new("high_contrast", vec![
                Color::BLACK,
                Color::rgb(200, 0, 0),
                Color::rgb(0, 100, 0),
                Color::rgb(0, 0, 200),
                Color::rgb(200, 150, 0),
            ]),
            figure_facecolor: Color::WHITE,
            axes_facecolor: Color::WHITE,
            axes_edgecolor: Color::BLACK,
            xgrid: false,
            ygrid: true,
            grid_axis: GridAxis::Y,
        }
    }

    /// Classic matplotlib theme.
    pub fn classic() -> Self {
        Self {
            theme: Theme::Classic,
            background_color: Color::WHITE,
            plot_background: Color::WHITE,
            text_color: Color::BLACK,
            axis_color: Color::BLACK,
            grid_color: Color::rgb(200, 200, 200),
            grid_style: LineStyle::Dashed,
            font_family: "serif".into(),
            title_size: 16.0,
            label_size: 12.0,
            tick_size: 10.0,
            legend_size: 11.0,
            line_width: 1.5,
            border_width: 1.0,
            show_grid: false,
            show_border: true,
            show_ticks: true,
            show_spine: SpineVisibility::all(),
            palette: ColorPalette::default(),
            figure_facecolor: Color::WHITE,
            axes_facecolor: Color::WHITE,
            axes_edgecolor: Color::BLACK,
            xgrid: false,
            ygrid: false,
            grid_axis: GridAxis::None,
        }
    }

    /// Academic paper theme.
    pub fn academic() -> Self {
        Self {
            theme: Theme::Academic,
            background_color: Color::WHITE,
            plot_background: Color::WHITE,
            text_color: Color::BLACK,
            axis_color: Color::BLACK,
            grid_color: Color::rgb(220, 220, 220),
            grid_style: LineStyle::Solid,
            font_family: "Times New Roman, serif".into(),
            title_size: 14.0,
            label_size: 12.0,
            tick_size: 10.0,
            legend_size: 10.0,
            line_width: 1.0,
            border_width: 1.0,
            show_grid: false,
            show_border: true,
            show_ticks: true,
            show_spine: SpineVisibility::bottom_left(),
            palette: ColorPalette::new("academic", vec![
                Color::BLACK,
                Color::rgb(100, 100, 100),
                Color::rgb(150, 150, 150),
            ]),
            figure_facecolor: Color::WHITE,
            axes_facecolor: Color::WHITE,
            axes_edgecolor: Color::BLACK,
            xgrid: false,
            ygrid: false,
            grid_axis: GridAxis::None,
        }
    }

    /// Presentation theme (larger fonts).
    pub fn presentation() -> Self {
        Self {
            theme: Theme::Presentation,
            background_color: Color::WHITE,
            plot_background: Color::WHITE,
            text_color: Color::BLACK,
            axis_color: Color::rgb(100, 100, 100),
            grid_color: Color::rgb(220, 220, 220),
            grid_style: LineStyle::Solid,
            font_family: "Arial, sans-serif".into(),
            title_size: 24.0,
            label_size: 18.0,
            tick_size: 14.0,
            legend_size: 16.0,
            line_width: 2.5,
            border_width: 2.0,
            show_grid: true,
            show_border: true,
            show_ticks: true,
            show_spine: SpineVisibility::all(),
            palette: ColorPalette::seaborn_deep(),
            figure_facecolor: Color::WHITE,
            axes_facecolor: Color::WHITE,
            axes_edgecolor: Color::rgb(100, 100, 100),
            xgrid: false,
            ygrid: true,
            grid_axis: GridAxis::Y,
        }
    }

    /// Blueprint theme.
    pub fn blueprint() -> Self {
        Self {
        theme: Theme::Blueprint,
        background_color: Color::rgb(0, 40, 80),
        plot_background: Color::rgb(0, 50, 100),
        text_color: Color::rgb(200, 220, 255),
        axis_color: Color::rgb(100, 150, 200),
        grid_color: Color::rgb(50, 100, 150),
        grid_style: LineStyle::Solid,
        font_family: "Consolas, monospace".into(),
        title_size: 16.0,
        label_size: 12.0,
        tick_size: 10.0,
        legend_size: 11.0,
        line_width: 2.0,
        border_width: 1.0,
        show_grid: true,
        show_border: true,
        show_ticks: true,
        show_spine: SpineVisibility::all(),
        palette: ColorPalette::new("blueprint", vec![
            Color::rgb(100, 200, 255),
            Color::rgb(255, 200, 100),
            Color::rgb(100, 255, 150),
            Color::rgb(255, 150, 200),
        ]),
        figure_facecolor: Color::rgb(0, 40, 80),
        axes_facecolor: Color::rgb(0, 50, 100),
        axes_edgecolor: Color::rgb(100, 150, 200),
        xgrid: true,
        ygrid: true,
        grid_axis: GridAxis::Both,
    }
}

/// Matplotlib default style.
///
/// Closely mirrors `matplotlib.pyplot.rcParams` from matplotlib ≤ 3.x:
/// - tweak the default colormap away from pure `#1f77b4` blue
/// - set a slightly darker grid so it reads on white
/// - `seaborn-deep`-style qualitative palette (same 8-color cycle as the default
///   `tab10` table, which matplotlib ships from 2.0 onward)
pub fn matplotlib() -> Self {
    Self {
        theme: Theme::Matplotlib,
        background_color: Color::WHITE,
        plot_background: Color::WHITE,
        text_color: Color::rgb(60, 60, 60),
        axis_color: Color::rgb(120, 120, 120),
        grid_color: Color::rgb(210, 210, 210),
        grid_style: LineStyle::Solid,
        font_family: "DejaVu Sans, Arial, sans-serif".into(),
        title_size: 16.0,
        label_size: 13.0,
        tick_size: 10.0,
        legend_size: 10.5,
        line_width: 1.5,
        border_width: 0.8,
        show_grid: true,
        show_border: true,
        show_ticks: true,
        show_spine: SpineVisibility::all(),
        palette: ColorPalette::new(
            "matplotlib",
            vec![
                // tab10 — the default matplotlib qualitative colors
                Color::rgb(31, 119, 180),
                Color::rgb(255, 127, 14),
                Color::rgb(44, 160, 44),
                Color::rgb(214, 39, 40),
                Color::rgb(148, 103, 189),
                Color::rgb(140, 86, 75),
                Color::rgb(227, 119, 194),
                Color::rgb(127, 127, 127),
                Color::rgb(188, 189, 34),
                Color::rgb(23, 190, 207),
            ],
        ),
        figure_facecolor: Color::WHITE,
        axes_facecolor: Color::WHITE,
        axes_edgecolor: Color::rgb(120, 120, 120),
        xgrid: false,
        ygrid: true,
        grid_axis: GridAxis::Y,
    }
}

    /// Apply theme to SVG header.
    pub fn svg_header(&self, _width: u32, _height: u32) -> String {
        let mut svg = String::new();

        // Style definitions
        svg.push_str("<defs>\n");
        svg.push_str("  <style>\n");

        // Font
        svg.push_str("    text { font-family: ");
        svg.push_str(&self.font_family);
        svg.push_str("; }\n");

        // Grid style
        svg.push_str("    .grid { stroke: ");
        svg.push_str(&self.grid_color.to_hex());
        svg.push_str("; stroke-dasharray: ");
        svg.push_str(self.grid_style.to_svg());
        svg.push_str("; }\n");

        // Axis style
        svg.push_str("    .axis { stroke: ");
        svg.push_str(&self.axis_color.to_hex());
        svg.push_str("; stroke-width: ");
        svg.push_str(&self.border_width.to_string());
        svg.push_str("; }\n");

        // Text style
        svg.push_str("    .title { font-size: ");
        svg.push_str(&self.title_size.to_string());
        svg.push_str("px; font-weight: bold; fill: ");
        svg.push_str(&self.text_color.to_hex());
        svg.push_str("; }\n");

        svg.push_str("    .label { font-size: ");
        svg.push_str(&self.label_size.to_string());
        svg.push_str("px; fill: ");
        svg.push_str(&self.text_color.to_hex());
        svg.push_str("; }\n");

        svg.push_str("    .tick { font-size: ");
        svg.push_str(&self.tick_size.to_string());
        svg.push_str("px; fill: ");
        svg.push_str(&self.text_color.to_hex());
        svg.push_str("; }\n");

        svg.push_str("  </style>\n");

        // Patterns for grid
        if self.show_grid {
            svg.push_str("  <pattern id=\"grid\" width=\"40\" height=\"40\" patternUnits=\"userSpaceOnUse\">\n");
            svg.push_str("    <path d=\"M 40 0 L 0 0 0 40\" fill=\"none\" stroke=\"");
            svg.push_str(&self.grid_color.to_hex());
            svg.push_str("\" stroke-width=\"0.5\"/>\n");
            svg.push_str("  </pattern>\n");
        }

        svg.push_str("</defs>\n");

        // Background
        svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"");
        svg.push_str(&self.background_color.to_hex());
        svg.push_str("\"/>\n");

        svg
    }

    /// Get nth color from palette.
    pub fn color(&self, index: usize) -> Color {
        self.palette.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_configs_compile() {
        let _ = ThemeConfig::minimal();
        let _ = ThemeConfig::seaborn();
        let _ = ThemeConfig::ggplot();
        let _ = ThemeConfig::dark();
        let _ = ThemeConfig::high_contrast();
        let _ = ThemeConfig::classic();
        let _ = ThemeConfig::academic();
        let _ = ThemeConfig::presentation();
    let _ = ThemeConfig::blueprint();
    let _ = ThemeConfig::matplotlib();
}

    #[test]
    fn palette_colors() {
        let palette = ColorPalette::seaborn_deep();
        let c1 = palette.get(0);
        let c2 = palette.get(100); // Should cycle
        assert!(c1 != c2 || palette.colors.len() == 1);
    }

    #[test]
    fn theme_svg_header() {
        let theme = ThemeConfig::seaborn();
        let svg = theme.svg_header(600, 400);
        assert!(svg.contains("<defs>"));
        assert!(svg.contains("<rect"));
    }
}
