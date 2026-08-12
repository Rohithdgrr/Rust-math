//! Global runtime configuration — the analogue of matplotlib's `rcParams`.
//!
//! A process-wide, mutable, thread-local config tree that new [`RcParams`]
//! consumers snapshot when they are constructed. Later [`rc_set`] changes
//! affect newly-created plots but never retroactively mutate plots that have
//! already been built (exactly the snapshot semantics of matplotlib).
//!
//! # Example
//!
//! ```rust
//! use mathverse_plot::rcparams::{rc, rc_set, reset};
//! use mathverse_plot::theme::Theme;
//!
//! rc_set(|p| {
//!     p.theme = Theme::Dark;
//!     p.tick_count = 8;
//!     p.figsize = (1200, 700);
//! });
//! let snap = rc(); // snapshot for a new plot
//! assert_eq!(snap.theme, Theme::Dark);
//! reset();
//! ```

use std::sync::OnceLock;
use std::sync::RwLock;

use crate::axes::Scale;
use crate::style::Color;
use crate::theme::{ColorPalette, GridAxis, Theme, ThemeConfig};

static GLOBAL: OnceLock<RwLock<RcParams>> = OnceLock::new();

fn global() -> &'static RwLock<RcParams> {
    GLOBAL.get_or_init(|| RwLock::new(RcParams::builtin_default()))
}

/// Global plot defaults, mirroring matplotlib's `rcParams`.
#[derive(Debug, Clone)]
pub struct RcParams {
    /// Active theme preset (used to derive [`ThemeConfig`]).
    pub theme: Theme,
    /// Color palette cycled for new series (auto color-cycling).
    pub palette: ColorPalette,
    /// Default figure width/height in px.
    pub figsize: (u32, u32),
    /// Raster output DPI.
    pub dpi: u32,
    /// Target tick count per axis (nice-ticks).
    pub tick_count: usize,
    /// Fractional data margin applied around the data bounds.
    pub margin_frac: f64,
    /// Line width applied to new series.
    pub line_width: f64,
    /// Marker size applied to new series.
    pub marker_size: f64,
    /// Default x-axis scale.
    pub x_scale: Scale,
    /// Default y-axis scale.
    pub y_scale: Scale,
    /// Default grid axis selection.
    pub grid_axis: GridAxis,
    /// Whether legends render by default.
    pub show_legend: bool,
    /// Font family for new plots (matplotlib `font.family`).
    pub font_family: String,
    /// Base font size in px (matplotlib `font.size`).
    pub font_size: f64,
}

impl RcParams {
    /// Self-contained defaults, safe to call during lazy init.
    fn builtin_default() -> Self {
        Self {
            theme: Theme::Minimal,
            palette: ColorPalette::seaborn_deep(),
            figsize: (800, 600),
            dpi: 96,
            tick_count: 6,
            margin_frac: 0.05,
            line_width: 2.0,
            marker_size: 5.0,
            x_scale: Scale::Linear,
            y_scale: Scale::Linear,
            grid_axis: GridAxis::None,
            show_legend: true,
            font_family: "Arial, sans-serif".to_string(),
            font_size: 14.0,
        }
    }

    /// A snapshot of the current global params.
    pub fn snapshot() -> Self {
        rc()
    }

    /// Derived `ThemeConfig` from the current `theme` preset.
    pub fn theme_config(&self) -> ThemeConfig {
        ThemeConfig::new(self.theme)
    }

    /// The palette color at cycling index `i`.
    #[must_use]
    pub fn color(&self, index: usize) -> Color {
        self.palette.get(index)
    }
}

impl Default for RcParams {
    fn default() -> Self {
        rc()
    }
}

/// Return a snapshot of the current global params.
pub fn rc() -> RcParams {
    global().read().unwrap_or_else(|e| panic!("rcParams lock poisoned: {e}")).clone()
}

/// Mutate the global params in place.
///
/// The closure receives `&mut RcParams`. Existing plots are unaffected.
pub fn rc_set(f: impl FnOnce(&mut RcParams)) {
    let mut p = global().write().unwrap_or_else(|e| panic!("rcParams lock poisoned: {e}"));
    f(&mut p);
}

/// Set the global theme preset (also syncs the default palette).
pub fn set_theme(theme: Theme) {
    rc_set(|p| {
        p.theme = theme;
        p.palette = p.theme_config().palette;
    });
}

/// Set the global color palette.
pub fn set_palette(palette: ColorPalette) {
    rc_set(|p| p.palette = palette);
}

/// Set the global figure size in px.
pub fn set_figsize(width: u32, height: u32) {
    rc_set(|p| p.figsize = (width, height));
}

/// Set the global tick count (nice-tick density).
pub fn set_tick_count(count: usize) {
    rc_set(|p| p.tick_count = count.max(1));
}

/// Reset all global params to defaults.
pub fn reset() {
    *global().write().unwrap_or_else(|e| panic!("rcParams lock poisoned: {e}")) = RcParams::builtin_default();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip() {
        rc_set(|p| {
            p.tick_count = 9;
            p.figsize = (1600, 900);
        });
        let snap = rc();
        assert_eq!(snap.tick_count, 9);
        assert_eq!(snap.figsize, (1600, 900));
        reset();
        assert_eq!(rc().tick_count, 6);
    }

    #[test]
    fn theme_switch_syncs_palette() {
        set_theme(Theme::Dark);
        assert_eq!(rc().theme, Theme::Dark);
        assert_eq!(rc().palette.name, "dark");
        reset();
    }

    #[test]
    fn palette_cycles() {
        let palette = ColorPalette::seaborn_deep();
        assert_eq!(palette.get(0), palette.get(palette.colors.len()));
    }

    #[test]
    fn snapshots_are_independent() {
        let a = rc();
        rc_set(|p| p.line_width = 7.5);
        let b = rc();
        assert_eq!(a.line_width, 2.0);
        assert_eq!(b.line_width, 7.5);
        reset();
    }
}