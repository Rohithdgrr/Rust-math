//! Animation / frame-sequence export via `mathverse-image`.
//!
//! Generates a sequence of SVG frames that can be combined into
//! an animated SVG or encoded as a GIF by external tools.

use crate::common::PlotConfig;
use crate::error::PlotResult;
use crate::svg::SvgPlot;

/// Animation frame configuration.
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    /// Plot configuration applied to each frame.
    pub plot_config: PlotConfig,
    /// Number of frames in the animation.
    pub num_frames: usize,
    /// Duration per frame in milliseconds.
    pub frame_duration_ms: u64,
    /// Width of each frame in pixels.
    pub width: u32,
    /// Height of each frame in pixels.
    pub height: u32,
}

impl AnimationConfig {
    /// Create a new animation config with sensible defaults.
    #[must_use]
    pub fn new(num_frames: usize) -> Self {
        Self {
            plot_config: PlotConfig::new(),
            num_frames: num_frames.max(1),
            frame_duration_ms: 100,
            width: 800,
            height: 600,
        }
    }

    /// Set the frame duration in milliseconds.
    #[must_use]
    pub fn with_frame_duration(mut self, ms: u64) -> Self {
        self.frame_duration_ms = ms;
        self
    }

    /// Set the frame dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the plot configuration.
    #[must_use]
    pub fn with_plot_config(mut self, config: PlotConfig) -> Self {
        self.plot_config = config;
        self
    }
}

/// Generate a sequence of SVG frames for an animation.
///
/// `frame_fn` receives the frame index (0-based) and the total
/// number of frames, and returns an SVG string for that frame.
pub fn generate_frames<F>(
    frame_fn: F,
    config: AnimationConfig,
) -> PlotResult<Vec<String>>
where
    F: Fn(usize, usize) -> PlotResult<String>,
{
    let mut frames = Vec::with_capacity(config.num_frames);
    for i in 0..config.num_frames {
        let svg = frame_fn(i, config.num_frames)?;
        frames.push(svg);
    }
    Ok(frames)
}

/// Generate an animated SVG from a sequence of frames.
///
/// Each frame is embedded as a `<g>` element with a `begin`
/// attribute for timed display.
pub fn assemble_animated_svg(
    frames: &[String],
    config: &AnimationConfig,
) -> String {
    if frames.is_empty() {
        return String::from("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    }

    let _total_duration = config.num_frames as u64 * config.frame_duration_ms;

    let mut output = String::new();
    output.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\n",
        config.width, config.height
    ));
    output.push_str(&format!(
        "  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n"
    ));

    for (i, frame) in frames.iter().enumerate() {
        let begin = i as u64 * config.frame_duration_ms;
        let end = begin + config.frame_duration_ms;
        let content = frame
            .lines()
            .skip(1)
            .take_while(|l| !l.trim().starts_with("</svg>"))
            .collect::<Vec<_>>()
            .join("\n");
        output.push_str(&format!(
            "  <g begin=\"{}ms\" end=\"{}ms\" visibility=\"hidden\">\n{}\n  </g>\n",
            begin, end, content
        ));
    }

    output.push_str("</svg>\n");
    output
}

/// Generate a single-frame SVG from a data series for a given time step.
///
/// This is a convenience wrapper that creates a plot with the
/// given config and adds a single series.
pub fn render_frame(
    config: PlotConfig,
    series: crate::DataSeries,
) -> PlotResult<String> {
    let mut plot = SvgPlot::new(config);
    plot.add_series(series);
    Ok(plot.generate())
}

/// Generate frames for a parametric animation.
///
/// `parametric_fn` computes a `DataSeries` for a given time `t` in [0, 1].
pub fn generate_parametric_animation<F>(
    parametric_fn: F,
    config: AnimationConfig,
) -> PlotResult<Vec<String>>
where
    F: Fn(f64) -> PlotResult<crate::DataSeries>,
{
    let plot_config = config.plot_config.clone();
    generate_frames(
        move |i, total| {
            let t = if total > 1 { i as f64 / (total - 1) as f64 } else { 0.0 };
            let series = parametric_fn(t)?;
            render_frame(plot_config.clone(), series)
        },
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_frames_returns_correct_count() {
        let config = AnimationConfig::new(5);
        let frames = generate_frames(
            |i, total| {
                let series = crate::DataSeries::new(
                    format!("frame_{}", i),
                    vec![crate::DataPoint::new(i as f64, i as f64 / total as f64)],
                );
                render_frame(PlotConfig::new(), series)
            },
            config,
        )
        .unwrap();
        assert_eq!(frames.len(), 5);
    }

    #[test]
    fn animated_svg_contains_frames() {
        let config = AnimationConfig::new(3);
        let frames = generate_frames(
            |i, _| {
                let series = crate::DataSeries::new(
                    format!("frame_{}", i),
                    vec![crate::DataPoint::new(i as f64, i as f64)],
                );
                render_frame(PlotConfig::new(), series)
            },
            config.clone(),
        )
        .unwrap();
        let svg = assemble_animated_svg(&frames, &config);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("begin="));
    }

    #[test]
    fn parametric_animation_renders() {
        let config = AnimationConfig::new(4);
        let frames = generate_parametric_animation(
            |t| {
                let points: Vec<crate::DataPoint> = (0..10)
                    .map(|i| {
                        let x = i as f64 * 0.1;
                        let y = (x + t).sin();
                        crate::DataPoint::new(x, y)
                    })
                    .collect();
                let series = crate::DataSeries::new("wave".to_string(), points);
                Ok(series)
            },
            config,
        )
        .unwrap();
        assert_eq!(frames.len(), 4);
    }

    #[test]
    fn empty_frames_returns_empty_svg() {
        let config = AnimationConfig::new(0);
        let svg = assemble_animated_svg(&[], &config);
        assert!(svg.contains("<svg"));
    }
}