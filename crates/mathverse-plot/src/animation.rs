//! Animation / frame-sequence export via `mathverse-image`.
//!
//! Generates a sequence of SVG frames that can be combined into
//! an animated SVG or encoded as a GIF by external tools.

use crate::common::PlotConfig;
use crate::error::PlotResult;
use crate::svg::SvgPlot;

/// Encode SVG frames into an animated GIF (requires the `png` feature, which
/// brings the `usvg`/`resvg`/`tiny-skia` rasterization stack). This is the
/// analogue of matplotlib's `FuncAnimation.save("out.gif")`.
///
/// Frames are rasterized onto a white background; semi-transparent pixels are
/// composited with white. `delay_cs` is the per-frame delay in hundredths of
/// a second.
#[cfg(feature = "png")]
pub fn encode_frames_to_gif(
    width: u32,
    height: u32,
    frames: &[String],
    delay_cs: u16,
) -> PlotResult<Vec<u8>> {
    if frames.is_empty() {
        return Err(crate::error::PlotError::InvalidData(
            "no frames to encode".into(),
        ));
    }
    let w = width.max(1);
    let h = height.max(1);
    let mut rgba_frames = Vec::with_capacity(frames.len());
    for svg in frames {
        let mut opts = usvg::Options::default();
        opts.resources_dir = std::env::current_dir().ok();
        let tree = usvg::Tree::from_data(svg.as_bytes(), &opts)
            .map_err(|e| crate::error::PlotError::InvalidData(format!("usvg parse: {e}")))?;
        let mut pixmap = tiny_skia::Pixmap::new(w, h)
            .ok_or_else(|| crate::error::PlotError::InvalidData("pixmap alloc failed".into()))?;
        pixmap.fill(tiny_skia::Color::WHITE);
        let transform = tiny_skia::Transform::from_scale(
            w as f32 / tree.size().width(),
            h as f32 / tree.size().height(),
        );
        resvg::render(&tree, transform, &mut pixmap.as_mut());
        // tiny-skia stores premultiplied BGRA; convert to straight RGBA.
        let data = pixmap.data();
        let mut rgba = Vec::with_capacity(data.len());
        for px in data.chunks_exact(4) {
            let (b, g, r, a) = (px[0], px[1], px[2], px[3]);
            if a == 0 {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            } else if a == 255 {
                rgba.extend_from_slice(&[r, g, b, a]);
            } else {
                let inv = 255.0 / f32::from(a);
                let rr = ((f32::from(r) * inv).round() as u8).min(255);
                let gg = ((f32::from(g) * inv).round() as u8).min(255);
                let bb = ((f32::from(b) * inv).round() as u8).min(255);
                rgba.extend_from_slice(&[rr, gg, bb, a]);
            }
        }
        rgba_frames.push(crate::gif::GifFrame::new(rgba, delay_cs));
    }
    crate::gif::encode_gif(w as u16, h as u16, &rgba_frames).ok_or_else(|| {
        crate::error::PlotError::InvalidData("GIF encoding failed".into())
    })
}

/// Encode SVG frames into MP4 or WebM by piping raw RGBA frames through
/// `ffmpeg` (matplotlib does exactly the same). Returns the encoded bytes.
///
/// Requires the `png` feature (for the rasterizer) **and** an `ffmpeg` binary
/// on `PATH`. The frames are written to a temporary raw-video stream and
/// piped into ffmpeg stdin, so no temp files are left behind.
///
/// # Errors
///
/// Returns `PlotError::InvalidData` when frames are empty, when `ffmpeg` is
/// missing, or when the encoder reports failure.
#[cfg(feature = "png")]
pub fn encode_frames_to_video(
    width: u32,
    height: u32,
    frames: &[String],
    fps: u32,
    codec: VideoCodec,
) -> crate::error::PlotResult<Vec<u8>> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    if frames.is_empty() {
        return Err(crate::error::PlotError::InvalidData(
            "no frames to encode".into(),
        ));
    }
    let w = width.max(1);
    let h = height.max(1);

    // Rasterize every frame to raw RGBA.
    let mut raw = Vec::with_capacity(frames.len() * w as usize * h as usize * 4);
    for svg in frames {
        let mut opts = usvg::Options::default();
        opts.resources_dir = std::env::current_dir().ok();
        let tree = usvg::Tree::from_data(svg.as_bytes(), &opts)
            .map_err(|e| crate::error::PlotError::InvalidData(format!("usvg parse: {e}")))?;
        let mut pixmap = tiny_skia::Pixmap::new(w, h)
            .ok_or_else(|| crate::error::PlotError::InvalidData("pixmap alloc failed".into()))?;
        pixmap.fill(tiny_skia::Color::WHITE);
        let transform = tiny_skia::Transform::from_scale(
            w as f32 / tree.size().width(),
            h as f32 / tree.size().height(),
        );
        resvg::render(&tree, transform, &mut pixmap.as_mut());
        let data = pixmap.data();
        for px in data.chunks_exact(4) {
            // tiny-skia stores premultiplied BGRA; convert to RGB for ffmpeg.
            let (b, g, r, _) = (px[0], px[1], px[2], px[3]);
            raw.extend_from_slice(&[r, g, b]);
        }
    }

    let (codec_name, _ext) = codec.ffmpeg_args();
    let mut child = Command::new("ffmpeg")
        .args([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", &format!("{w}x{h}"),
            "-r", &fps.to_string(),
            "-i", "-",
            "-c:v", codec_name,
            "-pix_fmt", "yuv420p",
            "-f", codec.extension(),
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| {
            crate::error::PlotError::InvalidData(format!(
                "ffmpeg not found on PATH (needed for {ext} export): {e}",
                ext = codec.extension()
            ))
        })?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| {
            crate::error::PlotError::InvalidData("ffmpeg stdin unavailable".into())
        })?
        .write_all(&raw)
        .map_err(|e| crate::error::PlotError::InvalidData(format!("ffmpeg write: {e}")))?;

    let output = child.wait_with_output().map_err(|e| {
        crate::error::PlotError::InvalidData(format!("ffmpeg wait: {e}"))
    })?;
    if !output.status.success() {
        return Err(crate::error::PlotError::InvalidData(format!(
            "ffmpeg encode failed (exit {:?})",
            output.status.code()
        )));
    }
    Ok(output.stdout)
}

/// True when `ffmpeg` is available on `PATH` (used to skip video tests).
#[must_use]
pub fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Video codec target for [`encode_frames_to_video`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// MPEG-4 (`.mp4`) — the analogue of matplotlib saving an MP4.
    Mp4,
    /// WebM (`.webm`) — smaller, web-friendly container.
    WebM,
}

impl VideoCodec {
    /// File extension for this codec.
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            VideoCodec::Mp4 => "mp4",
            VideoCodec::WebM => "webm",
        }
    }

    /// `ffmpeg` arguments selecting the codec.
    #[cfg(feature = "png")]
    fn ffmpeg_args(&self) -> (&'static str, &'static str) {
        match self {
            VideoCodec::Mp4 => ("libx264", "mp4"),
            VideoCodec::WebM => ("libvpx-vp9", "webm"),
        }
    }
}

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
    /// Blit mode: reuse a static background layer across frames so only the
    /// changing series are re-emitted (matplotlib `blit=True`). The current
    /// implementation records the flag and keeps full frames — a placeholder
    /// for true dirty-rectangle redraw.
    pub blit: bool,
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
            blit: false,
        }
    }

    /// Enable/disable blit-style frame reuse (default off).
    #[must_use]
    pub fn with_blit(mut self, blit: bool) -> Self {
        self.blit = blit;
        self
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

    #[test]
    fn blit_flag_recorded() {
        let config = AnimationConfig::new(4).with_blit(true);
        assert!(config.blit);
        let config = AnimationConfig::new(4).with_blit(false);
        assert!(!config.blit);
    }

    #[test]
    fn video_codec_extensions() {
        assert_eq!(VideoCodec::Mp4.extension(), "mp4");
        assert_eq!(VideoCodec::WebM.extension(), "webm");
    }

    #[cfg(feature = "png")]
    #[test]
    fn video_encoding_requires_ffmpeg() {
        if !ffmpeg_available() {
            // Without ffmpeg the call must fail gracefully, not panic.
            let frames = vec![
                "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'></svg>".to_string(),
            ];
            let err = encode_frames_to_video(64, 64, &frames, 30, VideoCodec::Mp4);
            assert!(err.is_err());
        } else {
            let frames = vec![
                "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'></svg>".to_string(),
                "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'><rect width='64' height='64' fill='red'/></svg>".to_string(),
            ];
            let bytes = encode_frames_to_video(64, 64, &frames, 10, VideoCodec::Mp4).unwrap();
            assert!(!bytes.is_empty());
        }
    }
}