//! End-to-end verification of the 9 matplotlib-parity features added to
//! `mathverse-plot`, using real data. Writes output files to
//! `target/feature_check/` and asserts the expected artifacts are produced.
//!
//! Run with: `cargo run --all-features --example feature_check`

use mathverse_plot::backend::PlotData;
use mathverse_plot::patches::{LineCollection, Patch, Path};
use mathverse_plot::style::Color;
use mathverse_plot::transforms::Position;
use mathverse_plot::imshow::Interpolation;
use mathverse_plot::plt::Axes;
use mathverse_plot::{
    PlotConfig, render_interactive_html, InteractiveConfig, ImageData, mathtext, ticks, color,
};
use mathverse_plot::webgl_3d::{Point3D, SurfaceMesh, WebGL3D, WebGL3DConfig, render_surface_html};
#[cfg(feature = "png")]
use mathverse_plot::{AnimationConfig, assemble_animated_svg, encode_frames_to_gif, generate_frames};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::Path::new("target/feature_check");
    std::fs::create_dir_all(out_dir)?;
    println!("output -> {}", out_dir.display());

    // ---- #8 Coordinate transforms: resolve positions in 4 coordinate systems ----
    let x_px = |x: f64| 60.0 + (x - 0.0) / 10.0 * 400.0; // data 0..10 -> px 60..460
    let y_px = |y: f64| 320.0 - (y - 0.0) / 10.0 * 300.0; // data 0..10 -> px 320..20
    let rect = (60.0, 20.0, 400.0, 300.0); // plot rect (left, top, w, h)
    let fig = (500.0, 400.0);
    let p_data = Position::data(5.0, 5.0).to_pixel(&x_px, &y_px, rect, fig);
    let p_ax = Position::AxesFraction(0.5, 0.5).to_pixel(&x_px, &y_px, rect, fig);
    let p_fig = Position::FigureFraction(0.5, 0.5).to_pixel(&x_px, &y_px, rect, fig);
    let p_blend = Position::BlendAxesXDataY { x_frac: 0.5, y_data: 5.0 }
        .to_pixel(&x_px, &y_px, rect, fig);
    assert!((p_data.0 - 260.0).abs() < 1e-9, "data x: {}", p_data.0);
    assert!((p_data.1 - 170.0).abs() < 1e-9, "data y: {}", p_data.1);
    assert!((p_ax.0 - 260.0).abs() < 1e-9 && (p_ax.1 - 170.0).abs() < 1e-9);
    assert!((p_fig.0 - 250.0).abs() < 1e-9 && (p_fig.1 - 200.0).abs() < 1e-9);
    assert!((p_blend.0 - 260.0).abs() < 1e-9 && (p_blend.1 - 170.0).abs() < 1e-9);
    println!("[8] transforms: data={p_data:?} axes-frac={p_ax:?} fig-frac={p_fig:?} blend={p_blend:?} OK");

    // ---- #3 Tick locators & formatters ----
    use ticks::{TickFormatter, TickLocator};
    let ml = ticks::MultipleLocator::new(0.5).unwrap();
    let ticks_050 = ml.locate(0.0, 3.0, 10);
    assert!(ticks_050.len() >= 5, "multiple locator tick count");
    let mx = ticks::MaxNLocator::new();
    let ticks_mx = mx.locate(0.0, 1000.0, 10);
    // Soft target: ~10 intervals means up to 11 positions (0..1000 step 100).
    assert!(ticks_mx.len() <= 11 && !ticks_mx.is_empty(), "max-n locator count");
    let fixed = ticks::FixedLocator::new(vec![0.0, 1.0, 3.0]).locate(-1.0, 4.0, 10);
    assert_eq!(fixed.len(), 3);
    let f = ticks::FuncFormatter::new(|v| format!("{v:.1} V"));
    let fmt = f.format(2.0);
    assert_eq!(fmt, "2.0 V");
    let nice = ticks::nice_ticks(0.0, 1.0, 6);
    assert!(nice.len() <= 6 && !nice.is_empty());
    println!("[3] ticks: multi={ticks_050:?} maxn={ticks_mx:?} fixed={fixed:?} func={fmt} nice={nice:?} OK");

    // ---- #4 Colormap construction + norms ----
    let cmap = color::LinearSegmentedColormap::from_list_rgb(&[(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]);
    let c0 = cmap.map(0.0);
    let c1 = cmap.map(1.0);
    assert_eq!(c0, Color::rgb(0, 0, 0));
    assert_eq!(c1, Color::rgb(255, 255, 255));
    let mid = color::normalize(&[-5.0, 0.0, 5.0], color::Normalization::Linear);
    assert_eq!(mid, vec![0.0, 0.5, 1.0]);
    let two = color::normalize_two_slope(&[-10.0, 0.0, 10.0], -10.0, 0.0, 10.0);
    assert!((two[0] - 0.0).abs() < 1e-12 && (two[1] - 0.5).abs() < 1e-12 && (two[2] - 1.0).abs() < 1e-12);
    // BoundaryNorm: each bin maps to the center of its color segment -> [1/6, 0.5, 5/6].
    let bnd = color::normalize_boundary(&[0.5, 1.5, 2.5], &[0.0, 1.0, 2.0, 3.0]);
    assert!((bnd[0] - 1.0 / 6.0).abs() < 1e-9 && (bnd[1] - 0.5).abs() < 1e-9 && (bnd[2] - 5.0 / 6.0).abs() < 1e-9);
    let vals = color::color_by_value(&[-1.0, 0.0, 1.0], color::Normalization::Linear, color::viridis);
    assert_eq!(vals.len(), 3);
    println!("[4] colormaps+norms: stops {c0:?}..{c1:?} mid={mid:?} two-slope={two:?} boundary={bnd:?} OK");

    // ---- #5 imshow ----
    let n = 32;
    let grid: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            (0..n)
                .map(|c| {
                    let x = c as f64 / (n - 1) as f64 * 8.0 - 4.0;
                    let y = r as f64 / (n - 1) as f64 * 8.0 - 4.0;
                    (x * x + y * y).sqrt().sin() * (x * y).cos()
                })
                .collect()
        })
        .collect();
    let img = ImageData::new(grid.clone(), color::viridis)?
        .with_vmin_vmax(-1.0, 1.0)
        .with_extent(-4.0, 4.0, -4.0, 4.0)
        .with_interpolation(Interpolation::Bilinear);
    assert_eq!(img.rows(), n);
    assert_eq!(img.cols(), n);
    let cells = img.cells(16);
    assert!(cells.len() > 0, "imshow must produce cells");
    let rs = img.resample(8, 8);
    assert_eq!(rs.len(), 8);
    assert_eq!(rs[0].len(), 8);
    println!("[5] imshow: {}x{} grid -> {} cells, resample 8x8 OK", n, n, cells.len());

    // ---- #6 Paths / patches / collections ----
    let tri = Patch::filled(Path::polygon(&[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]), Color::rgb(200, 80, 80));
    let rect_p = Patch::rectangle(2.0, 2.0, 1.0, 0.5).with_opacity(0.5);
    let circ = Patch::circle(5.0, 5.0, 0.4).with_stroke(Color::BLUE);
    let segs = LineCollection::from_xy(
        &[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0],
        &[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0],
        Color::GREEN, 1.0,
    );
    assert_eq!(segs.segments.len(), 3);
    println!("[6] patches: triangle={} rect={} circle={} line-collection={} OK",
        tri.path.points().len(), rect_p.path.points().len(), circ.path.points().len(), segs.segments.len());

    // ---- #1 mathtext ----
    let math = mathtext::render(r"E = $mc^2$ and $\alpha \leq \infty$");
    assert!(math.contains("mc"), "superscript base kept: {math}");
    assert!(math.contains('\u{03B1}'), "greek alpha: {math}");
    assert!(math.contains('\u{2264}'), "leq symbol: {math}");
    assert!(mathtext::contains_math(r"$x^2$"));
    assert!(!mathtext::contains_math("plain text"));
    println!("[1] mathtext: {math:?} OK");

    // ---- Build the full plot with real data ----
    let mut ax = Axes::new();
    let n_pts = 200;
    let xs: Vec<f64> = (0..=n_pts).map(|i| i as f64 / 20.0).collect();
    let ys_sin: Vec<f64> = xs.iter().map(|x| (x * 1.5).sin()).collect();
    let ys_cos: Vec<f64> = xs.iter().map(|x| (x * 1.2).cos() * 0.8).collect();
    ax.set_title(r"Physics: $E = mc^2$, $\sin(x)$, $\alpha$");
    ax.set_xlabel(r"time $t$ [s]");
    ax.set_ylabel(r"amplitude $A$");
    ax.plot(&xs, &ys_sin, "sin(1.5t)");
    ax.plot(&xs, &ys_cos, "cos(1.2t)");
    ax.add_patch(&tri);
    ax.add_patch(&rect_p);
    ax.add_patch(&circ);
    ax.add_line_collection(&segs);
    ax.imshow(grid, color::viridis)?;
    ax.axvline(std::f64::consts::PI);
    let svg = ax.render();
    assert!(svg.contains("<svg"), "svg root");
    assert!(svg.contains("E = mc"), "mathtext in title");
    assert!(svg.contains('\u{03B1}') || svg.contains("alpha"), "greek in title");
    assert!(svg.contains("polygon") || svg.contains("<path"), "patch rendered");
    assert!(svg.contains("polyline") || svg.contains("<path"), "line collection rendered");
    println!("[plt] combined SVG: {} bytes", svg.len());
    std::fs::write(out_dir.join("all_features.svg"), &svg)?;

    // Rasterize the same real plot to PNG (default format).
    #[cfg(feature = "png")]
    {
        let png = mathverse_plot::save::PlotSaver::new(&svg)
            .with_quality(92)
            .raster_bytes(mathverse_plot::save::OutputFormat::Png, 144, 1.0)?;
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A], "PNG magic");
        std::fs::write(out_dir.join("all_features.png"), &png)?;
        println!("[plt] PNG: {} bytes (144 dpi) OK", png.len());
    }

    // ---- #2 Interactive HTML with click handler ----
    let pts2: Vec<mathverse_plot::DataPoint> = xs
        .iter()
        .zip(ys_sin.iter())
        .map(|(x, y)| mathverse_plot::DataPoint::new(*x, *y))
        .collect();
    let mut idata = PlotData::new(PlotConfig::new().with_title("Interactive".to_string()));
    idata
        .series
        .push(mathverse_plot::DataSeries::new("wave".to_string(), pts2));
    let html = render_interactive_html(
        &idata,
        &InteractiveConfig::new().with_on_point_click(
            "console.log('clicked', pt.series, pt.x, pt.y); alert('point: ' + pt.x + ', ' + pt.y);"
        ),
    )?;
    assert!(html.contains("addEventListener('click'"), "click handler wired");
    assert!(html.contains("pt.series"), "click handler payload");
    std::fs::write(out_dir.join("interactive.html"), &html)?;
    println!("[2] interactive html: {} bytes, click handler OK", html.len());

    // ---- #7 GIF animation (real frames, rasterized through resvg) ----
    #[cfg(feature = "png")]
    {
        let anim_cfg = AnimationConfig::new(12)
            .with_frame_duration(10)
            .with_dimensions(400, 300)
            .with_plot_config(
                PlotConfig::new()
                    .with_title("Animated wave")
                    .with_x_label("t")
                    .with_y_label("y"),
            );
        let frames = generate_frames(
            |i, total| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / total as f64;
                let pts: Vec<mathverse_plot::DataPoint> = xs
                    .iter()
                    .map(|x| mathverse_plot::DataPoint::new(*x, (x * 2.0 + phase).sin()))
                    .collect();
                mathverse_plot::render_frame(
                    PlotConfig::new(),
                    mathverse_plot::DataSeries::new("wave", pts),
                )
            },
            anim_cfg.clone(),
        )?;
        assert_eq!(frames.len(), 12);
        let animated_svg = assemble_animated_svg(&frames, &anim_cfg);
        std::fs::write(out_dir.join("animation.svg"), &animated_svg)?;
        let gif_bytes = encode_frames_to_gif(400, 300, &frames, 10)?;
        assert!(gif_bytes.len() > 100, "gif non-trivial");
        assert_eq!(&gif_bytes[..6], b"GIF89a", "gif magic");
        std::fs::write(out_dir.join("animation.gif"), &gif_bytes)?;
        println!("[7] gif: {} frames -> {} bytes, animated svg {} bytes OK",
            frames.len(), gif_bytes.len(), animated_svg.len());
    }

    // ---- #9 3D surface mesh ----
    let g = 24;
    let zgrid: Vec<Vec<f64>> = (0..g)
        .map(|r| {
            (0..g)
                .map(|c| {
                    let x = c as f64 / (g - 1) as f64 * 4.0 - 2.0;
                    let y = r as f64 / (g - 1) as f64 * 4.0 - 2.0;
                    (x * x + y * y).exp().recip() * 0.8
                })
                .collect()
        })
        .collect();
    let mesh = SurfaceMesh::from_grid(zgrid, (-2.0, 2.0), (-2.0, 2.0))?;
    let tris = mesh.triangles();
    let wires = mesh.wireframe_segments();
    assert_eq!(tris.len(), (g - 1) * (g - 1) * 2, "triangle count");
    assert_eq!(wires.len(), (g - 1) * g * 2, "wireframe segment count");
    let b = mesh.bounds();
    assert!((b.0 - -2.0).abs() < 1e-9 && (b.1 - 2.0).abs() < 1e-9);
    let html3d = render_surface_html(&mesh, &WebGL3DConfig::new());
    assert!(html3d.contains("canvas"), "3d canvas");
    std::fs::write(out_dir.join("surface.html"), &html3d)?;
    let pts: Vec<Point3D> = (0..50)
        .map(|i| {
            let t = i as f64 * 0.13;
            Point3D::new(t, (t * 2.0).sin(), (t * 1.7).cos())
        })
        .collect();
    let scatter3d = WebGL3D::new(pts).render_html();
    assert!(scatter3d.contains("canvas"));
    println!("[9] 3d: {} triangles, {} wire segments, surface.html {} bytes, scatter OK",
        tris.len(), wires.len(), html3d.len());

    // ---- Output file manifest ----
    let mut entries: Vec<String> = Vec::new();
    for e in std::fs::read_dir(out_dir)? {
        let e = e?;
        entries.push(format!("  {} ({} bytes)", e.file_name().to_string_lossy(), e.metadata()?.len()));
    }
    entries.sort();
    println!("== files written ==");
    for e in entries {
        println!("{e}");
    }
    println!("\nALL 9 FEATURE CHECKS PASSED");
    Ok(())
}
