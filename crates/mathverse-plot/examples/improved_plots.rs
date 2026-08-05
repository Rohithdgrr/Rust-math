//! Improved UI demo — uses the new Matplotlib-style defaults + Seaborn theme.
//!
//! Run:  cargo run --example improved_plots -p mathverse-plot
//!
//! Output: target/plot_images/*.svg (all viewable in any browser).

use mathverse_plot::*;

fn main() {
    let out = "target/plot_images";
    std::fs::create_dir_all(out).expect("create output dir");

    // ---------- helpers ----------
    let theme = ThemeConfig::matplotlib();
    let seaborn = ThemeConfig::seaborn();

    macro_rules! save {
        ($name:expr, $plot:expr) => {
            std::fs::write(format!("{out}/{name}.svg"), $plot.generate())
                .expect("write svg");
        };
    }

    // ─── 1. Sine wave — theme=matplotlib ───
    {
        let cfg = PlotConfig::new()
            .with_title("Sine Wave")
            .with_x_label("x (radians)")
            .with_y_label("sin(x)")
            .with_dimensions(900, 500);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let pts: Vec<DataPoint> = (0..300)
            .map(|i| DataPoint::new(i as f64 * 0.05, (i as f64 * 0.05).sin()))
            .collect();
        plot.add_series(DataSeries::new("sin(x)", pts));
        save!("01_sine_wave", plot);
    }

    // ─── 2. Multi-series line ───
    {
        let cfg = PlotConfig::new()
            .with_title("Temperature & Humidity — 24 Hours")
            .with_x_label("Hour of Day")
            .with_y_label("Value")
            .with_dimensions(900, 500);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let temp: Vec<DataPoint> = (0..24)
            .map(|i| DataPoint::new(i as f64, 20.0 + 10.0 * ((i as f64 - 6.0) * PI / 12.0).cos()))
            .collect();
        let humid: Vec<DataPoint> = (0..24)
            .map(|i| DataPoint::new(i as f64, 60.0 + 20.0 * ((i as f64 - 12.0) * PI / 12.0).sin()))
            .collect();
        plot.add_series(DataSeries::new("Temperature °C", temp));
        plot.add_series(DataSeries::new("Humidity %", humid));
        save!("02_multi_series", plot);
    }

    // ─── 3. Exponential growth & decay ───
    {
        let cfg = PlotConfig::new()
            .with_title("Exponential Growth & Decay")
            .with_dimensions(900, 500);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let growth: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint::new(i as f64 * 0.1, (i as f64 * 0.05).exp()))
            .collect();
        let decay: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint::new(i as f64 * 0.1, (-i as f64 * 0.05).exp()))
            .collect();
        plot.add_series(DataSeries::new("Growth  e^(0.05x)", growth));
        plot.add_series(DataSeries::new("Decay  e^(-0.05x)", decay));
        save!("03_exponential", plot);
    }

    // ─── 4. Scatter with correlation ───
    {
        let cfg = PlotConfig::new()
            .with_title("Study Hours vs Exam Score")
            .with_x_label("Study Hours")
            .with_y_label("Exam Score")
            .with_dimensions(900, 500);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let pts: Vec<DataPoint> = (0..50)
            .map(|i| {
                let hours = i as f64 * 0.2;
                let score = 40.0 + hours * 5.0 + ((i as f64 * 1.7).sin() * 10.0);
                DataPoint::new(hours, score)
            })
            .collect();
        let style = PlotStyle::new(Color::BLUE, 2.0)
            .with_marker(MarkerStyle::Circle, 4.0, Color::BLUE);
        plot.add_series(DataSeries::with_style("Students", pts, style));
        save!("04_scatter", plot);
    }

    // ─── 5. Smoothed noisy signal ───
    {
        let cfg = PlotConfig::new()
            .with_title("Noisy Signal → Catmull-Rom Smooth")
            .with_dimensions(900, 500);
        let mut plot = SvgPlot::new(cfg).with_theme(seaborn.clone());
        let raw: Vec<DataPoint> = (0..30)
            .map(|i| {
                let x = i as f64 * 0.2;
                let y = x.sin() + ((i as f64 * 1.7).sin() * 0.3);
                DataPoint::new(x, y)
            })
            .collect();
        let smooth_cfg = SmoothConfig::default();
        let smoothed = smooth_points(&raw, &smooth_cfg);
        plot.add_series(DataSeries::new("Raw", raw));
        plot.add_series(DataSeries::new("Smoothed", smoothed));
        save!("05_smooth", plot);
    }

    // ─── 6. Damped oscillation ───
    {
        let cfg = PlotConfig::new()
            .with_title("Damped Harmonic Oscillator")
            .with_x_label("Time t")
            .with_y_label("Amplitude")
            .with_dimensions(900, 450);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let pts: Vec<DataPoint> = (0..400)
            .map(|i| {
                let t = i as f64 * 0.05;
                DataPoint::new(t, (-t * 0.15).sin() * (-t * 0.08).exp())
            })
            .collect();
        plot.add_series(DataSeries::new("Damped", pts));
        save!("06_damped", plot);
    }

    // ─── 7. Fourier magnitude spectrum ───
    {
        let cfg = PlotConfig::new()
            .with_title("Power Spectrum (Mock FFT Magnitude)")
            .with_x_label("Frequency (Hz)")
            .with_y_label("Magnitude |F|")
            .with_dimensions(900, 450);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let pts: Vec<DataPoint> = (0..200)
            .map(|i| {
                let f = i as f64 * 0.5;
                let mag = 10.0 * (-((f - 50.0).powi(2)) / 200.0).exp()
                    + 5.0 * (-((f - 120.0).powi(2)) / 100.0).exp()
                    + 3.0 * (-((f - 80.0).powi(2)) / 50.0).exp();
                DataPoint::new(f, mag)
            })
            .collect();
        plot.add_series(DataSeries::new("Spectrum", pts));
        save!("07_spectrum", plot);
    }

    // ─── 8. Phase portrait (Van der Pol) ───
    {
        let cfg = PlotConfig::new()
            .with_title("Phase Portrait: Van der Pol Oscillator (μ=2)")
            .with_x_label("x")
            .with_y_label("dx/dt")
            .with_dimensions(700, 500);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let mu = 2.0;
        let mut x = 0.1;
        let mut v = 0.0;
        let dt = 0.01;
        let mut pts = Vec::new();
        for _ in 0..8000 {
            let dv = mu * (1.0 - x * x) * v - x;
            x += v * dt;
            v += dv * dt;
            pts.push(DataPoint::new(x, v));
        }
        plot.add_series(DataSeries::new("Trajectory", pts));
        save!("08_phase_portrait", plot);
    }

    // ─── 9. Lissajous 3:2 ───
    {
        let cfg = PlotConfig::new()
            .with_title("Lissajous Figure (frequency ratio 3:2)")
            .with_dimensions(600, 600);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let pts: Vec<DataPoint> = (0..600)
            .map(|i| {
                let t = i as f64 * 0.05;
                DataPoint::new((3.0 * t).sin(), (2.0 * t).cos())
            })
            .collect();
        plot.add_series(DataSeries::new("Lissajous", pts));
        save!("09_lissajous", plot);
    }

    // ─── 10. Rose curve r = cos(3θ) ───
    {
        let cfg = PlotConfig::new()
            .with_title("Rose Curve r = cos(3θ)")
            .with_dimensions(600, 600);
        let mut plot = SvgPlot::new(cfg).with_theme(seaborn.clone());
        let pts: Vec<DataPoint> = (0..720)
            .map(|i| {
                let theta = i as f64 * PI / 180.0;
                let r = (3.0 * theta).cos();
                DataPoint::new(r * theta.cos(), r * theta.sin())
            })
            .collect();
        plot.add_series(DataSeries::new("Rose", pts));
        save!("10_rose_curve", plot);
    }

    // ─── 11. Error band (mean ± 2σ) ───
    {
        let cfg = PlotConfig::new()
            .with_title("Experimental Data with Confidence Band")
            .with_x_label("Trial")
            .with_y_label("Measurement")
            .with_dimensions(900, 450);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let baseline: Vec<DataPoint> = (0..20)
            .map(|i| DataPoint::new(i as f64, 50.0 + (i as f64 * 0.5).sin() * 5.0))
            .collect();
        let upper: Vec<DataPoint> = (0..20)
            .map(|i| DataPoint::new(i as f64, 50.0 + (i as f64 * 0.5).sin() * 5.0 + 8.0))
            .collect();
        let lower: Vec<DataPoint> = (0..20)
            .map(|i| DataPoint::new(i as f64, 50.0 + (i as f64 * 0.5).sin() * 5.0 - 8.0))
            .collect();
        plot.add_series(DataSeries::new("Mean", baseline));
        plot.add_series(DataSeries::new("+2σ", upper));
        plot.add_series(DataSeries::new("−2σ", lower));
        save!("11_error_bands", plot);
    }

    // ─── 12. Multi-harmonic signal ───
    {
        let cfg = PlotConfig::new()
            .with_title("Fourier Synthesis: f + 2f + 3f")
            .with_x_label("Sample")
            .with_y_label("Amplitude")
            .with_dimensions(900, 450);
        let mut plot = SvgPlot::new(cfg).with_theme(seaborn.clone());
        let f1: Vec<DataPoint> = (0..200)
            .map(|i| DataPoint::new(i as f64, (i as f64 * 0.1).sin()))
            .collect();
        let f2: Vec<DataPoint> = (0..200)
            .map(|i| DataPoint::new(i as f64, 0.5 * (i as f64 * 0.2).sin()))
            .collect();
        let f3: Vec<DataPoint> = (0..200)
            .map(|i| DataPoint::new(i as f64, 0.33 * (i as f64 * 0.3).sin()))
            .collect();
        plot.add_series(DataSeries::new("Fundamental", f1));
        plot.add_series(DataSeries::new("2nd Harmonic", f2));
        plot.add_series(DataSeries::new("3rd Harmonic", f3));
        save!("12_harmonics", plot);
    }

    // ─── 13. Log vs sqrt ───
    {
        let cfg = PlotConfig::new()
            .with_title("Logarithmic vs Square-root Growth")
            .with_dimensions(900, 500);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let ln: Vec<DataPoint> = (1..100)
            .map(|i| DataPoint::new(i as f64, (i as f64).ln()))
            .collect();
        let sq: Vec<DataPoint> = (1..100)
            .map(|i| DataPoint::new(i as f64, (i as f64).sqrt()))
            .collect();
        plot.add_series(DataSeries::new("ln(x)", ln));
        plot.add_series(DataSeries::new("√x", sq));
        save!("13_log_sqrt", plot);
    }

    // ─── 14. Stock price (sinusoidal mock) ───
    {
        let cfg = PlotConfig::new()
            .with_title("Stock Price — 60 Trading Days")
            .with_x_label("Day")
            .with_y_label("Price (USD)")
            .with_dimensions(900, 450);
        let mut plot = SvgPlot::new(cfg).with_theme(seaborn.clone());
        let price: Vec<DataPoint> = (0..60)
            .map(|i| {
                let base = 150.0 + i as f64 * 0.8;
                let noise = (i as f64 * 2.3).sin() * 10.0 + ((i * 7) % 5) as f64;
                DataPoint::new(i as f64, base + noise)
            })
            .collect();
        plot.add_series(DataSeries::new("AAPL", price));
        save!("14_stock", plot);
    }

    // ─── 15. Normal CDF ───
    {
        let cfg = PlotConfig::new()
            .with_title("Standard Normal CDF  Φ(x)")
            .with_x_label("x")
            .with_y_label("P(X ≤ x)")
            .with_dimensions(800, 450);
        let mut plot = SvgPlot::new(cfg).with_theme(theme.clone());
        let pts: Vec<DataPoint> = (0..200)
            .map(|i| {
                let x = (i as f64 - 100.0) * 0.1;
                let t = x / std::f64::consts::SQRT_2;
                // Approximate CDF via tanh (smooth, monotonic, no erf needed)
                let cdf = 0.5 * (1.0 + t.tanh());
                DataPoint::new(x, cdf)
            })
            .collect();
        plot.add_series(DataSeries::new("Φ(x)", pts));
        save!("15_normal_cdf", plot);
    }

    println!("\n✅  Improved plots written to {out}/");
    for entry in std::fs::read_dir(out).unwrap() {
        let e = entry.unwrap();
        let name = e.file_name().to_string_lossy().to_string();
        let size = e.metadata().unwrap().len();
        println!("   {name}  ({size} bytes)");
    }
}
