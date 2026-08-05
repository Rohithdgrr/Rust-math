//! Radar chart example: Skill comparison of two candidates.

use mathverse_plot::radar::{render_radar_chart, RadarConfig, RadarPoint, RadarSeries};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let axes = vec![
        RadarPoint::new("Leadership", 8.0),
        RadarPoint::new("Technical", 9.0),
        RadarPoint::new("Communication", 7.0),
        RadarPoint::new("Teamwork", 8.5),
        RadarPoint::new("Creativity", 6.0),
        RadarPoint::new("Problem Solving", 9.0),
    ];

    let series = vec![
        RadarSeries::new("Alice", axes.clone(), Color::rgb(66, 133, 244)).with_opacity(0.25),
        RadarSeries::new(
            "Bob",
            vec![
                RadarPoint::new("Leadership", 7.0),
                RadarPoint::new("Technical", 6.0),
                RadarPoint::new("Communication", 9.0),
                RadarPoint::new("Teamwork", 7.0),
                RadarPoint::new("Creativity", 8.5),
                RadarPoint::new("Problem Solving", 7.5),
            ],
            Color::rgb(255, 99, 71),
        )
        .with_opacity(0.25),
    ];

    let config = RadarConfig::new()
        .with_radius(150.0)
        .with_center(300.0, 220.0);

    let mut cfg = config;
    cfg.plot_config.title = "Candidate Skills".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 450;

    let svg = render_radar_chart(&series, &cfg)?;
    PlotSaver::new(svg).save_png("radar.png")?;
    println!("wrote radar.png");

    Ok(())
}
