//! Marimekko chart example: Market segments by company size.

use mathverse_plot::marimekko::{render_marimekko, MarimekkoColumn, MarimekkoConfig, MarimekkoSegment};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let columns = vec![
        MarimekkoColumn::new(
            "Startup",
            30.0,
            vec![
                MarimekkoSegment::new("SaaS", 40.0, Color::rgb(66, 133, 244)),
                MarimekkoSegment::new("Hardware", 15.0, Color::rgb(255, 150, 50)),
                MarimekkoSegment::new("Services", 10.0, Color::rgb(0, 200, 83)),
            ],
        ),
        MarimekkoColumn::new(
            "Mid-Market",
            40.0,
            vec![
                MarimekkoSegment::new("SaaS", 50.0, Color::rgb(66, 133, 244)),
                MarimekkoSegment::new("Hardware", 20.0, Color::rgb(255, 150, 50)),
                MarimekkoSegment::new("Services", 25.0, Color::rgb(0, 200, 83)),
            ],
        ),
        MarimekkoColumn::new(
            "Enterprise",
            50.0,
            vec![
                MarimekkoSegment::new("SaaS", 35.0, Color::rgb(66, 133, 244)),
                MarimekkoSegment::new("Hardware", 30.0, Color::rgb(255, 150, 50)),
                MarimekkoSegment::new("Services", 40.0, Color::rgb(0, 200, 83)),
            ],
        ),
    ];

    let config = MarimekkoConfig::new();

    let mut cfg = config;
    cfg.plot_config.title = "Market Segments by Company Size".into();
    cfg.plot_config.width = 700;
    cfg.plot_config.height = 450;

    let svg = render_marimekko(&columns, &cfg)?;
    PlotSaver::new(svg).save_png("marimekko.png")?;
    println!("wrote marimekko.png");

    Ok(())
}
