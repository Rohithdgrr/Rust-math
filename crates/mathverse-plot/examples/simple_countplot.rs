//! Count plot example.

use mathverse_plot::{render_countplot, CountConfig, PlotSaver};

fn main() -> mathverse_plot::PlotResult<()> {
    let categories = vec![
        "Cat", "Dog", "Bird", "Cat", "Dog", "Cat",
        "Fish", "Dog", "Cat", "Bird", "Dog", "Cat",
        "Fish", "Fish", "Dog", "Cat", "Bird", "Cat",
    ];

    let cats: Vec<String> = categories.iter().map(|s| s.to_string()).collect();

    let mut config = CountConfig::new();
    config.plot_config = config.plot_config
        .with_title("Count Plot: Pet Ownership");

    let svg = render_countplot(&cats, &config)?;
PlotSaver::new(&svg).save_png("countplot.png")?;
println!("wrote countplot.png");

    Ok(())
}
