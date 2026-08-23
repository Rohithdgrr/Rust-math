//! Colorbar example with heatmap.

use mathverse_plot::color::viridis;
use mathverse_plot::save::PlotSaver;
use mathverse_plot::{render_colorbar, ColorbarConfig, PlotConfig, SvgPlot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple heatmap
    let grid: Vec<Vec<f64>> = (0..10)
        .map(|i| {
            (0..10)
                .map(|j| {
                    let x = f64::from(i) / 9.0;
                    let y = f64::from(j) / 9.0;
                    (x * y * 100.0).sin()
                })
                .collect()
        })
        .collect();

    let config = PlotConfig::new().with_title("Heatmap with Colorbar");

    let mut plot = SvgPlot::new(config);
    plot.add_heatmap("data", grid, viridis)?;

    // Generate the plot SVG
    let mut svg = plot.generate();

    // Add colorbar
    let colorbar_config = ColorbarConfig::new()
        .with_dimensions(20.0, 200.0)
        .with_title("Value");

    let colorbar_svg = render_colorbar(
        750.0, // x position (right side of plot)
        50.0,  // y position
        -1.0,  // data min
        1.0,   // data max
        viridis,
        &colorbar_config,
    );

    // Insert colorbar before closing </svg>
    svg = svg.replace("</svg>", &format!("{colorbar_svg}\n</svg>"));

    PlotSaver::new(&svg).save_png("colorbar.png")?;
    println!("Wrote colorbar.svg ({} bytes)", svg.len());

    Ok(())
}
