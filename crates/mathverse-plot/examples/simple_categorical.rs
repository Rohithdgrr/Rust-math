//! Categorical axis example with string labels.

use mathverse_plot::{CategoryMap, DataPoint, DataSeries, PlotConfig, SvgPlot};
use mathverse_plot::style::PlotStyle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create categorical data
    let categories = CategoryMap::from_labels(vec!["Apple", "Banana", "Cherry", "Date", "Elderberry"]);

    // Data: fruit sales
    let sales = vec![45.0, 32.0, 58.0, 21.0, 39.0];

    // Create series with categorical x positions
    let points: Vec<DataPoint> = sales
        .iter()
        .enumerate()
        .map(|(i, &y)| DataPoint::new(i as f64, y))
        .collect();

    let config = PlotConfig::new()
        .with_title("Fruit Sales")
        .with_x_label("Fruit Type")
        .with_y_label("Sales ($)");

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("Sales".to_string(), points));

    // Generate SVG
    let svg = plot.generate();

    // Add category labels to the SVG (simplified - in real use, this would be integrated)
    let mut labeled_svg = svg.clone();

    // Insert category labels before the closing </svg> tag
    let category_labels: String = categories
        .labels()
        .iter()
        .enumerate()
        .map(|(i, label)| {
            format!(
                r#"  <text x="{}" y="{}" text-anchor="middle" font-size="11">{}</text>"#,
                50.0 + i as f64 * 150.0, // Simplified positioning
                580.0,
                label
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    labeled_svg = labeled_svg.replace("</svg>", &format!("{}\n</svg>", category_labels));

    PlotSaver::new(labeled_svg).save_png("categorical.png")?;
    println!("Wrote categorical.svg ({} bytes)", labeled_svg.len());

    Ok(())
}
