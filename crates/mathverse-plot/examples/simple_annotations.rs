//! Annotation system example with text, arrows, and reference lines.

use mathverse_plot::{
    Annotations, Arrow, DataPoint, DataSeries, PlotConfig, ReferenceLine, SvgPlot,
    TextAnnotation,
};
use mathverse_plot::style::Color;
use mathverse_plot::save::PlotSaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some data
    let points: Vec<DataPoint> = (0..=100)
        .map(|i| {
            let x = i as f64 * 0.1;
            DataPoint::new(x, x.sin())
        })
        .collect();

    let config = PlotConfig::new()
        .with_title("Sine Wave with Annotations")
        .with_x_label("x (radians)")
        .with_y_label("sin(x)");

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("sin(x)".to_string(), points));

    // Generate the base SVG
    let mut svg = plot.generate();

    // Create annotations
    let annotations = Annotations::new()
        // Add a text annotation at the peak
        .add_text(
            TextAnnotation::new(DataPoint::new(1.57, 1.0), "Peak")
                .with_bold()
                .with_font_size(14.0)
                .with_color(Color::RED)
                .with_offset(10.0, -10.0),
        )
        // Add an arrow pointing to the peak
        .add_arrow(
            Arrow::new(DataPoint::new(2.5, 0.8), DataPoint::new(1.57, 1.0))
                .with_color(Color::BLUE)
                .with_width(2.0),
        )
        // Add a horizontal reference line at y=0
        .add_line(
            ReferenceLine::horizontal(0.0)
                .with_color(Color::GRAY)
                .with_dash("5,5"),
        )
        // Add a vertical reference line at x=π
        .add_line(
            ReferenceLine::vertical(std::f64::consts::PI)
                .with_color(Color::GREEN)
                .with_label("π"),
        );

    // Render annotations as SVG (simplified - in real use, this would be integrated)
    let mut annotation_svg = String::new();

    // Text annotation
    for text in &annotations.texts {
        let (dx, dy) = text.position.data_xy().unwrap_or((0.0, 0.0));
        let x = 50.0 + dx * 70.0; // Simplified scaling
        let y = 300.0 - dy * 250.0;
        annotation_svg.push_str(&format!(
            r#"  <text x="{}" y="{}" font-size="{}" fill="{}" font-weight="{}">{}</text>"#,
            x + text.x_offset,
            y + text.y_offset,
            text.font_size,
            text.color.to_hex(),
            text.font_weight,
            mathverse_plot::common::xml_escape(&text.text)
        ));
        annotation_svg.push('\n');
    }

    // Arrows
    for arrow in &annotations.arrows {
        let (x1d, y1d) = arrow.from.data_xy().unwrap_or((0.0, 0.0));
        let (x2d, y2d) = arrow.to.data_xy().unwrap_or((0.0, 0.0));
        let x1 = 50.0 + x1d * 70.0;
        let y1 = 300.0 - y1d * 250.0;
        let x2 = 50.0 + x2d * 70.0;
        let y2 = 300.0 - y2d * 250.0;
        annotation_svg.push_str(&format!(
            r#"  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{}" stroke-width="{}" marker-end="url(#arrowhead)"/>"#,
            arrow.color.to_hex(),
            arrow.width
        ));
        annotation_svg.push('\n');
    }

    // Reference lines
    for line in &annotations.lines {
        match line.orientation {
            mathverse_plot::annotations::LineOrientation::Horizontal => {
                let y = 300.0 - line.position * 250.0;
                annotation_svg.push_str(&format!(
                    r#"  <line x1="50" y1="{y}" x2="750" y2="{y}" stroke="{}" stroke-width="{}" stroke-dasharray="{}"/>"#,
                    line.color.to_hex(),
                    line.width,
                    line.dash.as_deref().unwrap_or("none")
                ));
            }
            mathverse_plot::annotations::LineOrientation::Vertical => {
                let x = 50.0 + line.position * 70.0;
                annotation_svg.push_str(&format!(
                    r#"  <line x1="{x}" y1="50" x2="{x}" y2="300" stroke="{}" stroke-width="{}" stroke-dasharray="{}"/>"#,
                    line.color.to_hex(),
                    line.width,
                    line.dash.as_deref().unwrap_or("none")
                ));
            }
        }
        annotation_svg.push('\n');
    }

    // Add arrow marker definition
    let marker_def = r#"  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="blue"/>
    </marker>
  </defs>"#;

    svg = svg.replace("<svg", &format!("<svg\n{}", marker_def));
    svg = svg.replace("</svg>", &format!("{}\n</svg>", annotation_svg));

    PlotSaver::new(&svg).save_png("annotations.png")?;
    println!("Wrote annotations.svg ({} bytes)", svg.len());

    Ok(())
}
