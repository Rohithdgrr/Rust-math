//! FacetGrid / trellis plots example.
//!
//! Demonstrates how to create multi-panel conditioned visualizations.

use mathverse_plot::facet::{FacetBuilder, FacetData, FacetGrid, FacetScale, FacetWrap};
use mathverse_plot::style::Color;

fn main() {
    // Create sample data with categorical variables
    let data = vec![
        FacetData::new(1.0, 10.0).with_column("group", "A").with_column("time", "T1"),
        FacetData::new(2.0, 20.0).with_column("group", "B").with_column("time", "T1"),
        FacetData::new(3.0, 15.0).with_column("group", "A").with_column("time", "T2"),
        FacetData::new(4.0, 25.0).with_column("group", "B").with_column("time", "T2"),
        FacetData::new(1.5, 12.0).with_column("group", "A").with_column("time", "T1"),
        FacetData::new(2.5, 22.0).with_column("group", "B").with_column("time", "T1"),
        FacetData::new(3.5, 18.0).with_column("group", "A").with_column("time", "T2"),
        FacetData::new(4.5, 28.0).with_column("group", "B").with_column("time", "T2"),
    ];

    println!("Sample data: {} points", data.len());

    // Get unique values
    let groups = FacetGrid::unique_values(&data, "group");
    let times = FacetGrid::unique_values(&data, "time");
    println!("Groups: {:?}", groups);
    println!("Times: {:?}", times);

    // Create a simple column facet grid
    let grid_col = FacetGrid::new()
        .col("group")
        .with_title("Faceted by Group");

    println!("\nColumn facet grid:");
    let (rows, cols) = grid_col.grid_dims(&data);
    println!("  Grid dimensions: {} rows x {} cols", rows, cols);
    println!("  Total panels: {}", grid_col.total_panels(&data));

    // Create a row facet grid
    let grid_row = FacetGrid::new()
        .row("time")
        .with_title("Faceted by Time");

    println!("\nRow facet grid:");
    let (rows, cols) = grid_row.grid_dims(&data);
    println!("  Grid dimensions: {} rows x {} cols", rows, cols);

    // Create a 2D facet grid (row x col)
    let grid_2d = FacetGrid::new()
        .col("group")
        .row("time")
        .with_title("2D Facet Grid")
        .with_panel_size(150.0, 100.0);

    println!("\n2D facet grid:");
    let (rows, cols) = grid_2d.grid_dims(&data);
    println!("  Grid dimensions: {} rows x {} cols", rows, cols);

    // Create a wrapped facet grid
    let grid_wrap = FacetGrid::new()
        .col("group")
        .wrap(FacetWrap::Columns(2))
        .with_title("Wrapped Grid");

    println!("\nWrapped facet grid:");
    let (rows, cols) = grid_wrap.grid_dims(&data);
    println!("  Grid dimensions: {} rows x {} cols", rows, cols);

    // Create with free scales
    let grid_free = FacetGrid::new()
        .col("group")
        .scale(FacetScale::Free)
        .with_title("Free Scales");

    println!("\nFree scales grid:");
    println!("  Scale: {:?}", grid_free.scale);

    // Render SVG
    let svg = grid_2d.render_svg(&data);
    println!("\nSVG output: {} bytes", svg.len());
    println!("SVG preview (first 300 chars):");
    println!("{}", &svg[..300.min(svg.len())]);

    // Use builder presets
    let grid_from_builder = FacetBuilder::col("group");
    println!("\nBuilder preset:");
    println!("  Col var: {:?}", grid_from_builder.col_var);

    let grid_from_builder2 = FacetBuilder::grid("group", "time");
    println!("  Grid builder: col={:?}, row={:?}", grid_from_builder2.col_var, grid_from_builder2.row_var);

    // Test subset filtering
    let subset_a = FacetGrid::subset(&data, Some("group"), Some("A"), None, None);
    println!("\nSubset filtering:");
    println!("  Group A: {} points", subset_a.len());

    let subset_b_t1 = FacetGrid::subset(&data, Some("group"), Some("B"), Some("time"), Some("T1"));
    println!("  Group B, Time T1: {} points", subset_b_t1.len());

    // Convert to DataSeries
    let series = FacetGrid::to_series(&data);
    println!("\nConverted to DataSeries: {} points", series.points.len());

    println!("\nFacetGrid example complete!");
}
