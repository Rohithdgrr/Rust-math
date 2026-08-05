use mathverse_plot::*;

fn main() {
    std::fs::create_dir_all("target/verify_png").unwrap();

    // 1. Titled line plot via resvg path (PlotSaver::save_png)
    {
        let config = PlotConfig::new()
            .with_title("Test Title Here")
            .with_x_label("X Axis Label")
            .with_y_label("Y Axis Label")
            .with_dimensions(600, 400);
        let mut plot = SvgPlot::new(config);
        let xs: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let pts: Vec<DataPoint> = xs.iter().zip(ys.iter()).map(|(x, y)| DataPoint::new(*x, *y)).collect();
        plot.add_series(DataSeries::new("sin(x)".to_string(), pts));
        let svg = plot.generate();
        let r = PlotSaver::new(&svg).save_png("target/verify_png/titled_line");
        println!("titled_line: {:?}", r.success);
        PlotSaver::new(svg).save_png("target/verify_png/titled_line.png").unwrap();
    }

    // 2. Heatmap via resvg path
    {
        let config = PlotConfig::new()
            .with_title("Heatmap")
            .with_dimensions(400, 300);
        let mut plot = SvgPlot::new(config);
        let grid = vec![
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![1.0, 0.75, 0.5, 0.25, 0.0],
            vec![0.2, 0.4, 0.6, 0.8, 0.3],
            vec![0.9, 0.1, 0.7, 0.2, 0.5],
        ];
        plot.add_heatmap("hm", grid, viridis).unwrap();
        let svg = plot.generate();
        let r = PlotSaver::new(&svg).save_png("target/verify_png/heatmap");
        println!("heatmap: {:?}", r.success);
        PlotSaver::new(svg).save_png("target/verify_png/heatmap.png").unwrap();
    }

    // 3. Log scale via resvg path
    {
        let config = PlotConfig::new()
            .with_title("Log")
            .with_y_scale(Scale::Log)
            .with_dimensions(600, 400);
        let mut plot = SvgPlot::new(config);
        let xs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let pts: Vec<DataPoint> = xs.iter().zip(ys.iter()).map(|(x, y)| DataPoint::new(*x, *y)).collect();
        plot.add_series(DataSeries::new("x^2".to_string(), pts));
        let svg = plot.generate();
        let r = PlotSaver::new(&svg).save_png("target/verify_png/log");
        println!("log: {:?}", r.success);
        PlotSaver::new(svg).save_png("target/verify_png/log.png").unwrap();
    }
}
