//! Model registry: save/load tensors to text files, parameter counting.

use std::fs;
use std::io::{self, Write, BufWriter, BufRead, BufReader};
use crate::tensor::Tensor;

/// Save named tensors to a text file.
/// Format per tensor: name shape dim0 dim1 ... \n val0 val1 ...
pub fn save_model(path: &str, params: &[(String, Tensor)]) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for (name, t) in params {
        write!(writer, "{}", name)?;
        for &d in &t.shape {
            write!(writer, " {}", d)?;
        }
        writeln!(writer)?;
        for (i, v) in t.data.iter().enumerate() {
            if i > 0 { write!(writer, " ")?; }
            write!(writer, "{}", v)?;
        }
        writeln!(writer)?;
    }
    writer.flush()
}

/// Load named tensors from a text file.
pub fn load_model(path: &str) -> io::Result<Vec<(String, Tensor)>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut result = Vec::new();

    while let Some(header_res) = lines.next() {
        let header = header_res?;
        let header = header.trim().to_string();
        if header.is_empty() { continue; }
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() < 2 { continue; }
        let name = parts[0].to_string();
        let shape: Vec<usize> = parts[1..].iter()
            .filter_map(|s| s.parse().ok())
            .collect();

        let data_line = match lines.next() {
            Some(Ok(line)) => line,
            _ => break,
        };
        let data: Vec<f64> = data_line.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if let Ok(t) = Tensor::from_vec(&shape, data) {
            result.push((name, t));
        }
    }
    Ok(result)
}

/// Count total parameters across all tensors.
pub fn count_parameters(params: &[(String, Tensor)]) -> usize {
    params.iter().map(|(_, t)| t.numel()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn save_and_load_roundtrip() {
        let tmp = std::env::temp_dir().join("mathverse_test_model.txt");
        let path_str = tmp.to_str().unwrap();

        let params = vec![
            ("layer1.weight".to_string(), Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
            ("layer1.bias".to_string(), Tensor::new(&[3], &[0.1, 0.2, 0.3]).unwrap()),
        ];

        save_model(path_str, &params).unwrap();
        let loaded = load_model(path_str).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "layer1.weight");
        assert_eq!(loaded[0].1.shape, vec![2, 3]);
        assert!((loaded[0].1.data[0] - 1.0).abs() < 1e-12);
        assert_eq!(loaded[1].0, "layer1.bias");
        assert_eq!(loaded[1].1.shape, vec![3]);

        fs::remove_file(path_str).ok();
    }

    #[test]
    fn count_parameters_test() {
        let params = vec![
            ("a".to_string(), Tensor::zeros(&[10, 10])),
            ("b".to_string(), Tensor::zeros(&[5])),
        ];
        assert_eq!(count_parameters(&params), 105);
    }

    #[test]
    fn save_load_scalar_tensor() {
        let tmp = std::env::temp_dir().join("mathverse_test_scalar.txt");
        let path_str = tmp.to_str().unwrap();

        let params = vec![
            ("scalar".to_string(), Tensor::new(&[1], &[42.0]).unwrap()),
        ];
        save_model(path_str, &params).unwrap();
        let loaded = load_model(path_str).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!((loaded[0].1.data[0] - 42.0).abs() < 1e-12);

        fs::remove_file(path_str).ok();
    }
}
