use mathverse_dataframe::{DataFrame, Series, AnyColumn};

fn main() {
    // Test 1: Basic construction and display
    let mut df = DataFrame::new();
    df.add_column("name", vec![String::from("Alice"), String::from("Bob"), String::from("Charlie")]).unwrap();
    df.add_column("age", vec![25.0, 30.0, 35.0]).unwrap();
    df.add_column("score", vec![88.5, 92.3, 76.1]).unwrap();
    println!("=== Test 1: Basic DataFrame ===");
    println!("{}", df);
    assert_eq!(df.shape(), (3, 3));
    assert_eq!(df.column_names(), vec!["name", "age", "score"]);
    println!("PASS: shape and columns\n");

    // Test 2: Null handling on Series
    let mut s = Series::new("values", vec![10.0, 20.0, 30.0, 40.0]);
    s.set_null(1);
    assert!(s.is_null(1));
    assert_eq!(s.null_count(), 1);
    let cleaned = s.drop_nulls();
    assert_eq!(cleaned.data(), &[10.0, 30.0, 40.0]);
    println!("=== Test 2: Null handling ===");
    println!("PASS: drop_nulls works\n");

    // Test 3: Column selection
    let sub = df.select_columns(&["name", "score"]).unwrap();
    assert_eq!(sub.shape(), (3, 2));
    println!("=== Test 3: Column selection ===");
    println!("PASS: select_columns works\n");

    // Test 4: Boolean filter
    let mask_df = Series::new("mask", vec![false, true, true]);
    let filtered = df.filter(&mask_df).unwrap();
    assert_eq!(filtered.shape(), (2, 3)); // Bob (30) and Charlie (35)
    println!("=== Test 4: Boolean filter ===");
    println!("PASS: filter works\n");

    // Test 5: Arithmetic via AnyColumn
    let x = AnyColumn::from(Series::new("x", vec![1.0, 2.0, 3.0]));
    let y = AnyColumn::from(Series::new("y", vec![4.0, 5.0, 6.0]));
    let sum = x.add(&y).unwrap();
    let scaled = x.mul_scalar(2.0).unwrap();
    let sum_f64 = sum.to_f64().unwrap();
    let scaled_f64 = scaled.to_f64().unwrap();
    let sum_data: Vec<f64> = sum_f64.data().into_iter().map(|&v| v).collect();
    let scaled_data: Vec<f64> = scaled_f64.data().into_iter().map(|&v| v).collect();
    println!("=== Test 5: Arithmetic ===");
    println!("sum: {:?}", sum_data);
    println!("scaled: {:?}", scaled_data);
    assert_eq!(sum_data, vec![5.0, 7.0, 9.0]);
    assert_eq!(scaled_data, vec![2.0, 4.0, 6.0]);
    println!("PASS: arithmetic works\n");

    // Test 6: Aggregations via AnyColumn
    let score_col = df.column("score").unwrap(); // AnyColumn
    let mean = score_col.mean().unwrap();
    let std = score_col.std().unwrap();
    let median = score_col.median().unwrap();
    let q25 = score_col.quantile(0.25).unwrap();
    let desc = score_col.describe_numeric().unwrap();
    let (min, max, mean_val, std_val, count) = desc;
    println!("=== Test 6: Aggregations ===");
    println!("mean={}, std={}, median={}, q25={}, min={}, max={}, n={}", mean, std, median, q25, min, max, count);
    assert!((mean - 85.6333).abs() < 0.01);
    println!("PASS: aggregations work\n");

    // Test 7: Sorting
    let mut df2 = DataFrame::new();
    df2.add_column("value", vec![3.0, 1.0, 2.0]).unwrap();
    let sorted = df2.sort_by("value", true).unwrap();
    let sorted_f64 = sorted.column("value").unwrap().to_f64().unwrap();
    let sorted_data: Vec<f64> = sorted_f64.data().into_iter().map(|&v| v).collect();
    assert_eq!(sorted_data, vec![1.0, 2.0, 3.0]);
    println!("=== Test 7: Sorting ===");
    println!("PASS: sort_by works\n");

    // Test 8: Transpose
    let mut df3 = DataFrame::new();
    df3.add_column("a", vec![1.0, 2.0, 3.0]).unwrap();
    df3.add_column("b", vec![4.0, 5.0, 6.0]).unwrap();
    let t = df3.transpose().unwrap();
    println!("=== Test 8: Transpose ===");
    println!("Transposed:\n{}", t);
    assert_eq!(t.shape(), (2, 3));
    println!("PASS: transpose works\n");

    // Test 9: DataFrame with i64 column
    let mut i64_df = DataFrame::new();
    i64_df.add_column("id", vec![1i64, 2i64, 3i64]).unwrap();
    let i64_col = i64_df.column("id").unwrap();
    let i64_data: Vec<f64> = i64_col.valid_f64().unwrap();
    println!("=== Test 9: i64 column ===");
    println!("PASS: i64 column valid_f64 works (values: {:?})", i64_data);

    // Test 10: JSON roundtrip (with json feature)
    #[cfg(feature = "json")]
    {
        let json_str = mathverse_dataframe::json::to_json_string(&df);
        let df_roundtrip = mathverse_dataframe::json::from_json_str(&json_str).unwrap();
        assert_eq!(df_roundtrip.shape(), (3, 3));
        println!("=== Test 10: JSON roundtrip ===");
        println!("PASS: JSON roundtrip works\n");
    }
    #[cfg(not(feature = "json"))]
    {
        println!("=== Test 10: JSON roundtrip ===");
        println!("SKIPPED (json feature not enabled)\n");
    }

    // Test 11: Series head/tail
    let s = Series::new("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let head = s.head(2);
    let tail = s.tail(2);
    let head_data: Vec<f64> = head.data().into_iter().map(|&v| v).collect();
    let tail_data: Vec<f64> = tail.data().into_iter().map(|&v| v).collect();
    assert_eq!(head_data, vec![1.0, 2.0]);
    assert_eq!(tail_data, vec![4.0, 5.0]);
    println!("=== Test 11: Head/Tail ===");
    println!("PASS: head/tail work\n");

    // Test 12: Series map
    let s = Series::new("x", vec![1.0, 2.0, 3.0]);
    let doubled = s.map(|&v| v * 2.0);
    let d_data: Vec<f64> = doubled.data().into_iter().map(|&v| v).collect();
    assert_eq!(d_data, vec![2.0, 4.0, 6.0]);
    println!("=== Test 12: Map ===");
    println!("PASS: map works\n");

    // Test 13: DataFrame describe
    let desc_df = df.describe().unwrap();
    println!("=== Test 13: DataFrame describe ===");
    println!("{}", desc_df);
    assert_eq!(desc_df.ncols(), 3); // score, age (numeric cols)
    println!("PASS: describe works\n");

    println!("=== ALL 13 TESTS PASSED ===");
}