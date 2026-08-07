#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() == 3 {
        let a = data[0] as f64;
        let b = data[1] as f64;
        let c = data[2] as f64;
        let _ = mathverse_equations::polynomial::solve_quadratic(a, b, c);
    }
});
