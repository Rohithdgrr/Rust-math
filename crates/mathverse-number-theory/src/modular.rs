pub fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 { return 0; }
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 { result = result * base % m; }
        base = base * base % m;
        exp >>= 1;
    }
    result
}

pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let (mut r0, mut r1) = (m as i128, a as i128 % m as i128);
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 { let q = r0 / r1; (r0, r1) = (r1, r0 - q * r1); (t0, t1) = (t1, t0 - q * t1); }
    if r0 != 1 { None } else { Some(t0.rem_euclid(m as i128) as u64) }
}

pub fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    let (ra, rb) = (a % m, b % m);
    // Both ra and rb are < m, so ra + rb <= 2*(m-1) which fits in u64.
    let sum = ra + rb;
    if sum >= m { sum - m } else { sum }
}

pub fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    let (ra, rb) = (a % m, b % m);
    if ra >= rb { ra - rb } else { m - (rb - ra) }
}

pub fn mod_mul(a: u64, b: u64, m: u64) -> u64 { (a as u128 * b as u128 % m as u128) as u64 }

pub fn mod_div(a: u64, b: u64, m: u64) -> Option<u64> { mod_inverse(b, m).map(|bi| mod_mul(a, bi, m)) }

pub fn crt(rems: &[u64], mods: &[u64]) -> Option<u64> {
    let (mut result, mut lcm) = (rems[0] as i128, mods[0] as i128);
    for i in 1..rems.len() {
        let (a1, m1) = (result, lcm);
        let (a2, m2) = (rems[i] as i128, mods[i] as i128);
        let g = gcd_i128(m1, m2);
        if (a2 - a1) % g != 0 { return None; }
        let (_g, s, _) = extended_gcd_i128(m1 / g, m2 / g);
        result = a1 + m1 * ((a2 - a1) / g * s % (m2 / g));
        lcm = lcm / g * m2;
        result = result.rem_euclid(lcm);
    }
    Some(result as u64)
}

fn gcd_i128(a: i128, b: i128) -> i128 { if b == 0 { a.abs() } else { gcd_i128(b, a % b) } }

fn extended_gcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 { return (b, 0, 1); }
    let (g, x1, y1) = extended_gcd_i128(b % a, a);
    (g, y1 - (b / a) * x1, x1)
}

pub fn wietschke(a: u64, b: u64, m: u64) -> Option<u64> {
    let g = crate::factorization::prime_factors(m).into_iter().fold(1u64, |acc, p| {
        if b % p == 0 { acc } else { acc * p }
    });
    if a % g != 0 { None } else { mod_div(a, b, m) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 13, 7), 3);
    }

    #[test]
    fn inverse() {
        assert_eq!(mod_inverse(3, 11), Some(4));
        assert_eq!(mod_inverse(4, 6), None);
    }

    #[test]
    fn crt_test() {
        assert_eq!(crt(&[2, 3], &[3, 5]), Some(8));
        assert_eq!(crt(&[1, 2], &[2, 4]), None);
    }

    #[test]
    fn arithmetic() {
        assert_eq!(mod_add(5, 7, 3), 0);
        assert_eq!(mod_sub(2, 5, 7), 4);
        assert_eq!(mod_mul(3, 4, 5), 2);
    }

    #[test]
    fn mod_add_overflow() {
        assert_eq!(mod_add(u64::MAX, 1, 100), (u64::MAX % 100 + 1) % 100);
        assert_eq!(mod_add(u64::MAX, u64::MAX, 7), (u64::MAX % 7 * 2) % 7);
        assert_eq!(mod_add(0, 0, 1), 0);
    }

    #[test]
    fn mod_sub_underflow() {
        assert_eq!(mod_sub(0, 1, 7), 6);
        assert_eq!(mod_sub(3, 5, 7), 5);
        assert_eq!(mod_sub(u64::MAX, 0, 99), u64::MAX % 99);
    }
}
