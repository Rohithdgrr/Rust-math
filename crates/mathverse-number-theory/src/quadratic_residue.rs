pub fn legendre(a: u64, p: u64) -> i64 {
    if p == 2 { return if a % 2 == 0 { 0 } else { 1 }; }
    let ls = crate::modular::mod_pow(a, (p - 1) / 2, p);
    if ls == p - 1 { -1 } else { ls as i64 }
}

pub fn jacobi(mut a: u64, mut n: u64) -> i64 {
    if n == 0 || n % 2 == 0 { return 0; }
    a %= n;
    let mut result = 1i64;
    while a != 0 {
        while a % 2 == 0 { a /= 2; if n % 8 == 3 || n % 8 == 5 { result = -result; } }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 { result = -result; }
        a %= n;
    }
    if n == 1 { result } else { 0 }
}

pub fn tonelli_shanks(n: u64, p: u64) -> Option<u64> {
    if legendre(n, p) != 1 { return None; }
    if p % 4 == 3 { return Some(crate::modular::mod_pow(n, (p + 1) / 4, p)); }
    let mut q = p - 1;
    let mut s = 0u64;
    while q % 2 == 0 { q /= 2; s += 1; }
    let mut z = 2;
    while legendre(z, p) != -1 { z += 1; }
    let mut m = s;
    let mut c = crate::modular::mod_pow(z, q, p);
    let mut t = crate::modular::mod_pow(n, q, p);
    let mut r = crate::modular::mod_pow(n, (q + 1) / 2, p);
    loop {
        if t == 1 { return Some(r); }
        let mut i = 1;
        let mut tt = t;
        while tt != 1 { tt = crate::modular::mod_mul(tt, tt, p); i += 1; if i >= m { return None; } }
        let b = crate::modular::mod_pow(c, 1 << (m - i - 1), p);
        m = i;
        c = crate::modular::mod_mul(b, b, p);
        t = crate::modular::mod_mul(t, c, p);
        r = crate::modular::mod_mul(r, b, p);
    }
}

pub fn quadratic_residues(p: u64) -> Vec<u64> {
    let mut res = Vec::new();
    for x in 0..p { res.push(crate::modular::mod_pow(x, 2, p)); }
    res.sort();
    res.dedup();
    res
}

pub fn is_quadratic_residue(a: u64, p: u64) -> bool { legendre(a, p) == 1 }

pub fn chinese_remainder_quadratic(a: u64, m: u64, n: u64) -> Option<u64> {
    let r1 = tonelli_shanks(a, m)?;
    let r2 = tonelli_shanks(a, n)?;
    crate::modular::crt(&[r1, r2], &[m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legendre_test() {
        assert_eq!(legendre(2, 7), 1);
        assert_eq!(legendre(3, 7), -1);
    }

    #[test]
    fn tonelli_test() {
        let r = tonelli_shanks(2, 7).unwrap();
        assert_eq!(crate::modular::mod_mul(r, r, 7), 2);
    }

    #[test]
    fn residues() {
        let r = quadratic_residues(7);
        assert!(r.contains(&1));
        assert!(r.contains(&2));
        assert!(r.contains(&4));
    }
}
