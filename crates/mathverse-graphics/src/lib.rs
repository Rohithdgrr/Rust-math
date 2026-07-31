//! 2D/3D graphics math: affine matrices (row-vector convention),
//! quaternions, look-at and perspective projection.

use mathverse_matrix::Matrix;
use mathverse_vector::Vector;

pub mod quat;

/// 3D translation as a 4×4 affine matrix (column convention: `p' = T·p`).
pub fn translation(tx: f64, ty: f64, tz: f64) -> Matrix {
    Matrix::from_rows(&[
        &[1.0, 0.0, 0.0, tx],
        &[0.0, 1.0, 0.0, ty],
        &[0.0, 0.0, 1.0, tz],
        &[0.0, 0.0, 0.0, 1.0],
    ]).unwrap()
}

/// 3D rotation about the X axis, angle in radians.
pub fn rotation_x(a: f64) -> Matrix {
    let (s, c) = a.sin_cos();
    Matrix::from_rows(&[
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0, c, -s, 0.0],
        &[0.0, s, c, 0.0],
        &[0.0, 0.0, 0.0, 1.0],
    ]).unwrap()
}
/// 3D rotation about the Y axis.
pub fn rotation_y(a: f64) -> Matrix {
    let (s, c) = a.sin_cos();
    Matrix::from_rows(&[
        &[c, 0.0, s, 0.0],
        &[0.0, 1.0, 0.0, 0.0],
        &[-s, 0.0, c, 0.0],
        &[0.0, 0.0, 0.0, 1.0],
    ]).unwrap()
}
/// 3D rotation about the Z axis.
pub fn rotation_z(a: f64) -> Matrix {
    let (s, c) = a.sin_cos();
    Matrix::from_rows(&[
        &[c, -s, 0.0, 0.0],
        &[s, c, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 0.0],
        &[0.0, 0.0, 0.0, 1.0],
    ]).unwrap()
}

/// Uniform 3D scale.
pub fn scale(s: f64) -> Matrix {
    Matrix::from_rows(&[
        &[s, 0.0, 0.0, 0.0],
        &[0.0, s, 0.0, 0.0],
        &[0.0, 0.0, s, 0.0],
        &[0.0, 0.0, 0.0, 1.0],
    ]).unwrap()
}

/// 2D affine transform matrix (3×3, homogeneous, column convention).
pub fn transform2d(a: f64, tx: f64, ty: f64, s: f64) -> Matrix {
    let (sn, cs) = a.sin_cos();
    Matrix::from_rows(&[
        &[s * cs, -s * sn, tx],
        &[s * sn, s * cs, ty],
        &[0.0, 0.0, 1.0],
    ]).unwrap()
}

/// Apply an affine matrix to a 3D point (homogeneous divide).
pub fn apply(m: &Matrix, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    assert_eq!((m.rows, m.cols), (4, 4));
    let p = m.mul_vec(&Vector::new(vec![x, y, z, 1.0])).unwrap();
    (p.get(0) / p.get(3), p.get(1) / p.get(3), p.get(2) / p.get(3))
}

/// Camera-space matrix `look_at(eye, target, up)` (column convention).
pub fn look_at(eye: [f64; 3], target: [f64; 3], up: [f64; 3]) -> mathverse_core::error::MathResult<Matrix> {
    let f = Vector::new(vec![target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]).normalized()?;
    let upv = Vector::new(up.to_vec()).normalized()?;
    let s = f.cross3(&upv)?.normalized()?;
    let u = s.cross3(&f)?;
    // columns: s, u, -f
    let rot = Matrix::from_rows(&[
        &[s.get(0), u.get(0), -f.get(0), 0.0],
        &[s.get(1), u.get(1), -f.get(1), 0.0],
        &[s.get(2), u.get(2), -f.get(2), 0.0],
        &[0.0, 0.0, 0.0, 1.0],
    ])?;
    rot.mul(&translation(-eye[0], -eye[1], -eye[2]))
}

/// Perspective projection matrix (column convention, right-handed, depth in [0,1]).
/// `fovy` vertical field of view, `aspect = w/h`, near/far planes.
pub fn perspective(fovy: f64, aspect: f64, near: f64, far: f64) -> Matrix {
    let f = 1.0 / (fovy / 2.0).tan();
    Matrix::from_rows(&[
        &[f / aspect, 0.0, 0.0, 0.0],
        &[0.0, f, 0.0, 0.0],
        &[0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
        &[0.0, 0.0, -1.0, 0.0],
    ]).unwrap()
}

/// 2D rotation of a point around the origin.
pub fn rotate2d(x: f64, y: f64, a: f64) -> (f64, f64) {
    let (s, c) = a.sin_cos();
    (c * x - s * y, s * x + c * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_roundtrip() {
        let (x, y, z) = (1.0, 2.0, 3.0);
        // translation then inverse translation
        let t = translation(5.0, -1.0, 2.0);
        let ti = translation(-5.0, 1.0, -2.0);
        let p = apply(&t, x, y, z);
        let back = apply(&ti, p.0, p.1, p.2);
        assert!((back.0 - x).abs() < 1e-12 && (back.1 - y).abs() < 1e-12 && (back.2 - z).abs() < 1e-12);
        // rotate then rotate back
        let r = rotation_z(core::f64::consts::FRAC_PI_2);
        let ri = rotation_z(-core::f64::consts::FRAC_PI_2);
        let p = apply(&r, 1.0, 0.0, 0.0);
        assert!(p.0.abs() < 1e-12 && (p.1 - 1.0).abs() < 1e-12);
        let back = apply(&ri, p.0, p.1, p.2);
        assert!((back.0 - 1.0).abs() < 1e-12 && back.1.abs() < 1e-12);
        // rotate2d
        let (rx, ry) = rotate2d(1.0, 0.0, core::f64::consts::FRAC_PI_2);
        assert!(rx.abs() < 1e-12 && (ry - 1.0).abs() < 1e-12);
    }

    #[test]
    fn look_at_orients() {
        let m = look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]).unwrap();
        // a point straight ahead should land on -z axis... in row-vector
        // convention forward is -f; camera at origin looking down -z:
        // point (0,0,-1) stays (0,0,-1)
        let p = apply(&m, 0.0, 0.0, -1.0);
        assert!((p.2 + 1.0).abs() < 1e-9 && p.0.abs() < 1e-9 && p.1.abs() < 1e-9);
        // degenerate eye == target errors
        assert!(look_at([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]).is_err());
    }

    #[test]
    fn projection_basics() {
        let p = perspective(core::f64::consts::FRAC_PI_2, 2.0, 0.1, 100.0);
        // point at center of view stays centered: (0, 0, -5)
        let v = p.mul_vec(&Vector::new(vec![0.0, 0.0, -5.0, 1.0])).unwrap();
        let w = v.get(3);
        let (ndc_x, ndc_y) = (v.get(0) / w, v.get(1) / w);
        assert!(ndc_x.abs() < 1e-9 && ndc_y.abs() < 1e-9);
        // point at the right edge of the frustum: x/z = aspect·tan(fovy/2) = 2 -> ndc x = 1
        let v = p.mul_vec(&Vector::new(vec![10.0, 0.0, -5.0, 1.0])).unwrap();
        let w = v.get(3);
        assert!(((v.get(0) / w) - 1.0).abs() < 1e-9);
    }
}
