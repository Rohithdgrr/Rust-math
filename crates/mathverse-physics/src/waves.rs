//! Waves and optics

/// Calculate wave speed
/// 
/// # Arguments
/// * `frequency` - Frequency (Hz)
/// * `wavelength` - Wavelength (m)
/// 
/// # Returns
/// Wave speed (m/s)
pub fn wave_speed(frequency: f64, wavelength: f64) -> f64 {
    frequency * wavelength
}

/// Calculate frequency from wavelength and speed
/// 
/// # Arguments
/// * `speed` - Wave speed (m/s)
/// * `wavelength` - Wavelength (m)
/// 
/// # Returns
/// Frequency (Hz), or `None` if the wavelength is zero.
pub fn frequency_from_wavelength(speed: f64, wavelength: f64) -> Option<f64> {
    if wavelength == 0.0 { return None; }
    Some(speed / wavelength)
}

/// Calculate wavelength from frequency and speed
/// 
/// # Arguments
/// * `speed` - Wave speed (m/s)
/// * `frequency` - Frequency (Hz)
/// 
/// # Returns
/// Wavelength (m), or `None` if the frequency is zero.
pub fn wavelength_from_frequency(speed: f64, frequency: f64) -> Option<f64> {
    if frequency == 0.0 { return None; }
    Some(speed / frequency)
}

/// Calculate wave number
/// 
/// # Arguments
/// * `wavelength` - Wavelength (m)
/// 
/// # Returns
/// Wave number (rad/m), or `None` if the wavelength is zero.
pub fn wave_number(wavelength: f64) -> Option<f64> {
    if wavelength == 0.0 { return None; }
    Some(2.0 * std::f64::consts::PI / wavelength)
}

/// Calculate angular frequency
/// 
/// # Arguments
/// * `frequency` - Frequency (Hz)
/// 
/// # Returns
/// Angular frequency (rad/s)
pub fn angular_frequency(frequency: f64) -> f64 {
    2.0 * std::f64::consts::PI * frequency
}

/// Calculate period from frequency
/// 
/// # Arguments
/// * `frequency` - Frequency (Hz)
/// 
/// # Returns
/// Period (s), or `None` if the frequency is zero.
pub fn period(frequency: f64) -> Option<f64> {
    if frequency == 0.0 { return None; }
    Some(1.0 / frequency)
}

/// Calculate speed of sound in air (linear approximation: ~331 m/s at 0 °C, ~343 m/s at 20 °C)
///
/// # Arguments
/// * `temperature` - Temperature in Celsius
///
/// # Returns
/// Speed of sound (m/s)
pub fn speed_of_sound_air(temperature: f64) -> f64 {
    331.0 + 0.6 * temperature
}

/// Calculate Doppler effect (moving source, stationary observer)
/// 
/// # Arguments
/// * `f_source` - Source frequency (Hz)
/// * `v_source` - Source velocity (m/s, positive if moving away)
/// * `v_wave` - Wave speed (m/s)
/// 
/// # Returns
/// Observed frequency (Hz), or `None` if the source approaches at the wave speed or faster.
pub fn doppler_source_moving(f_source: f64, v_source: f64, v_wave: f64) -> Option<f64> {
    let denom = v_wave + v_source;
    if denom <= 0.0 { return None; }
    Some(f_source * v_wave / denom)
}

/// Calculate Doppler effect (stationary source, moving observer)
/// 
/// # Arguments
/// * `f_source` - Source frequency (Hz)
/// * `v_observer` - Observer velocity (m/s, positive if moving toward source)
/// * `v_wave` - Wave speed (m/s)
/// 
/// # Returns
/// Observed frequency (Hz), or `None` if the wave speed is zero.
pub fn doppler_observer_moving(f_source: f64, v_observer: f64, v_wave: f64) -> Option<f64> {
    if v_wave <= 0.0 { return None; }
    Some(f_source * (v_wave + v_observer) / v_wave)
}

/// Calculate beat frequency
/// 
/// # Arguments
/// * `f1` - First frequency (Hz)
/// * `f2` - Second frequency (Hz)
/// 
/// # Returns
/// Beat frequency (Hz)
pub fn beat_frequency(f1: f64, f2: f64) -> f64 {
    (f1 - f2).abs()
}

/// Calculate string wave speed
/// 
/// # Arguments
/// * `tension` - Tension (N)
/// * `linear_density` - Linear mass density (kg/m)
/// 
/// # Returns
/// Wave speed (m/s), or `None` if the linear density is zero.
pub fn string_wave_speed(tension: f64, linear_density: f64) -> Option<f64> {
    if linear_density == 0.0 { return None; }
    Some((tension / linear_density).sqrt())
}

/// Calculate fundamental frequency of string
/// 
/// # Arguments
/// * `length` - String length (m)
/// * `tension` - Tension (N)
/// * `linear_density` - Linear mass density (kg/m)
/// 
/// # Returns
/// Fundamental frequency (Hz), or `None` if the length or linear density is zero.
pub fn string_fundamental_frequency(length: f64, tension: f64, linear_density: f64) -> Option<f64> {
    if length <= 0.0 { return None; }
    string_wave_speed(tension, linear_density).map(|v| v / (2.0 * length))
}

/// Calculate speed of light in medium
/// 
/// # Arguments
/// * `refractive_index` - Refractive index of medium
/// 
/// # Returns
/// Speed of light in medium (m/s), or `None` if the refractive index is not positive.
pub fn light_speed_medium(refractive_index: f64) -> Option<f64> {
    if refractive_index <= 0.0 { return None; }
    Some(crate::constants::C / refractive_index)
}

/// Calculate Snell's law refraction angle
/// 
/// # Arguments
/// * `n1` - Refractive index of first medium
/// * `theta1` - Incident angle (radians)
/// * `n2` - Refractive index of second medium
/// 
/// # Returns
/// Refracted angle (radians), or `None` when total internal reflection occurs.
pub fn snells_law(n1: f64, theta1: f64, n2: f64) -> Option<f64> {
    if n2 <= 0.0 { return None; }
    let arg = (n1 / n2) * theta1.sin();
    if !(-1.0..=1.0).contains(&arg) { return None; }
    Some(arg.asin())
}

/// Calculate critical angle for total internal reflection
/// 
/// # Arguments
/// * `n1` - Refractive index of denser medium
/// * `n2` - Refractive index of less dense medium
/// 
/// # Returns
/// Critical angle (radians), or `None` if `n1 <= 0` or `n2 > n1`.
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if n1 <= 0.0 || n2 > n1 { return None; }
    Some((n2 / n1).asin())
}

/// Calculate lens maker's equation (thin lens)
/// 
/// # Arguments
/// * `n` - Refractive index of lens material
/// * `r1` - Radius of first surface (m)
/// * `r2` - Radius of second surface (m)
/// 
/// # Returns
/// Focal length (m), or `None` if the lens is physically degenerate (`n == 1` or `r1 == r2`).
pub fn lens_focal_length(n: f64, r1: f64, r2: f64) -> Option<f64> {
    let denom = (n - 1.0) * (1.0 / r1 - 1.0 / r2);
    if denom == 0.0 { return None; }
    Some(1.0 / denom)
}

/// Calculate thin lens equation
/// 
/// # Arguments
/// * `f` - Focal length (m)
/// * `d_o` - Object distance (m)
/// 
/// # Returns
/// Image distance (m), or `None` if the object is at the focal point.
pub fn thin_lens_equation(f: f64, d_o: f64) -> Option<f64> {
    let denom = 1.0 / f - 1.0 / d_o;
    if denom == 0.0 { return None; }
    Some(1.0 / denom)
}

/// Calculate magnification
/// 
/// # Arguments
/// * `d_i` - Image distance (m)
/// * `d_o` - Object distance (m)
/// 
/// # Returns
/// Magnification, or `None` if the object distance is zero.
pub fn magnification(d_i: f64, d_o: f64) -> Option<f64> {
    if d_o == 0.0 { return None; }
    Some(-d_i / d_o)
}

/// Calculate diffraction angle (single slit)
/// 
/// # Arguments
/// * `m` - Order of diffraction (integer)
/// * `wavelength` - Wavelength (m)
/// * `slit_width` - Slit width (m)
/// 
/// # Returns
/// Diffraction angle (radians), or `None` if `|mλ/a| > 1` (order does not exist).
pub fn single_slit_diffraction(m: i32, wavelength: f64, slit_width: f64) -> Option<f64> {
    if slit_width == 0.0 { return None; }
    let arg = f64::from(m) * wavelength / slit_width;
    if !(-1.0..=1.0).contains(&arg) { return None; }
    Some(arg.asin())
}

/// Calculate double slit interference
/// 
/// # Arguments
/// * `m` - Order of interference (integer)
/// * `wavelength` - Wavelength (m)
/// * `slit_separation` - Distance between slits (m)
/// 
/// # Returns
/// Angle to bright fringe (radians), or `None` if `|mλ/d| > 1` (order does not exist).
pub fn double_slit_interference(m: i32, wavelength: f64, slit_separation: f64) -> Option<f64> {
    if slit_separation == 0.0 { return None; }
    let arg = f64::from(m) * wavelength / slit_separation;
    if !(-1.0..=1.0).contains(&arg) { return None; }
    Some(arg.asin())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_wave_speed() {
        assert_relative_eq!(wave_speed(440.0, 0.77), 338.8, epsilon = 1e-2);
    }

    #[test]
    fn test_period() {
        assert_relative_eq!(period(440.0).unwrap(), 1.0 / 440.0, epsilon = 1e-6);
        assert!(period(0.0).is_none());
        assert_relative_eq!(wavelength_from_frequency(343.0, 440.0).unwrap(), 343.0 / 440.0, epsilon = 1e-9);
        assert_relative_eq!(wave_number(0.5).unwrap(), 4.0 * std::f64::consts::PI, epsilon = 1e-9);
        assert!(frequency_from_wavelength(343.0, 0.0).is_none());
    }

    #[test]
    fn test_speed_of_sound_air() {
        assert_relative_eq!(speed_of_sound_air(20.0), 343.0, epsilon = 1e-6);
    }

    #[test]
    fn test_doppler() {
        assert_relative_eq!(doppler_source_moving(440.0, 20.0, 343.0).unwrap(), 440.0 * 343.0 / 363.0, epsilon = 1e-9);
        assert!(doppler_source_moving(440.0, -343.0, 343.0).is_none());
        assert_relative_eq!(doppler_observer_moving(440.0, 20.0, 343.0).unwrap(), 440.0 * 363.0 / 343.0, epsilon = 1e-9);
    }

    #[test]
    fn test_snells_and_critical() {
        let theta = snells_law(1.0, 0.5, 1.5).unwrap();
        assert_relative_eq!(theta, (0.5_f64.sin() / 1.5).asin(), epsilon = 1e-9);
        assert!(snells_law(1.5, std::f64::consts::FRAC_PI_2, 1.0).is_none());
        assert_relative_eq!(critical_angle(1.5, 1.0).unwrap(), (1.0_f64 / 1.5).asin(), epsilon = 1e-9);
        assert!(critical_angle(1.0, 1.5).is_none());
        assert_relative_eq!(light_speed_medium(2.0).unwrap(), crate::constants::C / 2.0, epsilon = 1e-6);
        assert!(light_speed_medium(0.0).is_none());
    }

    #[test]
    fn test_lenses() {
        assert_relative_eq!(lens_focal_length(1.5, 0.1, -0.1).unwrap(), 0.1, epsilon = 1e-6);
        assert!(lens_focal_length(1.0, 0.1, -0.1).is_none());
        assert_relative_eq!(thin_lens_equation(0.1, 0.2).unwrap(), 0.2, epsilon = 1e-6);
        assert!(thin_lens_equation(0.1, 0.1).is_none());
        assert_relative_eq!(magnification(0.2, 0.4).unwrap(), -0.5, epsilon = 1e-9);
        assert!(magnification(0.2, 0.0).is_none());
    }

    #[test]
    fn test_diffraction() {
        assert_relative_eq!(single_slit_diffraction(1, 5e-7, 1e-3).unwrap(), 5e-4_f64.asin(), epsilon = 1e-9);
        assert!(single_slit_diffraction(3, 5e-7, 1e-6).is_none());
        assert_relative_eq!(double_slit_interference(1, 5e-7, 1e-3).unwrap(), 5e-4_f64.asin(), epsilon = 1e-9);
        assert!(double_slit_interference(2, 5e-7, 5e-7).is_none());
    }

    #[test]
    fn test_string_waves() {
        assert_relative_eq!(string_wave_speed(100.0, 0.01).unwrap(), 100.0, epsilon = 1e-9);
        assert!(string_wave_speed(100.0, 0.0).is_none());
        assert_relative_eq!(string_fundamental_frequency(0.5, 100.0, 0.01).unwrap(), 100.0, epsilon = 1e-9);
    }
}
