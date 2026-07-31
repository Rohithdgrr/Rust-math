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
/// Frequency (Hz)
pub fn frequency_from_wavelength(speed: f64, wavelength: f64) -> f64 {
    speed / wavelength
}

/// Calculate wavelength from frequency and speed
/// 
/// # Arguments
/// * `speed` - Wave speed (m/s)
/// * `frequency` - Frequency (Hz)
/// 
/// # Returns
/// Wavelength (m)
pub fn wavelength_from_frequency(speed: f64, frequency: f64) -> f64 {
    speed / frequency
}

/// Calculate wave number
/// 
/// # Arguments
/// * `wavelength` - Wavelength (m)
/// 
/// # Returns
/// Wave number (rad/m)
pub fn wave_number(wavelength: f64) -> f64 {
    2.0 * std::f64::consts::PI / wavelength
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
/// Period (s)
pub fn period(frequency: f64) -> f64 {
    1.0 / frequency
}

/// Calculate speed of sound in air
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
/// Observed frequency (Hz)
pub fn doppler_source_moving(f_source: f64, v_source: f64, v_wave: f64) -> f64 {
    f_source * v_wave / (v_wave + v_source)
}

/// Calculate Doppler effect (stationary source, moving observer)
/// 
/// # Arguments
/// * `f_source` - Source frequency (Hz)
/// * `v_observer` - Observer velocity (m/s, positive if moving toward source)
/// * `v_wave` - Wave speed (m/s)
/// 
/// # Returns
/// Observed frequency (Hz)
pub fn doppler_observer_moving(f_source: f64, v_observer: f64, v_wave: f64) -> f64 {
    f_source * (v_wave + v_observer) / v_wave
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
/// Wave speed (m/s)
pub fn string_wave_speed(tension: f64, linear_density: f64) -> f64 {
    (tension / linear_density).sqrt()
}

/// Calculate fundamental frequency of string
/// 
/// # Arguments
/// * `length` - String length (m)
/// * `tension` - Tension (N)
/// * `linear_density` - Linear mass density (kg/m)
/// 
/// # Returns
/// Fundamental frequency (Hz)
pub fn string_fundamental_frequency(length: f64, tension: f64, linear_density: f64) -> f64 {
    string_wave_speed(tension, linear_density) / (2.0 * length)
}

/// Calculate speed of light in medium
/// 
/// # Arguments
/// * `refractive_index` - Refractive index of medium
/// 
/// # Returns
/// Speed of light in medium (m/s)
pub fn light_speed_medium(refractive_index: f64) -> f64 {
    crate::constants::C / refractive_index
}

/// Calculate Snell's law refraction angle
/// 
/// # Arguments
/// * `n1` - Refractive index of first medium
/// * `theta1` - Incident angle (radians)
/// * `n2` - Refractive index of second medium
/// 
/// # Returns
/// Refracted angle (radians)
pub fn snells_law(n1: f64, theta1: f64, n2: f64) -> f64 {
    ((n1 / n2) * theta1.sin()).asin()
}

/// Calculate critical angle for total internal reflection
/// 
/// # Arguments
/// * `n1` - Refractive index of denser medium
/// * `n2` - Refractive index of less dense medium
/// 
/// # Returns
/// Critical angle (radians)
pub fn critical_angle(n1: f64, n2: f64) -> f64 {
    (n2 / n1).asin()
}

/// Calculate lens maker's equation (thin lens)
/// 
/// # Arguments
/// * `n` - Refractive index of lens material
/// * `r1` - Radius of first surface (m)
/// * `r2` - Radius of second surface (m)
/// 
/// # Returns
/// Focal length (m)
pub fn lens_focal_length(n: f64, r1: f64, r2: f64) -> f64 {
    1.0 / ((n - 1.0) * (1.0 / r1 - 1.0 / r2))
}

/// Calculate thin lens equation
/// 
/// # Arguments
/// * `f` - Focal length (m)
/// * `d_o` - Object distance (m)
/// 
/// # Returns
/// Image distance (m)
pub fn thin_lens_equation(f: f64, d_o: f64) -> f64 {
    1.0 / (1.0 / f - 1.0 / d_o)
}

/// Calculate magnification
/// 
/// # Arguments
/// * `d_i` - Image distance (m)
/// * `d_o` - Object distance (m)
/// 
/// # Returns
/// Magnification
pub fn magnification(d_i: f64, d_o: f64) -> f64 {
    -d_i / d_o
}

/// Calculate diffraction angle (single slit)
/// 
/// # Arguments
/// * `m` - Order of diffraction (integer)
/// * `wavelength` - Wavelength (m)
/// * `slit_width` - Slit width (m)
/// 
/// # Returns
/// Diffraction angle (radians)
pub fn single_slit_diffraction(m: i32, wavelength: f64, slit_width: f64) -> f64 {
    ((m as f64) * wavelength / slit_width).asin()
}

/// Calculate double slit interference
/// 
/// # Arguments
/// * `m` - Order of interference (integer)
/// * `wavelength` - Wavelength (m)
/// * `slit_separation` - Distance between slits (m)
/// 
/// # Returns
/// Angle to bright fringe (radians)
pub fn double_slit_interference(m: i32, wavelength: f64, slit_separation: f64) -> f64 {
    ((m as f64) * wavelength / slit_separation).asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_wave_speed() {
        assert_relative_eq!(wave_speed(440.0, 0.77), 338.8, epsilon = 1e-2);
    }

    #[test]
    fn test_period() {
        assert_relative_eq!(period(440.0), 1.0 / 440.0, epsilon = 1e-6);
    }

    #[test]
    fn test_speed_of_sound_air() {
        assert_relative_eq!(speed_of_sound_air(20.0), 343.0, epsilon = 1e-6);
    }
}
