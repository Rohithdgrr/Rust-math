//! Temperature conversion examples.
//!
//! This example demonstrates temperature unit conversions.

use mathverse_units::conversions::{
    celsius_to_kelvin, kelvin_to_celsius,
    fahrenheit_to_celsius, celsius_to_fahrenheit,
    fahrenheit_to_kelvin, kelvin_to_fahrenheit
};

fn main() {
    let celsius = 25.0;
    let fahrenheit = celsius_to_fahrenheit(celsius);
    let kelvin = celsius_to_kelvin(celsius);

    println!("{}°C = {}°F", celsius, fahrenheit);
    println!("{}°C = {}K", celsius, kelvin);

    let fahrenheit = 77.0;
    let celsius = fahrenheit_to_celsius(fahrenheit);
    let kelvin = fahrenheit_to_kelvin(fahrenheit);

    println!("{}°F = {}°C", fahrenheit, celsius);
    println!("{}°F = {}K", fahrenheit, kelvin);

    let kelvin = 300.0;
    let celsius = kelvin_to_celsius(kelvin);
    let fahrenheit = kelvin_to_fahrenheit(kelvin);

    println!("{}K = {}°C", kelvin, celsius);
    println!("{}K = {}°F", kelvin, fahrenheit);
}
