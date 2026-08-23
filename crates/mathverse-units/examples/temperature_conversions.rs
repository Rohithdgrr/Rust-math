//! Temperature conversion examples.
//!
//! This example demonstrates temperature unit conversions.

use mathverse_units::conversions::{
    celsius_to_fahrenheit, celsius_to_kelvin, fahrenheit_to_celsius, fahrenheit_to_kelvin,
    kelvin_to_celsius, kelvin_to_fahrenheit,
};

fn main() {
    let celsius = 25.0;
    let fahrenheit = celsius_to_fahrenheit(celsius);
    let kelvin = celsius_to_kelvin(celsius);

    println!("{celsius}°C = {fahrenheit}°F");
    println!("{celsius}°C = {kelvin}K");

    let fahrenheit = 77.0;
    let celsius = fahrenheit_to_celsius(fahrenheit);
    let kelvin = fahrenheit_to_kelvin(fahrenheit);

    println!("{fahrenheit}°F = {celsius}°C");
    println!("{fahrenheit}°F = {kelvin}K");

    let kelvin = 300.0;
    let celsius = kelvin_to_celsius(kelvin);
    let fahrenheit = kelvin_to_fahrenheit(kelvin);

    println!("{kelvin}K = {celsius}°C");
    println!("{kelvin}K = {fahrenheit}°F");
}
