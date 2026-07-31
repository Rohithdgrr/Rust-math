pub mod counting;
pub mod sequences;
pub mod partitions;
pub mod stirling;
pub mod subsets;
pub mod inclusion_exclusion;
pub mod factorials;

pub use counting::*;
pub use sequences::*;
pub use partitions::*;
pub use stirling::*;
pub use factorials::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined() {
        assert_eq!(combinations(10, 3), 120);
        assert_eq!(fibonacci(10), 55);
        assert_eq!(partition(5), 7);
        assert_eq!(stirling2(5, 3), 25);
        assert_eq!(factorial(5), 120);
    }
}
