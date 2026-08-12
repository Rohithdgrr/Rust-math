//! Datetime axis support for time series plotting.

/// A date/time value represented as seconds since Unix epoch (1970-01-01T00:00:00Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DateTime {
    /// Seconds since Unix epoch.
    pub timestamp: i64,
    /// Sub-second nanoseconds (0..999_999_999).
    pub nanos: u32,
}

impl DateTime {
    /// Create a new DateTime from components.
    pub fn new(year: i32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> Self {
        let timestamp = unix_timestamp(year, month, day, hour, min, sec);
        Self {
            timestamp,
            nanos: 0,
        }
    }

    /// Create from a Unix timestamp (seconds since epoch).
    pub fn from_timestamp(ts: i64) -> Self {
        Self {
            timestamp: ts,
            nanos: 0,
        }
    }

    /// Create from a Unix timestamp with nanoseconds.
    pub fn from_timestamp_nanos(ts: i64, nanos: u32) -> Self {
        Self {
            timestamp: ts,
            nanos,
        }
    }

    /// Convert to seconds since epoch (for plotting).
    pub fn to_f64(self) -> f64 {
        self.timestamp as f64 + self.nanos as f64 / 1_000_000_000.0
    }

    /// Create from seconds since epoch.
    pub fn from_f64(secs: f64) -> Self {
        let timestamp = secs.floor() as i64;
        let nanos = ((secs - secs as f64) * 1_000_000_000.0).round() as u32;
        Self { timestamp, nanos }
    }

    /// Format according to the given format string.
    pub fn format(&self, fmt: &str) -> String {
        let (y, mo, d, h, mi, s) = to_ymdhms(self.timestamp);
        let fractional = self.nanos as f64 / 1_000_000_000.0;

        fmt.replace("%Y", &format!("{:04}", y))
            .replace("%m", &format!("{:02}", mo))
            .replace("%d", &format!("{:02}", d))
            .replace("%H", &format!("{:02}", h))
            .replace("%M", &format!("{:02}", mi))
            .replace("%S", &format!("{:02}", s))
            .replace("%f", &format!("{:06}", (fractional * 1_000_000.0) as u32))
    }

    /// Auto-detect an appropriate format based on the time span.
    pub fn auto_format(&self, other: DateTime) -> &'static str {
        let span = (self.timestamp - other.timestamp).abs();
        if span < 60 {
            "%H:%M:%S" // seconds
        } else if span < 3600 {
            "%H:%M" // minutes
        } else if span < 86400 {
            "%H:%M" // hours
        } else if span < 86400 * 30 {
            "%m-%d" // days
        } else if span < 86400 * 365 {
            "%Y-%m" // months
        } else {
            "%Y" // years
        }
    }
}

impl Default for DateTime {
    fn default() -> Self {
        Self::from_timestamp(0)
    }
}

impl std::fmt::Display for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fmt = self.auto_format(*self);
        write!(f, "{}", self.format(fmt))
    }
}

/// Datetime axis configuration.
#[derive(Debug, Clone)]
pub struct DatetimeAxis {
    /// Format string for tick labels (auto-detected if None).
    pub format: Option<String>,
    /// Number of ticks to target.
    pub tick_count: usize,
    /// Rotation angle for labels (in degrees).
    pub label_rotation: f64,
}

impl Default for DatetimeAxis {
    fn default() -> Self {
        Self {
            format: None,
            tick_count: 6,
            label_rotation: 0.0,
        }
    }
}

impl DatetimeAxis {
    /// Create a new datetime axis with auto-formatting.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set an explicit format string.
    pub fn with_format(mut self, fmt: impl Into<String>) -> Self {
        self.format = Some(fmt.into());
        self
    }

    /// Set the number of ticks.
    pub fn with_tick_count(mut self, count: usize) -> Self {
        self.tick_count = count;
        self
    }

    /// Set label rotation in degrees.
    pub fn with_label_rotation(mut self, degrees: f64) -> Self {
        self.label_rotation = degrees;
        self
    }

    /// Generate tick positions and labels for a datetime range.
    pub fn ticks(&self, min: DateTime, max: DateTime) -> Vec<(f64, String)> {
        let span = max.timestamp - min.timestamp;
        if span <= 0 {
            return vec![(min.to_f64(), min.format("%Y-%m-%d %H:%M:%S"))];
        }

        let fmt_str = match &self.format {
            Some(f) => f.as_str(),
            None => min.auto_format(max),
        };

        // Generate "nice" tick intervals
        let interval = nice_time_interval(span, self.tick_count);
        let start = (min.timestamp / interval + 1) * interval;

        let mut ticks = Vec::new();
        let mut t = start;
        while t <= max.timestamp {
            let dt = DateTime::from_timestamp(t);
            ticks.push((dt.to_f64(), dt.format(fmt_str)));
            t += interval;
        }

        // Always include the start
        if ticks.is_empty() || ticks[0].0 > min.to_f64() {
            ticks.insert(0, (min.to_f64(), min.format(fmt_str)));
        }

        ticks
    }
}

/// matplotlib-style date tick locator operating on float timestamps (seconds
/// since the Unix epoch) — plugs into [`crate::ticks::TickLocator`].
#[derive(Debug, Clone)]
pub struct DateLocator {
    /// Label format (auto-detected when empty).
    pub format: String,
    /// Target tick count.
    pub tick_count: usize,
}

impl DateLocator {
    /// Create a locator with an explicit format (e.g. `"%Y-%m-%d"`).
    #[must_use]
    pub fn new(format: &str, tick_count: usize) -> Self {
        Self {
            format: format.to_string(),
            tick_count: tick_count.max(1),
        }
    }

    /// Locate tick positions for a `[lo, hi]` float-timestamp range.
    #[must_use]
    pub fn locate(&self, lo: f64, hi: f64, _max_ticks: usize) -> Vec<f64> {
        let lo_dt = DateTime::from_f64(lo);
        let hi_dt = DateTime::from_f64(hi);
        DatetimeAxis::new()
            .with_tick_count(self.tick_count)
            .ticks(lo_dt, hi_dt)
            .into_iter()
            .map(|(t, _)| t)
            .collect()
    }
}

impl crate::ticks::TickLocator for DateLocator {
    fn locate(&self, lo: f64, hi: f64, max_ticks: usize) -> Vec<f64> {
        DateLocator::locate(self, lo, hi, max_ticks)
    }
}

/// matplotlib-style date tick formatter: formats a float timestamp using a
/// strftime-like format (see [`DateTime::format`]).
#[derive(Debug, Clone)]
pub struct DateFormatter {
    /// strftime format string (auto-detected when empty).
    pub format: String,
}

impl DateFormatter {
    /// Create a formatter with a strftime-like format.
    #[must_use]
    pub fn new(format: &str) -> Self {
        Self {
            format: format.to_string(),
        }
    }

    /// Format one float timestamp.
    #[must_use]
    pub fn format(&self, t: f64) -> String {
        let dt = DateTime::from_f64(t);
        if self.format.is_empty() {
            let auto = dt.auto_format(DateTime::from_timestamp(dt.timestamp + 1));
            if auto.is_empty() {
                dt.format("%Y-%m-%d")
            } else {
                dt.format(auto)
            }
        } else {
            dt.format(&self.format)
        }
    }
}

impl Default for DateFormatter {
    fn default() -> Self {
        Self {
            format: String::new(),
        }
    }
}

impl crate::ticks::TickFormatter for DateFormatter {
    fn format(&self, value: f64) -> String {
        DateFormatter::format(self, value)
    }
}

/// Compute a "nice" time interval for tick spacing.
fn nice_time_interval(span: i64, target_ticks: usize) -> i64 {
    let raw = span / target_ticks.max(1) as i64;

    // Time intervals in seconds
    let intervals = [
        1,           // 1 second
        5,           // 5 seconds
        15,          // 15 seconds
        30,          // 30 seconds
        60,          // 1 minute
        300,         // 5 minutes
        900,         // 15 minutes
        1800,        // 30 minutes
        3600,        // 1 hour
        7200,        // 2 hours
        10800,       // 3 hours
        21600,       // 6 hours
        43200,       // 12 hours
        86400,       // 1 day
        604800,      // 1 week
        2592000,     // 30 days
        7776000,     // 90 days
        15552000,    // 180 days
        31536000,    // 1 year
        63072000,    // 2 years
        157680000,   // 5 years
        315360000,   // 10 years
    ];

    // Find the smallest interval >= raw
    for &interval in &intervals {
        if interval >= raw {
            return interval;
        }
    }
    intervals[intervals.len() - 1]
}

/// Convert Unix timestamp to (year, month, day, hour, minute, second).
fn to_ymdhms(ts: i64) -> (i32, u32, u32, u32, u32, u32) {
    let ts = ts.max(0) as u64;
    let seconds = ts % 60;
    let minutes = ts / 60;
    let hours = minutes / 60;
    let days = hours / 24;

    let mut year = 1970i32;
    let mut remaining_days = days;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let days_in_month = [
        31,
        if is_leap_year(year) { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];

    let mut month = 1u32;
    for &dim in &days_in_month {
        if remaining_days < dim as u64 {
            break;
        }
        remaining_days -= dim as u64;
        month += 1;
    }

    (
        year,
        month,
        remaining_days as u32 + 1,
        (hours % 24) as u32,
        (minutes % 60) as u32,
        seconds as u32,
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Compute Unix timestamp from date components.
fn unix_timestamp(year: i32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> i64 {
    let mut days = 0i64;

    // Days from 1970 to year
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }

    // Days in current year
    let days_in_month = [
        31,
        if is_leap_year(year) { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    for m in 1..month {
        days += days_in_month[(m - 1) as usize] as i64;
    }
    days += (day - 1) as i64;

    days * 86400 + hour as i64 * 3600 + min as i64 * 60 + sec as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_basic() {
        let dt = DateTime::new(2024, 1, 15, 10, 30, 0);
        assert_eq!(dt.format("%Y-%m-%d %H:%M"), "2024-01-15 10:30");
    }

    #[test]
    fn datetime_from_timestamp() {
        let dt = DateTime::from_timestamp(0);
        assert_eq!(dt.format("%Y-%m-%d"), "1970-01-01");
    }

    #[test]
    fn datetime_roundtrip() {
        let dt = DateTime::new(2024, 6, 15, 12, 0, 0);
        let f = dt.to_f64();
        let dt2 = DateTime::from_f64(f);
        assert_eq!(dt.timestamp, dt2.timestamp);
    }

    #[test]
    fn auto_format_seconds() {
        let a = DateTime::new(2024, 1, 1, 0, 0, 0);
        let b = DateTime::new(2024, 1, 1, 0, 0, 30);
        assert_eq!(a.auto_format(b), "%H:%M:%S");
    }

    #[test]
    fn auto_format_days() {
        let a = DateTime::new(2024, 1, 1, 0, 0, 0);
        let b = DateTime::new(2024, 1, 15, 0, 0, 0);
        assert_eq!(a.auto_format(b), "%m-%d");
    }

    #[test]
    fn auto_format_years() {
        let a = DateTime::new(2020, 1, 1, 0, 0, 0);
        let b = DateTime::new(2024, 1, 1, 0, 0, 0);
        assert_eq!(a.auto_format(b), "%Y");
    }

    #[test]
    fn date_locator_ticks_within_range() {
        let lo = DateTime::new(2024, 1, 1, 0, 0, 0).to_f64();
        let hi = DateTime::new(2024, 1, 10, 0, 0, 0).to_f64();
        let loc = DateLocator::new("%m-%d", 6);
        let ticks = loc.locate(lo, hi, 6);
        assert!(!ticks.is_empty());
        assert!(ticks.iter().all(|t| (*t >= lo - 1e-6) && (*t <= hi + 1e-6)));
        // Tick spacing should be at least 1 day apart (86400 s).
        assert!(ticks.len() <= 12);
    }

    #[test]
    fn date_locator_empty_range() {
        let t = DateTime::new(2024, 1, 1, 0, 0, 0).to_f64();
        let ticks = DateLocator::new("%Y", 6).locate(t, t, 6);
        assert!(!ticks.is_empty());
    }

    #[test]
    fn date_formatter_formats() {
        let t = DateTime::new(2024, 3, 5, 14, 30, 0).to_f64();
        let f = DateFormatter::new("%Y-%m-%d");
        assert_eq!(f.format(t), "2024-03-05");
        let auto = DateFormatter::default();
        let s = auto.format(t);
        assert!(!s.is_empty());
    }
}
