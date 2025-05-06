use std::time::SystemTime;

pub mod utils;

macro_rules! define_float_choice {
    ($float:ty, $zoom_delta:expr) => {
        /// Float type used for computations.
        pub type FloatChoice = $float;
        /// Maximum zoom delta to prevent floating point errors.
        /// The zoom delta is the difference between the maximum and minimum
        /// values of the x and y ranges.
        pub const MAX_ZOOM_DELTA: FloatChoice = $zoom_delta;
    };
}

#[cfg(not(feature = "f64"))]
define_float_choice!(f32, 1e-5);

#[cfg(feature = "f64")]
define_float_choice!(f64, 1e-20);

/// Returns the path to the save file with a unique name based on the current
/// time.
///
/// The format is `./{prefix}_{timestamp}.png`.
///
/// # Arguments
///
/// - `prefix`: A prefix for the filename.
pub fn get_save_path(prefix: &str) -> String {
    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("./{}_{:?}.png", prefix, time)
}
