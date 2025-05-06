use std::time::SystemTime;

pub mod utils;

/// Float type used for computations.
pub type FloatChoice = f64;

/// Maximum zoom delta to prevent floating point errors.
/// The zoom delta is the difference between the maximum and minimum values of
/// the x and y ranges.
pub const MAX_ZOOM_DELTA: FloatChoice = 1e-10;
// pub const MAX_ZOOM_DELTA: FloatChoice = 1e-5;

pub fn get_save_path(prefix: &str) -> String {
    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("./{}_{:?}.png", prefix, time)
}
