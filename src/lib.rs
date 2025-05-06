use std::time::SystemTime;

pub mod utils;

pub fn get_save_path(prefix: &str) -> String {
    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("./{}_{:?}.png", prefix, time)
}
