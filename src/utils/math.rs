use num::{Float, PrimInt, Unsigned};

/// Takes a number and maps it from one range to another.
///
/// # Arguments
///
/// - `input`: The number to map.
/// - `input_range`: The range of the input value.
/// - `output_range`: The range to map the input to. The returned value will
///   be in this range.
///
/// # Returns
///
/// - The mapped number.
#[inline(always)]
pub fn map<T: Float>(input: T, input_range: (T, T), output_range: (T, T)) -> T {
    (input - input_range.0) / (input_range.1 - input_range.0) * (output_range.1 - output_range.0)
        + output_range.0
}

/// Takes the x and y ranges and zooms in by a factor of `zoom_factor` centered
/// at `zoom_focus`.
///
/// # Arguments
///
/// - `x_range`: The range of x.
/// - `y_range`: The range of y.
/// - `zoom_factor`: The factor to zoom in by.
/// - `zoom_focus`: The coordinates of the zoom center in the x and y ranges.
///
/// # Returns
///
/// - The new x and y ranges after zooming in.
#[inline(always)]
pub fn zoom<T: Float>(
    x_range: (T, T),
    y_range: (T, T),
    zoom_factor: T,
    zoom_focus: (T, T),
) -> ((T, T), (T, T)) {
    // Move the range so that the center aligns with the origin
    let x_range_translated = shift(x_range, -zoom_focus.0);
    let y_range_translated = shift(y_range, -zoom_focus.1);

    // Scale the range
    let x_range_scaled = scale(x_range_translated, zoom_factor);
    let y_range_scaled = scale(y_range_translated, zoom_factor);

    // Move the range back so that the center returns to its original position
    let x_range_final = shift(x_range_scaled, zoom_focus.0);
    let y_range_final = shift(y_range_scaled, zoom_focus.1);

    (x_range_final, y_range_final)
}

/// Takes an x and y range and zooms-in by a factor of `zoom_factor` centered
/// at a relative `zoom_focus` point.
///
/// # Arguments
///
/// - `x_range`: The range of x.
/// - `y_range`: The range of y.
/// - `zoom_factor`: The factor to zoom in by.
/// - `zoom_focus`: The relative position of the zoom center in the x and y ranges.
///
/// # Returns
///
/// - The new x and y ranges after zooming in.
#[inline(always)]
pub fn zoom_relative<T: Float>(
    x_range: (T, T),
    y_range: (T, T),
    zoom_factor: T,
    zoom_focus: (T, T),
) -> ((T, T), (T, T)) {
    let focus_x = zoom_focus.0 * (x_range.1 - x_range.0) + x_range.0;
    let focus_y = zoom_focus.1 * (y_range.1 - y_range.0) + y_range.0;
    zoom(x_range, y_range, zoom_factor, (focus_x, focus_y))
}

/// Takes a range and scales it by a factor
///
/// # Arguments
///
/// - `range`: The range to scale.
/// - `factor`: The factor to scale by.
///
/// # Returns
///
/// - The new range after scaling.
#[inline(always)]
pub fn scale<T: Float>(range: (T, T), factor: T) -> (T, T) {
    (range.0 * factor, range.1 * factor)
}

/// Takes a range and shifts it by an offset
///
/// # Arguments
///
/// - `range`: The range to shift.
/// - `offset`: The offset to shift by.
///
/// # Returns
///
/// - The new range after shifting.
#[inline(always)]
pub fn shift<T: Float>(range: (T, T), offset: T) -> (T, T) {
    (range.0 + offset, range.1 + offset)
}

/// Takes a range and returns the shift speed
///
/// # Arguments
///
/// - `range`: The range to calculate the shift speed for.
/// - `factor`: How big the shift should be.
///
/// # Returns
///
/// - The shift speed.
#[inline(always)]
pub fn get_shift_speed<T: Float, U: Unsigned + PrimInt>(range: (T, T), factor: U) -> T {
    (range.1 - range.0) / T::from(factor).expect("Conversion failed")
}
