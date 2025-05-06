use nannou::wgpu;

use crate::FloatChoice;

// This struct is passed to the GPU as a uniform buffer
// See alignment rules for the GPU:
// https://www.w3.org/TR/WGSL/#alignment-and-size
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct ComputeData {
    pub max_iter: u32,
    pub num_particles: u32,
    _padding: [u32; 2], // Needed to align the vec2<f64> to 16 bytes
    pub dt: FloatChoice,
    pub mu: FloatChoice,
    /// Initial render range in x for function
    x_range: [FloatChoice; 2],
    /// Initial render range in y for function
    y_range: [FloatChoice; 2],
}

impl Default for ComputeData {
    fn default() -> Self {
        Self {
            max_iter: 100,
            num_particles: 20_000,
            _padding: [0; 2],
            dt: 0.1,
            mu: 4.5,
            x_range: [-2.0, 0.50],
            y_range: [-1.25, 1.25],
        }
    }
}

impl ComputeData {
    /// Returns the struct as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { wgpu::bytes::from(self) }
    }

    /// Gets the number x_range as a tuple.
    pub fn get_x_range(&self) -> (FloatChoice, FloatChoice) {
        (self.x_range[0], self.x_range[1])
    }

    /// Gets the number y_range as a tuple.
    pub fn get_y_range(&self) -> (FloatChoice, FloatChoice) {
        (self.y_range[0], self.y_range[1])
    }

    /// Updates the the x_range field of the struct.
    pub fn update_x_range(&mut self, x_range: (FloatChoice, FloatChoice)) {
        self.x_range = [x_range.0, x_range.1];
    }

    /// Updates the the y_range field of the struct.
    pub fn update_y_range(&mut self, y_range: (FloatChoice, FloatChoice)) {
        self.y_range = [y_range.0, y_range.1];
    }
}

// This struct is passed to the GPU as a storage buffer
// See alignment rules for the GPU:
// https://www.w3.org/TR/WGSL/#alignment-and-size
#[repr(C, align(4))]
#[derive(Clone, Copy)]
pub struct PostProcessingData {
    value_min: f32,
    value_max: f32,
    histogram_n: u32,
    histogram: [u32; 256],
    cdf_threshold: f32,
    cdf_non_zero: f32,
    cdf: [f32; 256],
}
impl Default for PostProcessingData {
    fn default() -> Self {
        Self {
            value_min: f32::MAX,
            value_max: 0.0,
            histogram_n: 0,
            histogram: [0; 256],
            cdf_threshold: 0.0,
            cdf_non_zero: 0.0,
            cdf: [0.0; 256],
        }
    }
}

impl PostProcessingData {
    /// Returns the struct as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { wgpu::bytes::from(self) }
    }
}
