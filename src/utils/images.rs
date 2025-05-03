use std::cell::Ref;

use nannou::{
    image::{self, ImageBuffer},
    wgpu::{self, WithDeviceQueuePair},
    window::Window,
};

/// Converts an image buffer to a texture.
///
/// # Arguments
///
/// - `window`: A reference to the window that will be used to create the texture.
/// - `image`: The image buffer to be converted to a texture.
pub fn create_texture(
    window: Ref<'_, Window>,
    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
) -> wgpu::Texture {
    let usage = nannou::wgpu::TextureUsages::COPY_SRC
        | nannou::wgpu::TextureUsages::COPY_DST
        | nannou::wgpu::TextureUsages::RENDER_ATTACHMENT
        | nannou::wgpu::TextureUsages::TEXTURE_BINDING;

    window.with_device_queue_pair(|device, queue| {
        wgpu::Texture::load_from_image_buffer(device, queue, usage, &image)
    })
}

/// Recalibrates the pixel values of a 2D array representing an image to the range [0, 255].
///
/// Taken from my teacher, Max Mignotte:
/// https://www.iro.umontreal.ca/~mignotte/
///
/// # Arguments
///
/// - `mat`: A mutable reference to a 2D vector (`Vec<Vec<f64>>`) containing pixel values.
///   The pixel values should be of type `f64`, but they may not necessarily be within the
///   [0, 255] range before recalibration.
///
/// # Panics
///
/// This function will panic if the 2D array `mat` is empty (i.e., has no rows or columns) or if
/// any row has a different length from the first row.
pub fn recalibrate(mat: &mut [Vec<f64>]) {
    let width = mat[0].len();
    let height = mat.len();

    // Find the min luma value
    let mut luma_min = mat[0][0];
    (0..height).for_each(|i| {
        (0..width).for_each(|j| {
            luma_min = mat[i][j].min(luma_min);
        });
    });

    // Subtract min from all pixels in the image
    (0..height).for_each(|i| {
        (0..width).for_each(|j| {
            mat[i][j] -= luma_min;
        });
    });

    // Find the max luma value
    let mut luma_max = mat[0][0];
    (0..height).for_each(|i| {
        (0..width).for_each(|j| {
            luma_max = mat[i][j].max(luma_max);
        });
    });

    // Recalibrate the image
    (0..height).for_each(|i| {
        (0..width).for_each(|j| {
            mat[i][j] *= 255.0 / luma_max;
        });
    });
}

/// Performs histogram equalization on a 2D array of pixel values (luminance).
///
/// Taken from my teacher, Max Mignotte:
/// https://www.iro.umontreal.ca/~mignotte/
///
/// # Arguments
///
/// - `array`: A mutable 2D vector of `f64` values representing the pixel intensities
///   of an image, typically in the range [0.0, 255.0].
/// - `thresh`: A threshold value that filters out pixel values lower than this value
///   when calculating the histogram. Pixels with values greater than `thresh` are
///   included in the histogram calculation.
///
/// # Description
///
/// The function performs the following steps:
/// 1. **Histogram Calculation**: It calculates a normalized histogram (`histo_ng`) for pixel values
///    greater than `thresh` across all pixels in the 2D array. The frequency of each pixel value
///    is counted and normalized by the total number of pixels greater than the threshold.
/// 2. **Cumulative Distribution Function (CDF)**: It computes the cumulative distribution of the
///    normalized histogram (`FnctRept`).
/// 3. **Scaling**: The CDF is scaled to fit within the range [0, 255].
/// 4. **Equalization**: The pixel values in the original 2D array are updated using the scaled CDF
///    to perform the histogram equalization, improving the contrast of the image.
///
/// # Panics
///
/// This function will panic if `array` is empty (i.e., no rows or columns) or if any row in the
/// 2D array has a different length than the first row.
pub fn equalize(array: &mut [Vec<f64>], thresh: f64) {
    let height = array.len();
    let width = array[0].len();

    // Calculate histogram Ng (normalized)
    let mut histo_ng = vec![0.0; 256];
    let mut n = 0;
    (0..height).for_each(|i| {
        (0..width).for_each(|j| {
            let luma = array[i][j];
            if luma > thresh {
                histo_ng[luma as usize] += 1.0;
                n += 1;
            }
        });
    });

    // Normalize the histogram
    (0..256).for_each(|i| {
        histo_ng[i] /= n as f64;
    });

    // Calculate cumulative distribution function (FnctRept)
    let mut fnct_rept = vec![0.0; 256];
    (1..256).for_each(|i| {
        fnct_rept[i] = fnct_rept[i - 1] + histo_ng[i];
    });

    // Scale the cumulative distribution to the 0-255 range
    (0..256).for_each(|i| {
        fnct_rept[i] = (fnct_rept[i] * 255.0).round();
    });

    // Equalize the image
    (0..height).for_each(|i| {
        (0..width).for_each(|j| {
            array[i][j] = fnct_rept[array[i][j] as usize];
        });
    });
}

/// Recalibrates the pixel values of a 2D RGB image to the range [0, 255].
///
/// # Arguments
///
/// - `mat`: A mutable reference to a 2D vector (`Vec<Vec<[f64; 3]>>`) containing pixel values.
///   Each pixel is a 3‐element array `[R, G, B]` of `f64`. Values need not be within [0, 255].
///
/// # Panics
///
/// Panics if `mat` is empty or if any row has a different width from the first row.
pub fn recalibrate_rgb(mat: &mut [Vec<[f64; 3]>]) {
    let width = mat[0].len();

    // For each channel, find min and max
    let mut mins = [f64::MAX; 3];
    let mut maxs = [f64::MIN; 3];

    for row in mat.iter() {
        assert_eq!(row.len(), width, "All rows must have the same length");
        for &pixel in row.iter() {
            for c in 0..3 {
                mins[c] = mins[c].min(pixel[c]);
                maxs[c] = maxs[c].max(pixel[c]);
            }
        }
    }

    // Subtract min and scale so that max becomes 255
    for row in mat.iter_mut() {
        for pixel in row.iter_mut() {
            for c in 0..3 {
                // Shift so that channel-min is 0
                pixel[c] -= mins[c];
                // Scale so that channel-max maps to 255
                if maxs[c] - mins[c] > 0.0 {
                    pixel[c] *= 255.0 / (maxs[c] - mins[c]);
                }
            }
        }
    }
}

/// Performs per‐channel histogram equalization on a 2D RGB image.
///
/// # Arguments
///
/// - `array`: A mutable 2D vector (`Vec<Vec<[f64; 3]>>`) of pixel values in [0, 255].
/// - `thresh`: Pixels with channel‐values ≤ `thresh` are skipped when building each channel histogram.
///
/// # Panics
///
/// Panics if `array` is empty or if any row has a different width from the first row.
pub fn equalize_rgb(array: &mut [Vec<[f64; 3]>], thresh: f64) {
    let width = array[0].len();

    // Build one histogram per channel
    let mut histo = vec![[0usize; 256]; 3];
    let mut counts = [0usize; 3];

    // Count
    for row in array.iter() {
        assert_eq!(row.len(), width, "All rows must have the same length");
        for &pixel in row.iter() {
            for c in 0..3 {
                let val = pixel[c].clamp(0.0, 255.0).round() as usize;
                if pixel[c] > thresh {
                    histo[c][val] += 1;
                    counts[c] += 1;
                }
            }
        }
    }

    // Compute normalized CDFs for each channel
    let mut cdfs = vec![[0f64; 256]; 3];
    for c in 0..3 {
        if counts[c] == 0 {
            // no valid pixels above threshold: leave channel untouched
            continue;
        }
        // histogram → probability
        let mut prob = [0f64; 256];
        for i in 0..256 {
            prob[i] = histo[c][i] as f64 / counts[c] as f64;
        }
        // CDF
        cdfs[c][0] = prob[0];
        for i in 1..256 {
            cdfs[c][i] = cdfs[c][i - 1] + prob[i];
        }
        // scale to [0,255]
        for i in 0..256 {
            cdfs[c][i] = (cdfs[c][i] * 255.0).round();
        }
    }

    // Apply equalization
    for row in array.iter_mut() {
        for pixel in row.iter_mut() {
            for c in 0..3 {
                let idx = pixel[c].clamp(0.0, 255.0).round() as usize;
                pixel[c] = cdfs[c][idx];
            }
        }
    }
}
