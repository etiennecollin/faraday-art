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
