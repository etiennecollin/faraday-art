use std::cell::RefMut;

use nannou::{
    color::ConvertInto,
    image::{self, ImageBuffer},
    prelude::*,
};

use super::pipeline_buffers::{FaradayData, GlobalData};

pub struct GPUPipeline {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    faraday_data_buffer: wgpu::Buffer,
    global_data_buffer: wgpu::Buffer,
    // Generate texture
    compute_bgl: wgpu::BindGroupLayout,
    compute_bg: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    // Post-processing
    min_max_pipeline: wgpu::ComputePipeline,
    recalibrate_pipeline: wgpu::ComputePipeline,
    histogram_pipeline: wgpu::ComputePipeline,
    cdf_pipeline: wgpu::ComputePipeline,
    equalize_pipeline: wgpu::ComputePipeline,
    // Render
    render_bgl: wgpu::BindGroupLayout,
    render_bg: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
}

impl GPUPipeline {
    /// Size of the workgroup for the compute shader.
    const WORKGROUP_SIZE: u32 = 16;
    /// Format of the texture used for the compute and render pipelines.
    const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
    /// Number of channels in the texture.
    const NUM_CHANNELS: u32 = 4;
    /// Number of bytes per channel for the texture.
    const BYTES_PER_CHANNEL: u32 = 4;
    /// Number of bytes per pixel for the texture.
    pub const BYTES_PER_PIXEL: u32 = Self::NUM_CHANNELS * Self::BYTES_PER_CHANNEL;

    /// Initializes a new GPU compute pipeline.
    ///
    /// # Arguments
    ///
    /// - `window`: A reference to the window used for the pipeline.
    /// - `faraday_data`: The Faraday data to be used in the pipeline. This
    ///   struct contains the data that will be passed to the compute shader.
    pub fn new(window: &Window, faraday_data: FaradayData) -> Self {
        // Initialize utilities
        let device = window.device();
        let msaa_samples = window.msaa_samples();
        let (width, height) = window.inner_size_pixels();
        let global_data = GlobalData::default();

        // Load shader
        let compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/compute.wgsl"));
        let render_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/render.wgsl"));
        let post_processing_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/post_processing.wgsl"));

        // Create texture
        let texture = Self::create_texture(device, [width, height], Self::TEXTURE_FORMAT);
        let texture_view = texture.view().build();

        // Create data buffer
        let faraday_data_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Faraday Data Uniforms Buffer"),
            contents: faraday_data.as_bytes(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let global_data_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("Global Data Buffer"),
            contents: global_data.as_bytes(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create the compute bind group
        let compute_bgl = Self::create_compute_bgl(device, &texture);
        let compute_bg = Self::create_compute_bg(
            device,
            &compute_bgl,
            &texture_view,
            &faraday_data_buffer,
            &global_data_buffer,
        );

        // Create the compute pipeline
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bgl],
                push_constant_ranges: &[],
            });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "cs_main",
        });

        let min_max_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Min/Max Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &post_processing_shader,
            entry_point: "cs_min_max",
        });

        let recalibrate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Recalibrate Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &post_processing_shader,
                entry_point: "cs_recalibrate",
            });

        let histogram_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Histogram Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &post_processing_shader,
            entry_point: "cs_histogram",
        });

        let cdf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CDF Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &post_processing_shader,
            entry_point: "cs_cdf",
        });

        let equalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Equalize Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &post_processing_shader,
            entry_point: "cs_equalize",
        });

        // Create the render bind group
        let render_bgl = Self::create_render_bgl(device, &texture);
        let render_bg = Self::create_render_bg(device, &render_bgl, &texture_view);

        // Create the pipeline layout
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            });

        let render_pipeline =
            wgpu::RenderPipelineBuilder::from_layout(&render_pipeline_layout, &render_shader)
                .vertex_entry_point("vs_main")
                .fragment_shader(&render_shader)
                .fragment_entry_point("fs_main")
                .color_format(Frame::TEXTURE_FORMAT)
                .color_blend(wgpu::BlendComponent::REPLACE)
                .alpha_blend(wgpu::BlendComponent::REPLACE)
                .primitive_topology(wgpu::PrimitiveTopology::TriangleList)
                .front_face(wgpu::FrontFace::Ccw)
                .cull_mode(Some(wgpu::Face::Back))
                .sample_count(msaa_samples)
                .build(device);

        GPUPipeline {
            texture,
            texture_view,
            faraday_data_buffer,
            global_data_buffer,
            // Generate texture
            compute_bgl,
            compute_bg,
            compute_pipeline,
            // Post-processing
            min_max_pipeline,
            recalibrate_pipeline,
            histogram_pipeline,
            cdf_pipeline,
            equalize_pipeline,
            // Render
            render_bgl,
            render_bg,
            render_pipeline,
        }
    }

    /// Dispatches the compute pipeline for rendering.
    ///
    /// # Arguments
    ///
    /// - `encoder`: A mutable reference to the command encoder used for rendering.
    /// - `frame_size`: The size of the frame to be rendered. This is used to set
    ///   the workgroup size for the compute shader.
    pub fn dispatch_compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        frame_size: [u32; 2],
    ) {
        let (w, h) = (frame_size[0], frame_size[1]);
        let dispatch_x = w.div_ceil(Self::WORKGROUP_SIZE);
        let dispatch_y = h.div_ceil(Self::WORKGROUP_SIZE);

        // Generate texture
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Clear global data buffer
        queue.write_buffer(
            &self.global_data_buffer,
            0,
            GlobalData::default().as_bytes(),
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Get Post-Processing Pass"),
            });

            // Get min/max of texture
            pass.set_pipeline(&self.min_max_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

            // Recalibrate texture
            pass.set_pipeline(&self.recalibrate_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

            // Generate histogram
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

            // Generate CDF
            pass.set_pipeline(&self.cdf_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            // Equalize texture
            pass.set_pipeline(&self.equalize_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    /// Dispatches the render pipeline for rendering.
    ///
    /// # Arguments
    ///
    /// - `encoder`: A mutable reference to the command encoder used for rendering.
    /// - `target_texture_view`: A reference to the texture view used for rendering.
    ///   This is the texture view on which the output of the compute shader will
    ///   be drawn.
    pub fn dispatch_render(
        &self,
        mut encoder: RefMut<'_, wgpu::CommandEncoder>,
        target_texture_view: &wgpu::TextureView,
    ) {
        let mut render_pass = wgpu::RenderPassBuilder::new()
            .color_attachment(target_texture_view, |color| color)
            .begin(&mut encoder);
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.render_bg, &[]);
        render_pass.draw(0..3, 0..1); // Draw the full-screen triangle
    }

    pub fn save_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        filename: &str,
    ) -> Result<(), &'static str> {
        let dimensions = self.texture.size();
        let (w, h) = (dimensions[0], dimensions[1]);

        // Create readback buffer
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Readback Buffer"),
            size: (w * h * Self::BYTES_PER_PIXEL) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create a new encoder for the transfer pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Texture Save Encoder"),
        });

        // Copy texture to buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(Self::BYTES_PER_PIXEL * w),
                    rows_per_image: Some(h),
                },
            },
            self.texture.extent(),
        );

        // Submit the encoder to the queue
        queue.submit(Some(encoder.finish()));

        // Map the buffer synchronously
        let slice = readback_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("Failed to map buffer");
        });

        // Poll the device until the buffer is mapped
        device.poll(wgpu::Maintain::Wait);

        // Read the mapped data
        let data = slice.get_mapped_range();

        // Convert the vector of bytes to a vector of f32
        let mut floats = Vec::with_capacity(data.len() / Self::NUM_CHANNELS as usize);
        for bytes in data.chunks_exact(Self::NUM_CHANNELS as usize) {
            let float = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            floats.push(float);
        }

        // Convert f32 RGBA to u8 RGBA
        let mut pixels_u8 = Vec::with_capacity((w * h * Self::BYTES_PER_PIXEL) as usize);
        for chunk in floats.chunks_exact(4) {
            let r = (chunk[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (chunk[1].clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (chunk[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            let a = (chunk[3].clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels_u8.extend_from_slice(&[r, g, b, a]);
        }

        // Create an image buffer from the u8 data
        let img = match ImageBuffer::<image::Rgba<u8>, _>::from_raw(w, h, pixels_u8) {
            Some(img) => img,
            None => {
                return Err("Failed to convert buffer to ImageBuffer");
            }
        };

        // Save the image as a PNG file
        if img.save(filename).is_err() {
            return Err("Failed to save texture to file");
        }

        // Unmap the buffer
        drop(data);
        readback_buffer.unmap();

        Ok(())
    }

    /// If needed, recreates the texture, its view, and the bind groups
    pub fn check_resize(&mut self, device: &wgpu::Device, new_size: [u32; 2]) {
        if self.texture.size() != new_size {
            self.resize(device, new_size);
        }
    }

    /// Recreates the texture, its view, and the bind groups
    pub fn resize(&mut self, device: &wgpu::Device, new_size: [u32; 2]) {
        // Recreate the texture & view
        self.texture = Self::create_texture(device, new_size, self.texture.format());
        self.texture_view = self.texture.view().build();

        // Rebuild the compute bind group
        self.compute_bg = Self::create_compute_bg(
            device,
            &self.compute_bgl,
            &self.texture_view,
            &self.faraday_data_buffer,
            &self.global_data_buffer,
        );

        // Rebuild the render bind group
        self.render_bg = Self::create_render_bg(device, &self.render_bgl, &self.texture_view);
    }

    /// Updates the Faraday data buffer with new data.
    ///
    /// # Arguments
    ///
    /// - `device`: A reference to the device used for the pipeline.
    /// - `encoder`: A mutable reference to the command encoder used for the
    ///   pipeline.
    /// - `faraday_data`: The new Faraday data to be used in the pipeline. This
    ///   data will replace the old data in the compute shader.
    pub fn update_faraday_data(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        faraday_data: FaradayData,
    ) {
        let faraday_data_storage_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Faraday Data Uniforms Buffer"),
            contents: faraday_data.as_bytes(),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Copy the new uniforms buffer to the uniform buffer.
        encoder.copy_buffer_to_buffer(
            &faraday_data_storage_buffer,
            0,
            &self.faraday_data_buffer,
            0,
            std::mem::size_of::<FaradayData>() as wgpu::BufferAddress,
        );
    }

    /// Creates a new texture for the compute and render pipelines.
    fn create_texture(
        device: &wgpu::Device,
        size: [u32; 2],
        texture_format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        wgpu::TextureBuilder::new()
            .size(size)
            .dimension(wgpu::TextureDimension::D2)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT)
            .mip_level_count(1)
            .sample_count(1)
            .format(texture_format)
            .usage(
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            )
            .build(device)
    }

    /// Creates a new bind group layout for the compute pipeline.
    fn create_compute_bgl(device: &wgpu::Device, texture: &wgpu::Texture) -> wgpu::BindGroupLayout {
        wgpu::BindGroupLayoutBuilder::new()
            .storage_texture(
                wgpu::ShaderStages::COMPUTE,
                texture.format(),
                texture.view_dimension(),
                wgpu::StorageTextureAccess::ReadWrite,
            )
            .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)
            .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false)
            .build(device)
    }

    /// Creates a new bind group for the compute pipeline.
    fn create_compute_bg(
        device: &wgpu::Device,
        compute_bgl: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
        faraday_data_buffer: &wgpu::Buffer,
        global_data_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        wgpu::BindGroupBuilder::new()
            .texture_view(texture_view)
            .binding(faraday_data_buffer.as_entire_binding())
            .binding(global_data_buffer.as_entire_binding())
            .build(device, compute_bgl)
    }

    /// Creates a new bind group layout for the render pipeline.
    fn create_render_bgl(device: &wgpu::Device, texture: &wgpu::Texture) -> wgpu::BindGroupLayout {
        wgpu::BindGroupLayoutBuilder::new()
            .texture_from(wgpu::ShaderStages::FRAGMENT, texture)
            .build(device)
    }

    /// Creates a new bind group for the render pipeline.
    fn create_render_bg(
        device: &wgpu::Device,
        render_bgl: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        wgpu::BindGroupBuilder::new()
            .texture_view(texture_view)
            .build(device, render_bgl)
    }

    /// Returns the size of the texture.
    pub fn texture_size(&self) -> [u32; 2] {
        self.texture.size()
    }

    /// Returns the extent of the texture.
    pub fn texture_extent(&self) -> wgpu::Extent3d {
        self.texture.extent()
    }
}
