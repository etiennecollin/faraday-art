use std::cell::RefMut;

use nannou::prelude::*;

use super::faraday::FaradayData;

pub struct GPUPipeline {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    faraday_data_buffer: wgpu::Buffer,
    compute_bgl: wgpu::BindGroupLayout,
    compute_bg: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    render_bgl: wgpu::BindGroupLayout,
    render_bg: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
}

impl GPUPipeline {
    /// Size of the workgroup for the compute shader.
    const WORKGROUP_SIZE: u32 = 16;

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

        // Load shader
        let compute_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/compute.wgsl"));
        let render_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/render.wgsl"));

        // Create texture
        let texture =
            Self::create_texture(device, [width, height], wgpu::TextureFormat::Rgba32Float);
        let texture_view = texture.view().build();

        // Create data buffer
        let faraday_data_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Faraday Data Uniforms Buffer"),
            contents: faraday_data.as_bytes(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create the compute bind group
        let compute_bgl = Self::create_compute_bgl(device, &texture);
        let compute_bg =
            Self::create_compute_bg(device, &compute_bgl, &texture_view, &faraday_data_buffer);

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
            compute_bgl,
            compute_bg,
            compute_pipeline,
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
    pub fn dispatch_compute(&self, encoder: &mut wgpu::CommandEncoder, frame_size: [u32; 2]) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
        });
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.compute_bg, &[]);

        // Set workgroup size
        let (w, h) = (frame_size[0], frame_size[1]);
        compute_pass.dispatch_workgroups(
            w.div_ceil(Self::WORKGROUP_SIZE),
            h.div_ceil(Self::WORKGROUP_SIZE),
            1,
        );
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
            .usage(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING)
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
            .build(device)
    }

    /// Creates a new bind group for the compute pipeline.
    fn create_compute_bg(
        device: &wgpu::Device,
        compute_bgl: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
        faraday_data_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        wgpu::BindGroupBuilder::new()
            .texture_view(texture_view)
            .binding(faraday_data_buffer.as_entire_binding())
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
}
