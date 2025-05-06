use std::cell::RefCell;

use faraday_art::{
    FloatChoice, MAX_ZOOM_DELTA, get_save_path,
    utils::{math::*, pipeline::GPUPipeline, pipeline_buffers::ComputeData},
};
use nannou::prelude::*;
use nannou_egui::{
    Egui,
    egui::{self},
};

/// The size of the window in pixels.
const WINDOW_SIZE: (u32, u32) = (1024, 1024);

struct State {
    /// Whether to compute the image continuously or not.
    continuous_compute: bool,
    /// Relative position of the mouse as a percentage of the function range.
    mouse_pos: (FloatChoice, FloatChoice),
    /// Previous position of the mouse as a percentage of the function range.
    /// This is used to compute the delta of the mouse movement.
    prev_drag_pos: (FloatChoice, FloatChoice),
    /// True if the mouse is currently being dragged.
    dragging: bool,
    /// Zoom speed factor.
    zoom_speed: FloatChoice,
    /// Shift speed factor.
    shift_speed: u32,
    /// Whether to save the image or not.
    save_image: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            continuous_compute: false,
            prev_drag_pos: (0.0, 0.0),
            dragging: false,
            zoom_speed: 0.001,
            shift_speed: 50,
            mouse_pos: (0.0, 0.0),
            save_image: false,
        }
    }
}

struct Model {
    egui: Egui,
    state: State,
    pipeline: RefCell<GPUPipeline>,
    /// Struct containing the data to be processed by the compute shader.
    compute_data: ComputeData,
    /// Indicates whether the compute data buffer needs to be updated.
    update_compute_data_buffer: RefCell<bool>,
    /// Indicates whether the texture needs to be recomputed.
    recompute_texture: RefCell<bool>,
}

fn main() {
    nannou::app(model).update(update).run()
}

fn model(app: &App) -> Model {
    let mut gpu_features =
        wgpu::Features::default() | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;

    #[cfg(feature = "f64")]
    {
        gpu_features |= wgpu::Features::SHADER_F64; // To support f64 in shaders
    }

    // Set GPU device descriptor
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("Point Cloud Renderer Device"),
        features: gpu_features,
        limits: wgpu::Limits {
            // max_texture_dimension_2d: 2 << 14, // To support the big 9x3 4K display wall
            ..Default::default()
        },
    };

    let window_id = app
        .new_window()
        .size(WINDOW_SIZE.0, WINDOW_SIZE.1)
        .view(view)
        .resized(resized)
        .raw_event(raw_window_event)
        .key_pressed(key_pressed)
        .mouse_wheel(mouse_wheel)
        .mouse_moved(mouse_moved)
        .mouse_pressed(mouse_pressed)
        .mouse_released(mouse_released)
        .device_descriptor(descriptor)
        .build()
        .unwrap();

    let window = app.window(window_id).unwrap();
    let state = State::default();
    let egui = Egui::from_window(&window);

    let compute_data = ComputeData::default();
    let pipeline = GPUPipeline::new(&window, compute_data);

    Model {
        egui,
        state,
        pipeline: pipeline.into(),
        compute_data,
        update_compute_data_buffer: false.into(),
        recompute_texture: true.into(),
    }
}

fn view(_app: &App, model: &Model, frame: Frame) {
    // Render the texture
    let encoder = frame.command_encoder();
    let target_view = frame.texture_view();
    model
        .pipeline
        .borrow()
        .dispatch_render(encoder, target_view);

    // Update the egui
    model.egui.draw_to_frame(&frame).unwrap();
}

fn update(app: &App, model: &mut Model, update: Update) {
    let state = &mut model.state;

    // Check if a texture recompute is requested
    if *model.recompute_texture.borrow() || state.continuous_compute {
        // Get the device and queue from the window
        let window = app.main_window();
        let (device, queue) = {
            let pair = window.device_queue_pair();
            (pair.device(), pair.queue())
        };

        let mut pipeline = model.pipeline.borrow_mut();

        // Create a new encoder for the compute pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        // Check if the data buffer needs to be updated
        if *model.update_compute_data_buffer.borrow() {
            pipeline.update_compute_data_buffer(device, &mut encoder, model.compute_data);
            model.update_compute_data_buffer.replace(false);
        }

        // Dispatch the compute pipeline
        let (width, height) = app.main_window().inner_size_pixels();
        pipeline.dispatch_compute(&mut encoder, queue, [width, height]);

        // Submit the command buffer
        queue.submit(Some(encoder.finish()));

        model.recompute_texture.replace(false);
    }

    // Check if an image save is requested
    if state.save_image {
        // Get the device and queue from the window
        let window = app.main_window();
        let (device, queue) = {
            let pair = window.device_queue_pair();
            (pair.device(), pair.queue())
        };

        // Save the image to a file
        let pipeline = model.pipeline.borrow_mut();
        let filename = get_save_path(&app.exe_name().unwrap());
        if pipeline.save_texture(device, queue, &filename).is_err() {
            println!("Error saving image");
        } else {
            println!("Image saved successfully to: {}", filename);
        }

        state.save_image = false;
    }

    // Update egui
    model.egui.set_elapsed_time(update.since_start);
    update_egui(model, app);
}

fn update_egui(model: &mut Model, _app: &App) {
    let ctx = model.egui.begin_frame();
    let state = &mut model.state;

    // Generate the settings window
    egui::Window::new("Settings")
        .default_width(0.0)
        .show(&ctx, |ui| {
            ui.label("Zoom speed:");
            ui.add(egui::Slider::new(&mut state.zoom_speed, 0.0001..=0.1));

            ui.label("Shift speed:");
            ui.add(egui::Slider::new(&mut state.shift_speed, 10..=100));

            ui.label("Max iterations:");
            let old_max_iterations = model.compute_data.max_iter;
            ui.add(egui::Slider::new(
                &mut model.compute_data.max_iter,
                200..=2000,
            ));
            if old_max_iterations != model.compute_data.max_iter {
                model.update_compute_data_buffer.replace(true);
                model.recompute_texture.replace(true);
            }

            ui.label("dt:");
            let old_dt = model.compute_data.dt;
            ui.add(egui::Slider::new(&mut model.compute_data.dt, 0.01..=1.0));
            if old_dt != model.compute_data.dt {
                model.update_compute_data_buffer.replace(true);
                model.recompute_texture.replace(true);
            }

            ui.label("mu:");
            let old_mu = model.compute_data.mu;
            ui.add(egui::Slider::new(&mut model.compute_data.mu, 0.0..=10.0));
            if old_mu != model.compute_data.mu {
                model.update_compute_data_buffer.replace(true);
                model.recompute_texture.replace(true);
            }

            ui.separator();

            ui.checkbox(&mut state.continuous_compute, "Continuous Redraw");

            if ui.button("Update").clicked() {
                model.recompute_texture.replace(true);
            }

            if ui.button("Save").clicked() {
                state.save_image = true;
            }
        });
}

fn resized(app: &App, model: &mut Model, _dim: Vec2) {
    let window = app.main_window();
    let device = window.device();
    let (width, height) = window.inner_size_pixels();

    // When the window size changes, recreate our texture to match and
    // ask to recompute the image
    model.pipeline.borrow_mut().resize(device, [width, height]);
    model.recompute_texture.replace(true);
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    let state = &mut model.state;

    // When we shift or zoom, we need to update the data buffer and
    // ask to recompute the texture
    match key {
        Key::Left => {
            let current_x_range = model.compute_data.get_x_range();
            let shift_x = get_shift_speed(current_x_range, state.shift_speed);
            let new_x_range = shift(current_x_range, -shift_x);
            model.compute_data.update_x_range(new_x_range);
            model.update_compute_data_buffer.replace(true);
            model.recompute_texture.replace(true);
        }
        Key::Right => {
            let current_x_range = model.compute_data.get_x_range();
            let shift_x = get_shift_speed(current_x_range, state.shift_speed);
            let new_x_range = shift(current_x_range, shift_x);
            model.compute_data.update_x_range(new_x_range);
            model.update_compute_data_buffer.replace(true);
            model.recompute_texture.replace(true);
        }
        Key::Up => {
            let current_y_range = model.compute_data.get_y_range();
            let shift_y = get_shift_speed(current_y_range, state.shift_speed);
            let new_y_range = shift(current_y_range, shift_y);
            model.compute_data.update_y_range(new_y_range);
            model.update_compute_data_buffer.replace(true);
            model.recompute_texture.replace(true);
        }
        Key::Down => {
            let current_y_range = model.compute_data.get_y_range();
            let shift_y = get_shift_speed(current_y_range, state.shift_speed);
            let new_y_range = shift(current_y_range, -shift_y);
            model.compute_data.update_y_range(new_y_range);
            model.update_compute_data_buffer.replace(true);
            model.recompute_texture.replace(true);
        }
        Key::Plus | Key::Equals => {
            let zoom_factor = 1.0 - 10.0 * state.zoom_speed;
            let current_x_range = model.compute_data.get_x_range();
            let current_y_range = model.compute_data.get_y_range();
            let (new_x_range, new_y_range) =
                zoom_relative(current_x_range, current_y_range, zoom_factor, (0.5, 0.5));
            model.compute_data.update_x_range(new_x_range);
            model.compute_data.update_y_range(new_y_range);
            model.update_compute_data_buffer.replace(true);
            model.recompute_texture.replace(true);
        }
        Key::Minus => {
            let zoom_factor = 1.0 + 10.0 * state.zoom_speed;
            let current_x_range = model.compute_data.get_x_range();
            let current_y_range = model.compute_data.get_y_range();
            let (new_x_range, new_y_range) =
                zoom_relative(current_x_range, current_y_range, zoom_factor, (0.5, 0.5));
            model.compute_data.update_x_range(new_x_range);
            model.compute_data.update_y_range(new_y_range);
            model.update_compute_data_buffer.replace(true);
            model.recompute_texture.replace(true);
        }
        Key::Q => app.quit(),
        Key::S => state.save_image = true,
        Key::Return => drop(model.recompute_texture.replace(true)),
        _other_key => {}
    }
}

fn mouse_wheel(_app: &App, model: &mut Model, delta: MouseScrollDelta, _phase: TouchPhase) {
    let state = &mut model.state;
    let current_x_range = model.compute_data.get_x_range();
    let current_y_range = model.compute_data.get_y_range();

    // Compute the zoom factor based on the mouse wheel delta
    let zoom_factor = match delta {
        MouseScrollDelta::LineDelta(_, y) => 1.0 + y as FloatChoice * state.zoom_speed,
        MouseScrollDelta::PixelDelta(pos) => 1.0 + pos.y as FloatChoice * state.zoom_speed,
    };

    // Compute the new x/y ranges based on the zoom factor and mouse position
    let (new_x_range, new_y_range) = zoom_relative(
        current_x_range,
        current_y_range,
        zoom_factor,
        state.mouse_pos,
    );

    // Make sure not to zoom too much to avoid numerical issues
    if (new_x_range.1 - new_x_range.0).abs() < MAX_ZOOM_DELTA
        || (new_y_range.1 - new_y_range.0).abs() < MAX_ZOOM_DELTA
    {
        return;
    }

    // Update the x/y ranges in the data buffer and recompute the texture
    model.compute_data.update_x_range(new_x_range);
    model.compute_data.update_y_range(new_y_range);
    model.update_compute_data_buffer.replace(true);
    model.recompute_texture.replace(true);
}

fn mouse_moved(app: &App, model: &mut Model, pos: Point2) {
    let state = &mut model.state;
    let (w, h) = app.window_rect().w_h();

    // Convert centered coords (-w/2..w/2) to [0..1]
    let x_norm = (pos.x + w * 0.5) / w;
    let y_norm = (pos.y + h * 0.5) / h;

    // Store the normalized mouse position
    state.mouse_pos = (x_norm as FloatChoice, y_norm as FloatChoice);

    // If we are dragging, compute how much the mouse moved (in normalized space)
    if state.dragging {
        let (prev_x, prev_y) = state.prev_drag_pos;
        let dx = state.mouse_pos.0 - prev_x;
        let dy = state.mouse_pos.1 - prev_y;

        // Get current ranges
        let (x0, x1) = model.compute_data.get_x_range();
        let (y0, y1) = model.compute_data.get_y_range();

        // Compute how much to shift in "range units"
        let range_w = x1 - x0;
        let range_h = y1 - y0;
        let shift_x = -dx * range_w;
        let shift_y = -dy * range_h;

        // Apply shift to the viewport
        let new_x_range = shift((x0, x1), shift_x);
        let new_y_range = shift((y0, y1), shift_y);
        model.compute_data.update_x_range(new_x_range);
        model.compute_data.update_y_range(new_y_range);

        // Ask to update data buffer and recompute the texture
        model.update_compute_data_buffer.replace(true);
        model.recompute_texture.replace(true);

        // Remember this pos for the next delta
        state.prev_drag_pos = state.mouse_pos;
    }
}

fn mouse_pressed(_app: &App, model: &mut Model, _button: MouseButton) {
    let state = &mut model.state;

    // Start a mouse drag
    state.dragging = true;
    state.prev_drag_pos = state.mouse_pos;
}

fn mouse_released(_app: &App, model: &mut Model, _button: MouseButton) {
    let state = &mut model.state;

    // End the mouse drag
    state.dragging = false;
}
