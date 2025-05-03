use std::cell::RefCell;

use faraday_art::utils::{faraday::*, math::*, pipeline::GPUPipeline};
use nannou::prelude::*;
use nannou_egui::{
    Egui,
    egui::{self},
};

const WINDOW_SIZE: (u32, u32) = (512, 512);
type FloatChoice = f32;

struct State {
    needs_compute: RefCell<bool>,
    continuous_compute: bool,
    /// Relative position of the mouse as a percentage of the function range
    mouse_position: (FloatChoice, FloatChoice),
    zoom_speed: FloatChoice,
    shift_speed: u32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            needs_compute: true.into(),
            continuous_compute: false,
            zoom_speed: 0.001,
            shift_speed: 50,
            mouse_position: (0.0, 0.0),
        }
    }
}

struct Model {
    egui: Egui,
    state: State,
    pipeline: RefCell<GPUPipeline>,
    faraday_data: FaradayData,
    update_faraday_data: RefCell<bool>,
}

fn main() {
    nannou::app(model).update(update).run()
}

fn model(app: &App) -> Model {
    // Set GPU device descriptor
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("Point Cloud Renderer Device"),
        features: wgpu::Features::default()
            // | wgpu::Features::SHADER_F64
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        limits: wgpu::Limits {
            // max_storage_buffer_binding_size: 2 << 30, // To support big point clouds
            // max_texture_dimension_2d: 2 << 14,        // To support the big 9x3 4K display wall
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
        .device_descriptor(descriptor)
        .build()
        .unwrap();

    let window = app.window(window_id).unwrap();
    let state = State::default();
    let egui = Egui::from_window(&window);

    let faraday_data = FaradayData::default();
    let pipeline = GPUPipeline::new(&window, faraday_data);

    Model {
        egui,
        state,
        pipeline: pipeline.into(),
        faraday_data,
        update_faraday_data: false.into(),
    }
}

fn view(_app: &App, model: &Model, frame: Frame) {
    // Render the image
    let pipeline = model.pipeline.borrow();
    let encoder = frame.command_encoder();
    let target_view = frame.texture_view();
    pipeline.dispatch_render(encoder, target_view);

    // Update the egui
    model.egui.draw_to_frame(&frame).unwrap();
}

fn update(app: &App, model: &mut Model, update: Update) {
    // Check if the we need to redraw the frame
    let state = &model.state;
    if *state.needs_compute.borrow() || state.continuous_compute {
        let mut pipeline = model.pipeline.borrow_mut();
        let window = app.main_window();
        let (device, queue) = {
            let pair = window.device_queue_pair();
            (pair.device(), pair.queue())
        };

        // Create a new encoder for the compute pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        // Check if the faraday data needs to be updated
        if *model.update_faraday_data.borrow() {
            pipeline.update_faraday_data(device, &mut encoder, model.faraday_data);
            model.update_faraday_data.replace(false);
        }

        // Dispatch the compute pipeline
        let (width, height) = app.main_window().inner_size_pixels();
        pipeline.dispatch_compute(&mut encoder, [width, height]);

        // Submit the command buffer
        let cmd_buf = encoder.finish();
        queue.submit(Some(cmd_buf));

        state.needs_compute.replace(false);
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

            ui.separator();

            ui.checkbox(&mut state.continuous_compute, "Continuous Redraw");

            if ui.button("Update").clicked() {
                state.needs_compute.replace(true);
            }

            // TODO: Implement downloading image from GPU to CPU buffer
            // if ui.button("Save").clicked() {
            //     state
            //         .image
            //         .save(get_save_path(&app.exe_name().unwrap()))
            //         .unwrap();
            // }
        });
}

fn resized(app: &App, model: &mut Model, dim: Vec2) {
    let mut pipeline = model.pipeline.borrow_mut();
    let window = app.main_window();
    let device = window.device();

    // When the window size changes, recreate our depth texture to match.
    pipeline.resize(device, [dim.x as u32 * 2, dim.y as u32 * 2]);

    // Signify that we need to redraw the frame.
    model.state.needs_compute.replace(true);
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    let state = &mut model.state;
    match key {
        Key::Left => {
            let current_x_range = model.faraday_data.get_x_range();
            let shift_x = get_shift_speed(current_x_range, state.shift_speed);
            let new_x_range = shift(current_x_range, -shift_x);
            model.faraday_data.update_x_range(new_x_range);
            model.update_faraday_data.replace(true);
            state.needs_compute.replace(true);
        }
        Key::Right => {
            let current_x_range = model.faraday_data.get_x_range();
            let shift_x = get_shift_speed(current_x_range, state.shift_speed);
            let new_x_range = shift(current_x_range, shift_x);
            model.faraday_data.update_x_range(new_x_range);
            model.update_faraday_data.replace(true);
            state.needs_compute.replace(true);
        }
        Key::Up => {
            let current_y_range = model.faraday_data.get_y_range();
            let shift_y = get_shift_speed(current_y_range, state.shift_speed);
            let new_y_range = shift(current_y_range, shift_y);
            model.faraday_data.update_y_range(new_y_range);
            model.update_faraday_data.replace(true);
            state.needs_compute.replace(true);
        }
        Key::Down => {
            let current_y_range = model.faraday_data.get_y_range();
            let shift_y = get_shift_speed(current_y_range, state.shift_speed);
            let new_y_range = shift(current_y_range, -shift_y);
            model.faraday_data.update_y_range(new_y_range);
            model.update_faraday_data.replace(true);
            state.needs_compute.replace(true);
        }
        Key::Plus | Key::Equals => {
            let zoom_factor = 1.0 - 10.0 * state.zoom_speed;
            let current_x_range = model.faraday_data.get_x_range();
            let current_y_range = model.faraday_data.get_y_range();
            let (new_x_range, new_y_range) =
                zoom_relative(current_x_range, current_y_range, zoom_factor, (0.5, 0.5));
            model.faraday_data.update_x_range(new_x_range);
            model.faraday_data.update_y_range(new_y_range);
            model.update_faraday_data.replace(true);
            state.needs_compute.replace(true);
        }
        Key::Minus => {
            let zoom_factor = 1.0 + 10.0 * state.zoom_speed;
            let current_x_range = model.faraday_data.get_x_range();
            let current_y_range = model.faraday_data.get_y_range();
            let (new_x_range, new_y_range) =
                zoom_relative(current_x_range, current_y_range, zoom_factor, (0.5, 0.5));
            model.faraday_data.update_x_range(new_x_range);
            model.faraday_data.update_y_range(new_y_range);
            model.update_faraday_data.replace(true);
            state.needs_compute.replace(true);
        }
        Key::Q => app.quit(),
        // TODO: Implement downloading image from GPU to CPU buffer
        // Key::S => model
        //     .state
        //     .image
        //     .save(get_save_path(&app.exe_name().unwrap()))
        //     .unwrap(),
        Key::Return => drop(state.needs_compute.replace(true)),
        _other_key => {}
    }
}

fn mouse_wheel(_app: &App, model: &mut Model, delta: MouseScrollDelta, _phase: TouchPhase) {
    let state = &mut model.state;

    let current_x_range = model.faraday_data.get_x_range();
    let current_y_range = model.faraday_data.get_y_range();

    // Compute the zoom factor based on the mouse wheel delta
    let zoom_factor = match delta {
        MouseScrollDelta::LineDelta(_, y) => 1.0 + y as FloatChoice * state.zoom_speed,
        MouseScrollDelta::PixelDelta(pos) => 1.0 + pos.y as FloatChoice * state.zoom_speed,
    };

    // Compute the new x and y ranges based on the zoom factor and mouse position
    let (new_x_range, new_y_range) = zoom_relative(
        current_x_range,
        current_y_range,
        zoom_factor,
        state.mouse_position,
    );

    // Update the x and y ranges in the FaradayData
    model.faraday_data.update_x_range(new_x_range);
    model.faraday_data.update_y_range(new_y_range);
    model.update_faraday_data.replace(true);
    state.needs_compute.replace(true);
}

fn mouse_moved(app: &App, model: &mut Model, pos: Point2) {
    let state = &mut model.state;
    let (_, max_x, _, max_y) = app.window_rect().l_r_b_t();
    let (width, height) = app.window_rect().w_h();
    let x_percent = (pos.x + max_x) / width;
    let y_percent = (pos.y + max_y) / height;
    state.mouse_position = (x_percent as FloatChoice, y_percent as FloatChoice);
}
