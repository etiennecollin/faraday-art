use std::cell::RefCell;

use faraday_art::{
    get_save_path,
    utils::{faraday::*, images::*, math::*},
};
use nannou::{
    image::{self, ImageBuffer, RgbaImage},
    prelude::*,
    rand::*,
};
use nannou_egui::{
    Egui, FrameCtx,
    egui::{self},
};
use rayon::iter::{ParallelBridge, ParallelIterator};

const WINDOW_SIZE: (u32, u32) = (512, 512);

struct State {
    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    redraw: bool,
    continuous_redraw: bool,
    no_redraw: bool,
    /// Initial render range in x for function
    x_range: (f64, f64),
    /// Initial render range in y for function
    y_range: (f64, f64),
    /// Relative position of the mouse as a percentage of the function range
    mouse_position: (f64, f64),
    zoom_speed: f64,
    shift_speed: u32,
    // Problem specific
    max_iter: usize,
    num_particles: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            image: ImageBuffer::new(WINDOW_SIZE.0, WINDOW_SIZE.1),
            redraw: true,
            continuous_redraw: false,
            no_redraw: false,
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
            zoom_speed: 0.001,
            shift_speed: 100,
            mouse_position: (0.0, 0.0),
            // Problem specific
            max_iter: 20_000,
            num_particles: 20_000,
        }
    }
}

/// Simple DLA implementation on a grayscale image grid.
/// Returns a Vec<Vec<f64>> with 255 for stuck particles, 0 for empty.
fn generate_faraday(width: usize, height: usize, state: &State) -> Vec<Vec<f64>> {
    let mut grid = vec![vec![0.0; width]; height];
    let mut rng = rand::thread_rng();

    let cx = width / 2;
    let cy = height / 2;
    grid[cy][cx] = 255.0;

    let mut max_radius_sq = 1.0;

    for _ in 0..state.num_particles {
        // Spawn just outside the current cluster
        let spawn_radius = max_radius_sq.sqrt() + 5.0;
        let angle = rng.gen_range(0.0..TAU_F64);
        let mut x = (cx as f64 + spawn_radius * angle.cos()).round() as isize;
        let mut y = (cy as f64 + spawn_radius * angle.sin()).round() as isize;

        // Clamp within bounds
        x = x.clamp(1, (width - 2) as isize);
        y = y.clamp(1, (height - 2) as isize);

        for _ in 0..state.max_iter {
            // Random walk
            const DXS: [isize; 8] = [-1, 0, 1, -1, 1, -1, 0, 1];
            const DYS: [isize; 8] = [-1, -1, -1, 0, 0, 1, 1, 1];
            let dir = rng.gen_range(0..8);
            x = (x + DXS[dir]).clamp(1, (width - 2) as isize);
            y = (y + DYS[dir]).clamp(1, (height - 2) as isize);

            let x_usize = x as usize;
            let y_usize = y as usize;

            grid[y_usize][x_usize] += 0.1;

            // Abandon if too far
            let dx = x_usize as isize - cx as isize;
            let dy = y_usize as isize - cy as isize;
            let r2 = (dx * dx + dy * dy) as f64;
            if r2 > (width as f64 / 2.0).powi(2) {
                break;
            }

            // Check if adjacent to cluster
            let mut stuck = false;
            for nx in x_usize.saturating_sub(1)..=x_usize + 1 {
                for ny in y_usize.saturating_sub(1)..=y_usize + 1 {
                    if grid[ny][nx] >= 255.0 {
                        stuck = true;
                        break;
                    }
                }
                if stuck {
                    break;
                }
            }

            if stuck {
                grid[y_usize][x_usize] = 255.0;

                // Update max radius
                max_radius_sq = max_radius_sq.max(r2);
                break;
            }
        }
    }

    grid
}

struct Model {
    egui: Egui,
    state: State,
    // shader_pipeline: RefCell<GPUPipeline>,
}

fn main() {
    nannou::app(model).update(update).run()
}

fn model(app: &App) -> Model {
    let window_id = app
        .new_window()
        .size(WINDOW_SIZE.0, WINDOW_SIZE.1)
        .view(view)
        .raw_event(raw_window_event)
        .key_pressed(key_pressed)
        .mouse_wheel(mouse_wheel)
        .mouse_moved(mouse_moved)
        .build()
        .unwrap();

    let window = app.window(window_id).unwrap();
    let state = State::default();
    let egui = Egui::from_window(&window);

    // let gpu_pipeline = GPUPipeline::new(app, window_id);

    Model { egui, state }
}

fn update(app: &App, model: &mut Model, update: Update) {
    let egui = &mut model.egui;
    let state = &mut model.state;
    let (width, height) = app.window_rect().w_h();

    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    update_egui(ctx, state, app);

    if (state.redraw || state.continuous_redraw) && !state.no_redraw {
        let mut image_array = generate_faraday(width as usize, height as usize, state);
        recalibrate(&mut image_array);
        equalize(&mut image_array, 0.0);
        state.image = to_image(image_array);
        state.redraw = false;
    }
}

fn update_egui(ctx: FrameCtx, state: &mut State, app: &App) {
    // Generate the settings window
    egui::Window::new("Settings")
        .default_width(0.0)
        .show(&ctx, |ui| {
            ui.label("Zoom speed:");
            ui.add(egui::Slider::new(&mut state.zoom_speed, 0.0001..=0.1));

            ui.label("Shift speed:");
            ui.add(egui::Slider::new(&mut state.shift_speed, 10..=100));

            ui.separator();

            ui.checkbox(&mut state.continuous_redraw, "Continuous Redraw");
            ui.checkbox(&mut state.no_redraw, "No Redraw");

            let update = ui.button("Update").clicked();
            if update {
                state.redraw = true;
            }

            let save = ui.button("Save").clicked();
            if save {
                state
                    .image
                    .save(get_save_path(&app.exe_name().unwrap()))
                    .unwrap();
            }
        });
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    // Let egui handle things like keyboard and mouse input.
    model.egui.handle_raw_event(event);
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    let state = &mut model.state;
    match key {
        Key::Left => {
            let shift_x = get_shift_speed(state.x_range, state.shift_speed);
            state.x_range = shift(state.x_range, -shift_x);
            state.redraw = true;
        }
        Key::Right => {
            let shift_x = get_shift_speed(state.x_range, state.shift_speed);
            state.x_range = shift(state.x_range, shift_x);
            state.redraw = true;
        }
        Key::Up => {
            let shift_y = get_shift_speed(state.y_range, state.shift_speed);
            state.y_range = shift(state.y_range, -shift_y);
            state.redraw = true;
        }
        Key::Down => {
            let shift_y = get_shift_speed(state.y_range, state.shift_speed);
            state.y_range = shift(state.y_range, shift_y);
            state.redraw = true;
        }
        Key::Plus | Key::Equals => {
            let zoom_factor = 1.0 - 10.0 * state.zoom_speed;
            (state.x_range, state.y_range) =
                zoom_relative(state.x_range, state.y_range, zoom_factor, (0.5, 0.5));
            state.redraw = true;
        }
        Key::Minus => {
            let zoom_factor = 1.0 + 10.0 * state.zoom_speed;
            (state.x_range, state.y_range) =
                zoom_relative(state.x_range, state.y_range, zoom_factor, (0.5, 0.5));
            state.redraw = true;
        }
        Key::Q => app.quit(),
        Key::S => model
            .state
            .image
            .save(get_save_path(&app.exe_name().unwrap()))
            .unwrap(),
        Key::Return => model.state.redraw = true,
        _other_key => {}
    }
}

fn mouse_wheel(_app: &App, model: &mut Model, delta: MouseScrollDelta, _phase: TouchPhase) {
    let state = &mut model.state;

    match delta {
        MouseScrollDelta::LineDelta(_, y) => {
            let zoom_factor = 1.0 + y as f64 * state.zoom_speed;
            (state.x_range, state.y_range) = zoom_relative(
                state.x_range,
                state.y_range,
                zoom_factor,
                state.mouse_position,
            );
        }
        MouseScrollDelta::PixelDelta(pos) => {
            let zoom_factor = 1.0 + pos.y * state.zoom_speed;
            (state.x_range, state.y_range) = zoom_relative(
                state.x_range,
                state.y_range,
                zoom_factor,
                state.mouse_position,
            );
        }
    }
    model.state.redraw = true;
}

fn mouse_moved(app: &App, model: &mut Model, pos: Point2) {
    let state = &mut model.state;
    let (_, max_x, _, max_y) = app.window_rect().l_r_b_t();
    let (width, height) = app.window_rect().w_h();
    let x_percent = (pos.x + max_x) / width;
    let y_percent = (pos.y + max_y) / height;
    state.mouse_position = (x_percent as f64, y_percent as f64);
}

fn view(app: &App, model: &Model, frame: Frame) {
    // Setup the drawing context
    let draw = app.draw();
    let state = &model.state;

    // Convert the image to a texture and draw it
    let texture = create_texture(app.main_window(), state.image.clone());
    draw.texture(&texture);

    // Move the drawing to the frame
    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}

fn to_image(array: Vec<Vec<f64>>) -> ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let width = array[0].len() as u32;
    let height = array.len() as u32;

    let mut image: RgbaImage = RgbaImage::new(width, height);
    image
        .enumerate_pixels_mut()
        .par_bridge()
        .for_each(|(x, y, pixel)| {
            let lightness = array[y as usize][x as usize] as u8;
            *pixel = image::Rgba([lightness, lightness, lightness, 255])
        });
    image
}
