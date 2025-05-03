@group(0) @binding(0)
var tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1)
var<uniform> data: FaradayData;

struct FaradayData {
    max_iter: u32,
    num_particles: u32,
    _padding: vec2<u32>,
    x_range: vec2<f32>,
    y_range: vec2<f32>,
};

// Aliases for types to quickly change the precision of the shader.
alias float = f32;
alias vec2float = vec2<f32>;

// Tolerance for a function.
// The pixel is part of the function if its y coordinate is within this tolerance of the function value.
const fx_tolerance = 0.1;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(tex);

    // Ensure the invocation is within bounds
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Normalize pixel coordinates to [0.0, 1.0] and flip Y-axis
    let uv = vec2float(
        float(gid.x) / float(dims.x),
        1.0 - float(gid.y) / float(dims.y)
    );

    // Linearly interpolate position in x and y ranges
    let x = mix(data.x_range[0], data.x_range[1], uv.x);
    let y = mix(data.y_range[0], data.y_range[1], uv.y);

    // Compute one-pixel sizes in world space:
    let dx = (data.x_range[1] - data.x_range[0]) / float(dims.x);
    let dy = (data.y_range[1] - data.y_range[0]) / float(dims.y);

    // let color = mandelbrot(x, y);
    let color = math_fn(x, y, dx, dy);

    textureStore(tex, vec2<i32>(gid.xy), color);
}

// We’ll sample f(x ± h) to approximate f′(x):
fn f(x: float) -> float {
    return -x * cos(sin(10.0 * x) * x);
}

fn math_fn(x: float, y: float, dx: float, dy: float) -> vec4<f32> {
    // Sample f at center and pixel edges
    let fx = f(x);
    let fL = f(x - 0.5 * dx);
    let fR = f(x + 0.5 * dx);

    // Approximate derivative
    let fpx   = (fR - fL) / dx;

    // Compute perpendicular distance 'd' from pixel center to curve,
    // normalized by pixel height:
    let d = abs(y - fx) / sqrt(1.0 + fpx * fpx);

    // True half-pixel distance to the curve in y-direction:
    let half_pixel = 0.5 * dy;
    let inside = d <= half_pixel;

    // Sign‐crossing test (catches vertical passes)
    // cross = (y - fL) and (y - fR) have opposite signs
    let left_above = (y - fL) > 0.0;
    let right_above = (y - fR) > 0.0;
    let cross_ok = left_above != right_above;

    // Smooth antialias alpha
    // Antialiased alpha: 1.0 at center, 0.0 beyond half-pixel:
    let alpha = clamp(1.0 - d / (0.5 * dy), 0.0, 1.0);
    let raw_alpha = clamp((half_pixel - d) / half_pixel, 0.0, 1.0);

    // Combine hard mask + smooth edge:
    let hard_mask = inside || cross_ok;
    let final_alpha = select(raw_alpha, 1.0, hard_mask);

    // Blend white->black by final_alpha:
    let bg = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let fg = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return mix(bg, fg, final_alpha);
}

fn math_fn_original(x: float, y: float, dx: float, dy: float) -> vec4<f32> {
    // Evaluate f and approximate its derivative:
    let fx = f(x);
    let fpx = (f(x + 0.5 * dx) - f(x - 0.5 * dx)) / (dx);
    let fpx2 = fpx * fpx;

    // True half-pixel distance to the curve in y-direction:
    let half_dist = 0.5 * dy * sqrt(1.0 + fpx2);
    let inside = abs(y - fx) <= half_dist;

    // Smooth antialias alpha
    // Compute perpendicular distance 'd' from pixel center to curve,
    // normalized by pixel height:
    let d = abs(y - fx) / sqrt(1.0 + fpx2);

    // Antialiased alpha: 1.0 at center, 0.0 beyond half-pixel:
    let alpha = clamp(1.0 - d / (0.5 * dy), 0.0, 1.0);

    // Blend between white background and black curve:
    let bg = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let fg = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    // Choose either full fg (inside) or bg (outside)
    let hard = select(bg, fg, inside);
    // Mix the two colors based on alpha
    return mix(hard, fg, alpha);
}


fn mandelbrot(real_initial: float, imag_initial: float) -> vec4<f32> {
    // Initialize mandelbrot at z = 0
    var real = 0.0;
    var imag = 0.0;
    var iter = 0u;

    loop {
        if iter >= data.max_iter {
            break;
        }

        let r2 = real * real;
        let i2 = imag * imag;

        // Check for divergence
        if r2 + i2 > 4.0 {
            break;
        }

        // Compute next iteration
        let new_real = r2 - i2 + real_initial;
        let new_imag = 2.0 * real * imag + imag_initial;

        // Update the real and imaginary parts
        real = new_real;
        imag = new_imag;
        iter = iter + 1u;
    }

    // Color based on iteration count
    var shade: f32;
    if iter == data.max_iter {
        shade = 0.0;
    } else {
        // Normalize the iteration count to [0.0, 1.0]
        shade =  f32(iter) / f32(data.max_iter);
    }

    return vec4<f32>(shade, shade, shade, 1.0);
}
