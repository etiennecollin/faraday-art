// Aliases for types to quickly change the precision of the shader.
alias float = f32;
alias vec2float = vec2<f32>;

struct FaradayData {
    max_iter: u32,
    num_particles: u32,
    _padding: vec2<u32>,
    dt: float,
    mu: float,
    x_range: vec2float,
    y_range: vec2float,
};

@group(0) @binding(0)
var tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1)
var<uniform> fdata: FaradayData;

@compute @workgroup_size(16, 16)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Ensure the invocation is within bounds
    let dims = textureDimensions(tex);
    if (any(gid.xy >= dims)) { return; }

    // Sample at pixel centers
    let dims_f = vec2float(dims);
    var uv = (vec2float(gid.xy) + vec2float(0.5, 0.5)) / dims_f;
    uv.y = float(1.0) - uv.y; // Flip Y

    // Get x/y in "math" space
    let x = mix(fdata.x_range[0], fdata.x_range[1], uv.x);
    let y = mix(fdata.y_range[0], fdata.y_range[1], uv.y);

    // Compute one-pixel sizes in world space:
    let dx = (fdata.x_range[1] - fdata.x_range[0]) / float(dims.x);
    let dy = (fdata.y_range[1] - fdata.y_range[0]) / float(dims.y);

    let color = mandelbrot(vec2float(x, y));
    // let color = math_fn(x, y, dx, dy, 3.0);
    // let color = van_der_pol(vec2<f32>(x, y));

    textureStore(tex, vec2<u32>(gid.xy), color);
}

fn mandelbrot(z_initial: vec2float) -> vec4<f32> {
    // Initialize mandelbrot at z = 0
    var z = vec2float(float(0.0), float(0.0));
    var iter = 0u;

    loop {
        if iter >= fdata.max_iter {
            break;
        }

        let z2 = z * z;

        // Check for divergence
        // We assume divergence if the modulus of z is greater than 4.0
        if z2[0] + z2[1] > float(4.0) {
            break;
        }

        // Compute next iteration
        z = vec2float(z2[0] - z2[1] + z_initial[0], float(2.0) * z[0] * z[1] + z_initial[1]);
        iter = iter + 1u;
    }

    // Color (BW) based on iteration count
    // var shade: f32;
    // if iter == fdata.max_iter {
    //     shade = 0.0;
    // } else {
    //     // Normalize the iteration count to [0.0, 1.0]
    //     shade = f32(iter) / f32(fdata.max_iter);
    // }
    // return vec4<f32>(shade, shade, shade, 1.0);


    // Color (RGB) based on iteration count
    let h = f32(iter) / f32(fdata.max_iter);
    let s = 1.0;
    var v: f32;
    if iter == fdata.max_iter {
        v = 0.0;
    } else {
        // Normalize the iteration count to [0.0, 1.0]
        v = 1.0;
    }
    return vec4<f32>(hsv2rgb(h, s, v), 1.0);
}

// We’ll sample f(x ± h) to approximate f′(x):
fn f(x: float) -> float {
    return -x * cos(exp(sin(float(10.0) * x)) * x);
}

fn math_fn(x: float, y: float, dx: float, dy: float, thickness: float) -> vec4<f32> {
    // Half‑pixel radius in world‑space
    // Scale that by thickness
    let half_px_x = float(0.5) * dx * thickness;
    let half_px_y = float(0.5) * dy * thickness;

    // Sample f at pixel left/right for slope and vertical‐crossing check
    let f_center = f(x);
    let f_left = f(x - half_px_x);
    let f_right = f(x + half_px_x);

    // Approximate slope and get inverse normal length
    let slope = (f_right - f_left) / (float(2.0) * half_px_x);
    let inv_len = float(1.0) / sqrt(float(1.0) + slope * slope);

    // Perpendicular distance from pixel center to curve (along the normal)
    let perp_dist = abs(y - f_center) * inv_len;

    // Vertical sign‑crossing: does the curve cross this pixel column?
    // i.e. function values at left/right straddle the pixel’s y
    let crosses_left = (y - f_left) > float(0.0);
    let crosses_right = (y - f_right) > float(0.0);
    let vert_cross = crosses_left != crosses_right;

    // Horizontal sign‑crossing: does the curve cross this pixel row?
    // i.e. the curve at this x goes above the top edge or below the bottom edge
    let crosses_bottom = (f_center - (y - half_px_y)) > float(0.0);
    let crosses_top = (f_center - (y + half_px_y)) > float(0.0);
    let horiz_cross = crosses_bottom != crosses_top;

    // Smooth alpha fall‑off from center to edges of the thick band
    let raw_alpha = clamp(f32((half_px_y - perp_dist) / half_px_y), 0.0, 1.0);

    // If either crossing test fired, force full coverage (alpha = 1)
    let any_cross = vert_cross || horiz_cross;
    let final_alpha = select(raw_alpha, 1.0, any_cross);

    // Blend from white (background) to black (curve)
    let bg = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let fg = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return mix(bg, fg, final_alpha);
}

fn step_vdp(z: vec2float) -> vec2float {
    // z.x = x, z.y = y
    let x = z.x;
    let y = z.y;

    // dx/dt = y
    // dy/dt = mu * (1 - x^2) * y - x
    return vec2float(
        y,
        fdata.mu * (float(1.0) - x * x) * y - x
    );
}

fn van_der_pol(initial: vec2float) -> vec4<f32> {
    var z = initial;
    var iter = 0u;
    while (iter < fdata.max_iter) {
        // Single Euler step (feel free to swap in RK4 for more accuracy)
        let dz = step_vdp(z);
        z = z + fdata.dt * dz;

        // divergence test
        if (dot(z, z) > float(200.0)) {
            break;
        }

        iter = iter + 1u;
    }

    // map iteration to grayscale
    var shade: f32;
    if (iter == fdata.max_iter) {
        // never diverged → black
        shade = 0.0;
    } else {
        shade = f32(iter) / f32(fdata.max_iter);
    }
    return vec4<f32>(shade, shade, shade, 1.0);
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let hp = fract(h) * 6.0;
    let x = c * (1.0 - abs(fract(hp) * 2.0 - 1.0));
    var rgb = vec3f(0.0);
    if hp < 1.0 {
        rgb = vec3f(c, x, 0.0);
    } else if hp < 2.0 {
        rgb = vec3f(x, c, 0.0);
    } else if hp < 3.0 {
        rgb = vec3f(0.0, c, x);
    } else if hp < 4.0 {
        rgb = vec3f(0.0, x, c);
    } else if hp < 5.0 {
        rgb = vec3f(x, 0.0, c);
    } else {
        rgb = vec3f(c, 0.0, x);
    }
    let m = v - c;
    return rgb + vec3<f32>(m);
}
