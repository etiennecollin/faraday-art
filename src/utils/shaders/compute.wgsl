// Bind group 0, binding 0: the storage texture we’ll write into.
@group(0) @binding(0)
var tex: texture_storage_2d<rgba32float, read_write>;

struct FaradayData {
    max_iter: u32,
    num_particles: u32,
    _padding: vec2<u32>,
    x_range: vec2<f32>,
    y_range: vec2<f32>,
};

@group(0) @binding(1)
var<uniform> data: FaradayData;

alias float = f32;
alias vec2float = vec2<f32>;

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

    let color = mandelbrot(x, y);

    textureStore(tex, vec2<i32>(gid.xy), color);
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
