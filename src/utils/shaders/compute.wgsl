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

// @compute @workgroup_size(8, 8)
// fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
//     // Get the texture dimensions.
//     let dims = textureDimensions(tex);
//
//     // Make sure we’re in‐bounds.
//     if (gid.x >= dims.x || gid.y >= dims.y) {
//         return;
//     }
//
//     // Normalized UV coordinates [0, 1]
//     let uv = vec2<f32>(gid.xy) / vec2<f32>(dims);
//
//     // Assign colors based on UV:
//     // Red increases from bottom to top (y)
//     // Green increases from left to right (x)
//     // Blue increases from top-left to bottom-right (mix of x and y)
//     let color = vec4<f32>(
//         uv.y,       // red in bottom-left to top-left
//         uv.x,       // green left to right
//         1.0 - uv.x * uv.y, // blue fades diagonally
//         1.0         // full alpha
//     );
//
//     textureStore(tex, vec2<i32>(gid.xy), color);
// }

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(tex);

    // Ensure the invocation is within bounds
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Normalize pixel coordinates to [0.0, 1.0]
    let uv = vec2<f32>(gid.xy) / vec2<f32>(dims);

    // Linearly interpolate position in x and y ranges
    let real_initial = mix(data.x_range[0], data.x_range.y, uv.x);
    let imag_initial = mix(data.y_range[0], data.y_range.y, uv.y);

    // Initialize z = 0
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

    let color = vec4<f32>(shade, shade, shade, 1.0);
    textureStore(tex, vec2<i32>(gid.xy), color);
}

