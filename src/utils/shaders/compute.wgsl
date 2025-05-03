// Bind group 0, binding 0: the storage texture we’ll write into.
@group(0) @binding(0)
var tex: texture_storage_2d<rgba16float, read_write>;


struct FaradayData {
    max_iter: u32,
    num_particles: u32,
    _padding: vec2<u32>,
    x_range: vec2<f32>,
    y_range: vec2<f32>,
};

@group(0) @binding(1)
var<uniform> data: FaradayData;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Get the texture dimensions.
    let dims = textureDimensions(tex);

    // Make sure we’re in‐bounds.
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Normalized UV coordinates [0, 1]
    let uv = vec2<f32>(gid.xy) / vec2<f32>(dims);

    // Assign colors based on UV:
    // Red increases from bottom to top (y)
    // Green increases from left to right (x)
    // Blue increases from top-left to bottom-right (mix of x and y)
    let color = vec4<f32>(
        uv.y,       // red in bottom-left to top-left
        uv.x,       // green left to right
        1.0 - uv.x * uv.y, // blue fades diagonally
        1.0         // full alpha
    );

    textureStore(tex, vec2<i32>(gid.xy), color);
}

