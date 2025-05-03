// Bind group 0, binding 0: the storage texture we’ll write into.
@group(0) @binding(0)
var output_tex: texture_storage_2d<rgba16float, read_write>;


struct FaradayData {
    max_iter: u32,
    num_particles: u32,
    _pad0: u32,
    _pad1: u32,
    x_range: vec2<f32>,
    y_range: vec2<f32>,
};

@group(0) @binding(1)
var<uniform> faraday: FaradayData;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Get the texture dimensions.
    let dims = textureDimensions(output_tex);
    // Make sure we’re in‐bounds.
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    // Compute normalized [0..1] coordinate across X.
    let uv = vec2<f32>(gid.xy) / vec2<f32>(dims);
    // Invert so that x=0 → 1.0 (white), x=1 → 0.0 (black).
    let v = 1.0 - uv.x;
    // Store as RGBA.
    textureStore(
        output_tex,
        vec2<i32>(gid.xy),
        vec4<f32>(v, v, v, 1.0)
    );
}

