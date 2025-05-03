// Bind group 0, binding 0: the texture we’ll sample (no sampler needed for textureLoad).
@group(0) @binding(0)
var src_tex: texture_2d<f32>;

// A single full‑screen triangle (no vertex buffer).
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos: vec2<f32>;

    // Those three points form one big triangle whose corners lie well outside
    // the normalized device coordinate (NDC) square. When rasterized, that
    // single triangle overlaps every pixel in the frame. So you get exactly
    // one fragment call per pixel.
    switch vi {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { pos = vec2<f32>(3.0, -1.0); }
        case 2u: { pos = vec2<f32>(-1.0, 3.0); }
        default: { pos = vec2<f32>(0.0, 0.0); } // Fallback, shouldn't happen
    }
    return vec4<f32>(pos, 0.0, 1.0);
}


@fragment
fn fs_main(
    @builtin(position) pos: vec4<f32>
) -> @location(0) vec4<f32> {
    // pos.xy is in window pixels; cast to ivec2 for textureLoad.
    let coord = vec2<i32>(pos.xy);
    // Fetch the generated gradient texel.
    return textureLoad(src_tex, coord, 0);
}

