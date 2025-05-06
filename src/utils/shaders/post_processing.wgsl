struct GlobalData {
    value_min: atomic<u32>, // Bitcast from f32
    value_max: atomic<u32>, // Bitcast from f32
    histogram_n: atomic<u32>,
    histogram: array<atomic<u32>, 256>, // Bitcast from f32
    cdf_threshold: f32,
    cdf_non_zero: f32,
    cdf: array<f32, 256>,
};

@group(0) @binding(0)
var tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2)
var<storage, read_write> gdata: GlobalData;

const WG_SIZE = 256u; // Workgroup size
var<workgroup> local_mins: array<f32, WG_SIZE>;
var<workgroup> local_maxs: array<f32, WG_SIZE>;
@compute @workgroup_size(16, 16)
fn cs_min_max(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    // Ensure the invocation is within bounds
    let dims = textureDimensions(tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Compute a scalar luminance/brightness from RGB
    let color = textureLoad(tex, vec2<u32>(gid.xy));
    let lum = get_luminance(color);

    // Write local
    local_mins[lidx] = lum;
    local_maxs[lidx] = lum;

    // Wait for all workgroup threads to write their local min/max
    workgroupBarrier();

    // Tree‐based reduction in shared memory
    var stride = WG_SIZE >> 1u;
    while (stride > 0u) {
        if (lidx < stride) {
            let a = local_mins[lidx];
            let b = local_mins[lidx + stride];
            local_mins[lidx] = select(a, b, b < a);

            let c = local_maxs[lidx];
            let d = local_maxs[lidx + stride];
            local_maxs[lidx] = select(c, d, d > c);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Wait for the workgroup to finish writing
    workgroupBarrier();

    // Workgroup‐leader atomically merges into the global
    if (lidx == 0u) {
        atomicMin(&gdata.value_min, bitcast<u32>(local_mins[0]));
        atomicMax(&gdata.value_max, bitcast<u32>(local_maxs[0]));
    }
}

@compute @workgroup_size(16,16)
fn cs_recalibrate(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Ensure the invocation is within bounds
    let dims = textureDimensions(tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let color = textureLoad(tex, vec2<u32>(gid.xy));

    // Read global min/max
    let min_val = bitcast<f32>(atomicLoad(&gdata.value_min));
    let max_val = bitcast<f32>(atomicLoad(&gdata.value_max));

    // Compute the range and its reciprocal (guarding against zero)
    let range = max_val - min_val;
    let inv_range = select(0.0, 1.0 / range, range > 0.0);

    // Remap RGB channels and clamp into [0,1]
    let rgb = (color.rgb - vec3<f32>(min_val)) * inv_range;
    let normalized = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    // Store the recalibrated color
    textureStore(tex, vec2<u32>(gid.xy),  vec4<f32>(normalized, color.a));
}

@compute @workgroup_size(16, 16)
fn cs_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Ensure the invocation is within bounds
    let dims = textureDimensions(tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let color = textureLoad(tex, vec2<u32>(gid.xy));
    let bin = u32(get_luminance(color) * 255.0);

    // Compute a scalar luminance/brightness from RGB
    let v = get_luminance(color);

    if v > gdata.cdf_threshold {
        atomicAdd(&gdata.histogram[bin], 1u);
        atomicAdd(&gdata.histogram_n, 1u);
    }
}

@compute @workgroup_size(1)
fn cs_cdf(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Compute the CDF
    let n = atomicLoad(&gdata.histogram_n);
    let n_inverse = select(0.0, 1.0 / f32(n), n > 0u);

    var cumulative = 0.0;
    var first_non_zero = -1.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let h = f32(atomicLoad(&gdata.histogram[i]));
        cumulative = cumulative + h * n_inverse;
        gdata.cdf[i] = cumulative;

        // Capture the first non-zero cumulative value
        if (first_non_zero < 0.0 && cumulative > 0.0) {
            first_non_zero = cumulative;
        }
    }
    gdata.cdf_non_zero = first_non_zero;
}

@compute @workgroup_size(16, 16)
fn cs_equalize(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Ensure the invocation is within bounds
    let dims = textureDimensions(tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let color = textureLoad(tex, vec2<u32>(gid.xy));

    // Compute bin index
    let lum = get_luminance(color);
    let bin = u32(clamp(lum * 255.0, 0.0, 255.0));

    // Read the precomputed CDF
    let mapped = gdata.cdf[bin];
    let cdfv = gdata.cdf[bin]; // in [0,1]
    let cdf_min = gdata.cdf_threshold; // cdf_min > 0
    let denom = max(1.0 - cdf_min, 1e-6); // avoid div0
    let equalized = clamp((cdfv - cdf_min) / denom, 0.0, 1.0);

    // Rescale RGB
    let scale = mapped / max(lum, 1e-6);
    let rgb = clamp(color.rgb * scale, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(tex, vec2<i32>(gid.xy), vec4<f32>(rgb, color.a));
}

fn get_luminance(color: vec4<f32>) -> f32 {
    // Compute a scalar luminance/brightness from RGB
    // return max(max(color.r, color.g), color.b);
    // Alternatively, for perceptual luminance:
    return dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
}
