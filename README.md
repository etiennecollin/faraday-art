# Faraday Art

## Enabling f64 Precision

> [!WARNING]
> Enabling `f64` precision might increase the precision of the rendered images
> and allow to zoom further. However, this comes with a potentially huge
> performance hit.

To enable `f64` precision (as opposed to the default `f32` precision), do the
following changes:

In `./src/lib.rs`, change the `f32` aliase to `f64`:

```diff
- alias FloatChoice = f32;
+ alias FloatChoice = f64;
```

In `./src/utils/shaders/compute.wgsl`, change the `f32` aliases to `f64`:

```diff
- alias float = f32;
- alias vec2float = vec2<f32>;
+ alias float = f64;
+ alias vec2float = vec2<f64>;
```

In `./src/main.rs`, uncomment the GPU feature to enable `f64` shaders:

```diff
fn model(app: &App) -> Model {
    // Set GPU device descriptor
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("Point Cloud Renderer Device"),
        features: wgpu::Features::default()
-           // | wgpu::Features::SHADER_F64 // To support f64 in shaders
+           | wgpu::Features::SHADER_F64 // To support f64 in shaders
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        limits: wgpu::Limits {
            // max_texture_dimension_2d: 2 << 14, // To support the big 9x3 4K display wall
            ..Default::default()
        },
    };
```

> [!WARNING]
> Uncommenting this last line might cause the code to crash on start if the
> GPU on which the code runs does not support `f64`.
