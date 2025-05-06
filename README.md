# Faraday Art

<!-- vim-markdown-toc GFM -->

- [Dependencies](#dependencies)
- [Running](#running)
  - [Enabling f64 Precision](#enabling-f64-precision)

<!-- vim-markdown-toc -->

## Dependencies

- `rust >= 1.86.0`

## Running

To execute the code:

```bash
cargo run -r
```

### Enabling f64 Precision

> [!WARNING]
> Enabling `f64` precision will increase the precision of the rendered images
> and allow to zoom further. However, this comes with a potentially huge
> performance hit.

> [!WARNING]
> Enabling `f64` precision on GPUs which do not support `f64` WGSL shaders will
> cause the code to crash on start.

To enable `f64` precision (as opposed to the default `f32` precision), modify
the following lines in `./src/utils/shaders/compute.wgsl`:

```diff
- alias float = f32;
- alias vec2float = vec2<f32>;
+ alias float = f64;
+ alias vec2float = vec2<f64>;
```

And execute the code using the `f64` feature:

```bash
cargo run -r --features f64
```
