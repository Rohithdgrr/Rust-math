@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;

const WORKGROUP_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * 256u + gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] + b[idx];
    }
}
