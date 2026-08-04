@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;
@group(0) @binding(3) var<storage, read> metadata: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let op = u32(metadata[2]);
        if (op == 0u) {
            // Add
            out[idx] = a[idx] + b[idx];
        } else if (op == 1u) {
            // Sub
            out[idx] = a[idx] - b[idx];
        } else if (op == 2u) {
            // Mul
            out[idx] = a[idx] * b[idx];
        }
    }
}
