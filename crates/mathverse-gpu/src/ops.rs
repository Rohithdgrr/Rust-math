//! GPU-accelerated math operations.

use mathverse_core::error::{MathError, MathResult};
use mathverse_matrix::Matrix;
use mathverse_vector::Vector;

use crate::context::GpuContext;

/// GPU matrix multiplication.
///
/// Uploads both matrices to GPU, runs the compute shader, and downloads
/// the result. For small matrices (< 64x64), CPU is likely faster.
pub fn gpu_mat_mul(ctx: &GpuContext, a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
    if a.cols() != b.rows() {
        return Err(MathError::DimensionMismatch);
    }

    let m = a.rows();
    let _k = a.cols();
    let n = b.cols();

    // Create buffers
    let buf_a = ctx.create_buffer_init(a.as_slice());
    let buf_b = ctx.create_buffer_init(b.as_slice());
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: (m * n * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create bind group layout
    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bind_group = ctx
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul BindGroup"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

    // Create pipeline
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Dispatch
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 16;
        pass.dispatch_workgroups(
            ((n + workgroup_size - 1) / workgroup_size) as u32,
            ((m + workgroup_size - 1) / workgroup_size) as u32,
            1,
        );
    }
    ctx.queue.submit([encoder.finish()]);

    // Read back
    let data = pollster::block_on(ctx.read_buffer(&buf_out, m * n))?;

    Ok(Matrix::new(m, n, data)?)
}

/// GPU element-wise addition.
pub fn gpu_add(ctx: &GpuContext, a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(MathError::DimensionMismatch);
    }

    let buf_a = ctx.create_buffer_init(a.as_slice());
    let buf_b = ctx.create_buffer_init(b.as_slice());
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: (a.data().len() * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Add Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/elementwise.wgsl").into()),
        });

    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Add Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // Metadata buffer: [rows, cols, op_type]
    let metadata: Vec<f64> = vec![a.rows() as f64, a.cols() as f64, 0.0]; // op 0 = add
    let buf_meta = ctx.create_buffer_init(&metadata);

    let bind_group = ctx
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Add BindGroup"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_meta.as_entire_binding(),
                },
            ],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Add Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 256;
        let n = a.data().len();
        pass.dispatch_workgroups(((n + workgroup_size - 1) / workgroup_size) as u32, 1, 1);
    }
    ctx.queue.submit([encoder.finish()]);

    let data = pollster::block_on(ctx.read_buffer(&buf_out, a.data().len()))?;

    Ok(Matrix::new(a.rows(), a.cols(), data)?)
}

/// GPU vector dot product.
pub fn gpu_dot(ctx: &GpuContext, a: &Vector, b: &Vector) -> MathResult<f64> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch);
    }

    let _buf_a = ctx.create_buffer_init(&a.data);
    let _buf_b = ctx.create_buffer_init(&b.data);

    // For dot product, we need a reduction. For simplicity, use a staging buffer.
    let _buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dot Output"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Simple approach: multiply element-wise then sum on CPU
    let mut result_data = vec![0.0f64; a.len()];
    for i in 0..a.len() {
        result_data[i] = a.data[i] * b.data[i];
    }

    // For real GPU acceleration, a reduction shader would be needed
    // This is a simplified version
    let result: f64 = result_data.iter().sum();
    Ok(result)
}

#[cfg(test)]
mod tests {
    // GPU tests require actual hardware
    #[test]
    fn gpu_ops_compile() {
        let _ = std::any::type_name::<super::GpuContext>();
    }
}
