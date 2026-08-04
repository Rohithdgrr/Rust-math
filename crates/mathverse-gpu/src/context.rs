//! GPU context and device management.

use mathverse_core::error::{MathError, MathResult};

/// GPU context holding the wgpu device and queue.
pub struct GpuContext {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The command queue.
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a new GPU context with the default adapter.
    ///
    /// Returns an error if no GPU adapter is available.
    pub async fn new() -> MathResult<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(MathError::NotImplemented)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("MathVerse GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|_| MathError::Io)?;

        Ok(Self { device, queue })
    }

    /// Get the device info name.
    pub fn device_name(&self) -> String {
        "GPU".to_string()
    }

    /// Create a buffer from a f64 slice.
    pub fn create_buffer_init(&self, data: &[f64]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MathVerse Buffer"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Read back a buffer to a Vec<f64>.
    pub async fn read_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> MathResult<Vec<f64>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (len * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (len * 8) as u64);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| MathError::Io)?
            .map_err(|_| MathError::Io)?;

        let data = slice.get_mapped_range();
        let floats: Vec<f64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(floats)
    }
}

#[cfg(test)]
mod tests {
    // GPU tests require actual hardware, so we only test compilation
    #[test]
    fn gpu_context_compiles() {
        // Just verify the types exist and are well-formed
        let _ = std::any::type_name::<super::GpuContext>();
    }
}
