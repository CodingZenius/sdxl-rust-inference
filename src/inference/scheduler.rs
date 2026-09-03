use anyhow::Result;
use candle_core::{Device, Tensor};

/// Standard SDXL scheduler (Euler / DPM++ etc). Owns the noise
/// schedule and runs on the inference GPU alongside the sampling loop.
pub struct Scheduler {
    pub num_steps: usize,
    pub device: Device,
}

impl Scheduler {
    pub fn new(num_steps: usize, device: Device) -> Self {
        Self { num_steps, device }
    }

    pub fn step(&self, _latents: &Tensor, _noise_pred: &Tensor, _timestep: usize) -> Result<Tensor> {
        todo!("Implement scheduler step (Euler / DPM++ 2M etc.)")
    }
}
