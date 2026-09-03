use anyhow::Result;
use candle_core::Device;
use image::RgbImage;
use std::path::PathBuf;

use crate::inference::scheduler::Scheduler;
use crate::models::SdxlModel;

/// Orchestrates a full generation: model lives on `model_gpu`,
/// the sampling loop and scheduler run on `inference_gpu`.
pub struct SdxlPipeline {
    model: SdxlModel,
    scheduler_device: Device,
}

impl SdxlPipeline {
    pub fn new(
        model_id: &str,
        lora_ids: &[String],
        model_gpu: Device,
        inference_gpu: Device,
    ) -> Result<Self> {
        let model_path = PathBuf::from(format!("/tmp/sdxl_models/{model_id}"));
        let mut model = SdxlModel::load(&model_path, model_gpu)?;

        for lora_id in lora_ids {
            let lora_path = PathBuf::from(format!("/tmp/sdxl_models/{lora_id}"));
            model.load_lora(&lora_path)?;
        }

        Ok(Self {
            model,
            scheduler_device: inference_gpu,
        })
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        negative_prompt: &str,
        cfg_scale: f32,
        num_steps: usize,
        denoise: f32,
    ) -> Result<RgbImage> {
        tracing::info!(
            "Generating: prompt='{}' cfg={} steps={} denoise={}",
            prompt,
            cfg_scale,
            num_steps,
            denoise
        );

        let _scheduler = Scheduler::new(num_steps, self.scheduler_device.clone());
        let _prompt_embeds = self.model.encode_prompt(prompt)?;
        let _negative_embeds = self.model.encode_prompt(negative_prompt)?;

        // TODO: run the denoising loop on `scheduler_device`, pulling
        // UNet forward passes from `self.model` (on model_gpu), then
        // decode the final latents through the VAE and return the image.
        todo!("Wire up the full denoising loop and VAE decode")
    }
}
