use anyhow::Result;
use candle_core::Device;
use sdxl_rust_inference::inference::SdxlPipeline;
use std::io::{self, Read};

#[derive(serde::Deserialize)]
struct GenerationConfig {
    model_id: String,
    #[serde(default)]
    lora_ids: Vec<String>,
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_cfg")]
    cfg_scale: f32,
    #[serde(default = "default_steps")]
    num_steps: usize,
    #[serde(default = "default_denoise")]
    denoise: f32,
}

fn default_cfg() -> f32 {
    7.5
}
fn default_steps() -> usize {
    30
}
fn default_denoise() -> f32 {
    1.0
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Initialize dual GPU split
    // GPU 0: holds base model + LoRA weights + VAE (~11GB + 150-300MB + 20-150MB)
    // GPU 1: runs the diffusion loop / scheduler steps, keeps headroom free
    let device_model = Device::new_cuda(0)?;
    let device_inference = Device::new_cuda(1)?;

    tracing::info!("GPU 0 (model store) and GPU 1 (inference loop) initialized");

    // Config is piped in via stdin as JSON by app.py
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let config: GenerationConfig = serde_json::from_str(&input)?;

    let mut pipeline = SdxlPipeline::new(
        &config.model_id,
        &config.lora_ids,
        device_model.clone(),
        device_inference.clone(),
    )?;

    let image = pipeline.generate(
        &config.prompt,
        &config.negative_prompt,
        config.cfg_scale,
        config.num_steps,
        config.denoise,
    )?;

    image.save("output.png")?;
    println!("{{\"status\": \"success\", \"output_path\": \"output.png\"}}");

    Ok(())
}
