#!/usr/bin/env python3
"""
Headless FastAPI wrapper for the SDXL Rust inference binary.
Called by the Kaggle notebook's inference UI cell.
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess
import json
from pathlib import Path
from typing import List, Optional

app = FastAPI()

RUST_BINARY = "./target/release/sdxl-rust-inference"
MODELS_DIR = Path("/tmp/sdxl_models")
OUTPUT_DIR = Path("/tmp/sdxl_output")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model_id: str = "jibjib-realistic"
    lora_ids: Optional[List[str]] = None
    cfg_scale: float = 7.5
    num_steps: int = 30
    denoise: float = 1.0


@app.post("/generate")
async def generate_image(req: GenerateRequest):
    """Generate an image by piping config JSON into the Rust binary."""
    config = {
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "model_id": req.model_id,
        "lora_ids": req.lora_ids or [],
        "cfg_scale": req.cfg_scale,
        "num_steps": req.num_steps,
        "denoise": req.denoise,
    }

    try:
        result = subprocess.run(
            [RUST_BINARY],
            input=json.dumps(config).encode(),
            capture_output=True,
            timeout=300,
            cwd="/workspace",
        )

        if result.returncode != 0:
            return {"error": result.stderr.decode()}

        return {
            "status": "success",
            "output_path": str(OUTPUT_DIR / "output.png"),
            "config": config,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Generation timeout (exceeded 5 minutes)"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/output")
async def get_output():
    """Fetch the last generated image for viewing/downloading."""
    output_path = OUTPUT_DIR / "output.png"
    if output_path.exists():
        return FileResponse(output_path)
    return {"error": "No output available"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5000)
