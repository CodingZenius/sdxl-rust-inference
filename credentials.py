#!/usr/bin/env python3
"""
Credential manager for Hugging Face and Civitai tokens.
Reads from environment variables only — never stored in git.
Works identically across Kaggle, Colab, and SageMaker.
"""
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


class CredentialManager:
    HF_TOKEN_ENV = "HF_TOKEN"
    CIVITAI_TOKEN_ENV = "CIVITAI_TOKEN"

    @classmethod
    def get_hf_token(cls) -> str:
        token = os.getenv(cls.HF_TOKEN_ENV)
        if not token:
            raise ValueError(
                f"Missing {cls.HF_TOKEN_ENV}. "
                f"Set it in your notebook: os.environ['{cls.HF_TOKEN_ENV}'] = 'your_token'"
            )
        return token

    @classmethod
    def get_civitai_token(cls) -> Optional[str]:
        return os.getenv(cls.CIVITAI_TOKEN_ENV)

    @classmethod
    def validate(cls) -> dict:
        status = {
            "hf_token_present": bool(os.getenv(cls.HF_TOKEN_ENV)),
            "civitai_token_present": bool(os.getenv(cls.CIVITAI_TOKEN_ENV)),
        }
        if not status["hf_token_present"]:
            print("HF_TOKEN not set — Hugging Face downloads will fail.")
        if not status["civitai_token_present"]:
            print("CIVITAI_TOKEN not set — some Civitai models may require auth.")
        return status


class HuggingFaceDownloader:
    """Downloads models from HF Hub using the token from CredentialManager."""

    def __init__(self):
        self.token = CredentialManager.get_hf_token()

    def download_model(self, model_id: str, cache_dir: Path = None) -> Path:
        from huggingface_hub import snapshot_download

        cache_dir = cache_dir or Path("/tmp/sdxl_models")
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {model_id} from HF Hub...")
        model_path = snapshot_download(
            model_id,
            token=self.token,
            cache_dir=str(cache_dir),
            repo_type="model",
        )
        print(f"Downloaded to {model_path}")
        return Path(model_path)


class CivitaiDownloader:
    """Downloads models/LoRAs from Civitai using an optional token."""

    def __init__(self):
        self.token = CredentialManager.get_civitai_token()

    def download_model(self, url: str, cache_dir: Path = None) -> Path:
        import requests

        cache_dir = cache_dir or Path("/tmp/sdxl_models")
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading from Civitai: {url}")

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        filename = None
        disposition = response.headers.get("content-disposition", "")
        if "filename=" in disposition:
            filename = disposition.split("filename=")[-1].strip('"\'')

        if not filename:
            filename = urlparse(url).path.split("/")[-1] or "model.safetensors"

        output_path = cache_dir / filename
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = 100 * downloaded / total_size
                        print(f"  {pct:.1f}% ({downloaded / 1e9:.2f}GB / {total_size / 1e9:.2f}GB)", end="\r")

        print(f"\nDownloaded to {output_path}")
        return output_path


if __name__ == "__main__":
    status = CredentialManager.validate()
    print(f"Credentials status: {status}")
