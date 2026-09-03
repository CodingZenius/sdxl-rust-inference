#!/usr/bin/env python3
"""
Model registry UI for the Kaggle notebook.
Lets you paste HF or Civitai links and preload models/LoRAs
before running the inference cell.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path

from credentials import HuggingFaceDownloader, CivitaiDownloader

MODELS_DIR = Path("/tmp/sdxl_models")


class ModelRegistry:
    def __init__(self):
        self.hf_link = widgets.Text(
            placeholder="https://huggingface.co/org/model",
            description="HF Link:",
            layout=widgets.Layout(width="500px"),
        )
        self.civitai_link = widgets.Text(
            placeholder="https://civitai.com/api/download/models/12345",
            description="Civitai Link:",
            layout=widgets.Layout(width="500px"),
        )
        self.lora_links = widgets.Textarea(
            placeholder="One Civitai or HF LoRA link per line",
            description="LoRAs:",
            layout=widgets.Layout(width="500px", height="80px"),
        )
        self.load_button = widgets.Button(description="Load Models", button_style="success")
        self.status = widgets.Output()

        self.load_button.on_click(self._on_load_click)

    def _on_load_click(self, _button):
        with self.status:
            clear_output()
            print("Loading models...")

            if self.hf_link.value.strip():
                self._download_hf(self.hf_link.value.strip())

            if self.civitai_link.value.strip():
                self._download_civitai(self.civitai_link.value.strip())

            for link in self.lora_links.value.splitlines():
                link = link.strip()
                if not link:
                    continue
                if "huggingface.co" in link:
                    self._download_hf(link)
                elif "civitai.com" in link:
                    self._download_civitai(link)

            print("Models loaded and cached.")

    def _download_hf(self, url_or_id: str):
        model_id = url_or_id.split("huggingface.co/")[-1].strip("/")
        HuggingFaceDownloader().download_model(model_id, cache_dir=MODELS_DIR)

    def _download_civitai(self, url: str):
        CivitaiDownloader().download_model(url, cache_dir=MODELS_DIR)

    def show(self):
        display(
            widgets.VBox(
                [
                    self.hf_link,
                    self.civitai_link,
                    self.lora_links,
                    self.load_button,
                    self.status,
                ]
            )
        )


if __name__ == "__main__":
    registry = ModelRegistry()
    registry.show()
