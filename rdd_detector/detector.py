"""Inference-only refactoring of jaewon1007's LG_Detection notebook.

This is an independent experimental implementation, not the authors' official code.
It preserves posterior sampling, independent noise draws and the notebook's score
convention. See docs/method.md for differences from a full paper reproduction.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import lpips

from .scores import image_score, combine_scores, classify


@dataclass(frozen=True)
class DetectionResult:
    image_score: float
    latent_score: float
    rdd_score: float
    threshold: float | None
    label: str | None

    def to_dict(self):
        return asdict(self)


class RDDDetector:
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", device=None,
                 size=512, noise_samples=16, noise_batch_size=1,
                 t_ratio=0.05, rotation_weight=0.5, seed=42):
        if size < 32 or size % 8:
            raise ValueError("size must be a multiple of 8 and at least 32")
        if noise_samples < 1 or noise_batch_size < 1:
            raise ValueError("noise sample and batch counts must be positive")
        if not 0 < t_ratio < 1:
            raise ValueError("t_ratio must be between 0 and 1")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.noise_samples, self.noise_batch_size = noise_samples, noise_batch_size
        self.t_ratio, self.rotation_weight, self.seed = t_ratio, rotation_weight, seed
        self.preprocess = T.Compose([T.Resize((size, size)), T.ToTensor(),
                                    T.Normalize([0.5] * 3, [0.5] * 3)])
        self.blur = T.GaussianBlur(kernel_size=3, sigma=0.8)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device).eval()
        self.metric = lpips.LPIPS(net="vgg").to(self.device).eval()
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        if self.scheduler.config.prediction_type != "epsilon":
            raise ValueError("This notebook implementation requires an epsilon-prediction model")
        for module in (self.vae, self.unet, self.metric, self.text_encoder):
            module.requires_grad_(False)
        with torch.inference_mode():
            tokens = self.tokenizer([""], padding="max_length",
                                    max_length=self.tokenizer.model_max_length, return_tensors="pt")
            self.null_prompt = self.text_encoder(tokens.input_ids.to(self.device))[0]

    def _ae_error(self, x, generator):
        z = self.vae.encode(x).latent_dist.sample(generator=generator)
        reconstructed = self.vae.decode(z).sample
        return self.metric(x, reconstructed).mean().item()

    def _latent_error(self, z, generator):
        t = int(self.scheduler.config.num_train_timesteps * self.t_ratio)
        total = 0.0
        for start in range(0, self.noise_samples, self.noise_batch_size):
            count = min(self.noise_batch_size, self.noise_samples - start)
            batch = z.repeat(count, 1, 1, 1)
            noise = torch.randn(batch.shape, generator=generator, device=self.device, dtype=batch.dtype)
            steps = torch.full((count,), t, device=self.device, dtype=torch.long)
            noisy = self.scheduler.add_noise(batch, noise, steps)
            prediction = self.unet(noisy, steps,
                                   encoder_hidden_states=self.null_prompt.repeat(count, 1, 1)).sample
            total += F.mse_loss(prediction, noise).item() * count
        return total / self.noise_samples

    @torch.inference_mode()
    def score(self, path: str | Path, threshold: float | None = None):
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        with Image.open(path) as image:
            x = self.preprocess(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(self.device)
        # Reset per image, so repeated calls with identical settings are reproducible
        # within the same software/hardware environment. Cross-device equality is not guaranteed.
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        rotated = torch.rot90(x, 1, [2, 3])
        si = image_score(self._ae_error(x, generator), self._ae_error(self.blur(x), generator),
                         self._ae_error(rotated, generator),
                         self._ae_error(self.blur(rotated), generator), self.rotation_weight)
        z = self.vae.encode(x).latent_dist.sample(generator=generator) * self.vae.config.scaling_factor
        original_error = self._latent_error(z, generator)
        sl = self._latent_error(torch.rot90(z, 1, [2, 3]), generator) - original_error
        score = combine_scores(si, sl)
        if not all(math.isfinite(v) for v in (si, sl, score)):
            raise RuntimeError("Non-finite score; inspect image and model numerical stability")
        return DetectionResult(si, sl, score, threshold,
                               None if threshold is None else classify(score, threshold))

