"""Frozen encoder and image preprocessing used by the current model."""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import timm
import torch
from torch import nn
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)
from torchvision.transforms import v2

IMAGE_SIZE = 224


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoder(name: str, dino_weights: Path | None) -> nn.Module:
    if name == "convnext":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Identity()
    elif name == "efficientnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Identity()
    else:
        if dino_weights is None:
            raise ValueError("DINOv2 requires --dino-weights.")
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(load_file(dino_weights), strict=True)
    return model.eval()


def embeddings(
    model: nn.Module, files: dict[str, Path | Image.Image], pad: bool
) -> dict[str, np.ndarray]:
    normalize = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    resize = v2.Compose([v2.Resize(IMAGE_SIZE, antialias=True), v2.CenterCrop(IMAGE_SIZE)])
    result = {}
    items = sorted(files.items())
    for start in range(0, len(items), 24):
        batch = items[start : start + 24]
        tensors = []
        for _, path in batch:
            if isinstance(path, Image.Image):
                image = ImageOps.exif_transpose(path).convert("RGB")
            else:
                with Image.open(path) as handle:
                    image = ImageOps.exif_transpose(handle).convert("RGB")
            if pad:
                image = ImageOps.pad(
                    image,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    method=Image.Resampling.BICUBIC,
                    color=(124, 116, 104),
                )
            else:
                image = resize(image)
            tensors.append(normalize(image))

        with torch.inference_mode():
            values = model(torch.stack(tensors)).numpy()
        result.update(zip((key for key, _ in batch), values, strict=True))
    return result
