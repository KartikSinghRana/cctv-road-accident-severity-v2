from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import RLock

import numpy as np
import timm
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from torchvision import transforms


EXPECTED_CLASSES = ("No_Accident", "Minor_Accident", "Severe_Accident")
EXPECTED_MODEL = "tf_efficientnetv2_s"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_IMAGE_PIXELS = 20_000_000
MAX_DISPLAY_SIDE = 1200


@dataclass
class LoadedModel:
    model: torch.nn.Module
    class_names: tuple[str, ...]
    image_size: int
    device: torch.device
    target_layer: torch.nn.Module
    inference_lock: RLock


def load_trained_model(checkpoint_path: str) -> LoadedModel:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {path}. Run the training notebook first."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint has an unsupported format.")

    class_names = tuple(checkpoint.get("class_names", ()))
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            "Checkpoint classes do not match the dashboard's expected class order: "
            f"{class_names!r}."
        )

    model_name = checkpoint.get("model_name", EXPECTED_MODEL)
    if model_name != EXPECTED_MODEL:
        raise ValueError(f"Unsupported checkpoint model architecture: {model_name!r}.")

    image_size = checkpoint.get("image_size")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Checkpoint does not contain a valid input image size.")

    model = timm.create_model(
        EXPECTED_MODEL,
        pretrained=False,
        num_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("Model architecture has no supported feature block for Grad-CAM.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return LoadedModel(
        model=model,
        class_names=class_names,
        image_size=image_size,
        device=device,
        target_layer=blocks[-1],
        inference_lock=RLock(),
    )


def decode_uploaded_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image dimensions must be valid and no larger than "
                    f"{MAX_IMAGE_PIXELS:,} pixels total."
                )
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded file is not a supported, valid image.") from error


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values - values.min()
    maximum = values.max()
    if maximum.item() > 0:
        values = values / maximum
    return values


def _overlay_heatmap(
    image: Image.Image,
    values: torch.Tensor,
    color: str,
) -> Image.Image:
    map_array = (values.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    grayscale = Image.fromarray(map_array, mode="L").resize(
        image.size,
        Image.Resampling.BILINEAR,
    )
    color_map = ImageOps.colorize(
        grayscale,
        black="#07111f",
        white=color,
    )
    return Image.blend(image, color_map, alpha=0.48)


def predict_with_explanations(
    loaded: LoadedModel,
    image: Image.Image,
    target_class: str | None = None,
) -> dict[str, object]:
    with loaded.inference_lock:
        return _predict_with_explanations(loaded, image, target_class)


def _predict_with_explanations(
    loaded: LoadedModel,
    image: Image.Image,
    target_class: str | None,
) -> dict[str, object]:
    if target_class is not None and target_class not in loaded.class_names:
        raise ValueError(f"Unknown explanation target class: {target_class!r}.")

    display_image = ImageOps.contain(
        image.convert("RGB"),
        (MAX_DISPLAY_SIDE, MAX_DISPLAY_SIDE),
        method=Image.Resampling.LANCZOS,
    )
    preprocess = transforms.Compose(
        [
            transforms.Resize((loaded.image_size, loaded.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(loaded.device)
    input_tensor.requires_grad_(True)

    captured: dict[str, torch.Tensor] = {}

    def capture_features(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["features"] = output

        def capture_gradients(gradient: torch.Tensor) -> None:
            captured["gradients"] = gradient

        output.register_hook(capture_gradients)

    hook = loaded.target_layer.register_forward_hook(capture_features)
    try:
        loaded.model.zero_grad(set_to_none=True)
        logits = loaded.model(input_tensor)
        probabilities = torch.softmax(logits.detach()[0], dim=0)
        predicted_index = int(probabilities.argmax().item())
        target_index = (
            predicted_index
            if target_class is None
            else loaded.class_names.index(target_class)
        )
        logits[0, target_index].backward()
    finally:
        hook.remove()
        loaded.model.zero_grad(set_to_none=True)

    if "features" not in captured or "gradients" not in captured:
        raise RuntimeError("Grad-CAM feature gradients were not captured.")

    features = captured["features"][0]
    gradients = captured["gradients"][0]
    channel_weights = gradients.mean(dim=(1, 2), keepdim=True)
    gradcam = (channel_weights * features).sum(dim=0).relu()
    gradcam = _normalize_map(gradcam)
    gradcam = functional.interpolate(
        gradcam[None, None],
        size=(loaded.image_size, loaded.image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    saliency = input_tensor.grad[0].detach().abs().amax(dim=0)
    saliency = _normalize_map(saliency)

    probabilities_by_class = {
        class_name: float(probabilities[index].item())
        for index, class_name in enumerate(loaded.class_names)
    }

    return {
        "predicted_class": loaded.class_names[predicted_index],
        "confidence": probabilities_by_class[loaded.class_names[predicted_index]],
        "probabilities": probabilities_by_class,
        "target_class": loaded.class_names[target_index],
        "display_image": display_image,
        "gradcam_overlay": _overlay_heatmap(display_image, gradcam, "#ff4f81"),
        "saliency_overlay": _overlay_heatmap(display_image, saliency, "#62e6c5"),
        "device": str(loaded.device),
        "image_size": loaded.image_size,
    }
