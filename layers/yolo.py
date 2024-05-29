from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("Please install ultralytics using `pip install ultralytics`")
    raise


def create_mask_from_bbox(
    bboxes: np.ndarray, shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, "black")
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill="white")
        masks.append(mask)
    return masks


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image

    Returns
    -------
    images: list[Image.Image]
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


class YoloDetetor:

    def __init__(self,
                 model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image, confidence: float = 0.3):
        pred = self.model(image, conf=confidence)

        bboxes = pred[0].boxes.xyxy.cpu().numpy()
        if bboxes.size == 0:
            return None

        if pred[0].masks is None:
            masks = create_mask_from_bbox(bboxes, image.size)
        else:
            masks = mask_to_pil(pred[0].masks.data, image.size)

        return masks