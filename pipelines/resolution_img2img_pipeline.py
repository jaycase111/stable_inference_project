from __future__ import annotations

import torch
from PIL import Image
from typing import Callable, List, Any, Mapping, Iterable
from functools import cached_property
from layers.yolo import YoloDetetor
from acceleration.stable_fast_accelerate import StableFastCompilePipeline
from asdff import AdPipeline

"""
    修脸Pipeline实现、参考: https://github.com/Bing-su/asdff
    人脸检测使用 Yolo
"""


class ResolutionImg2ImgPipeline:

    def __init__(self,
                 pipeline_path: str,
                 preset_lora_config: dict,
                 lora_root_path="",
                 lcm_file: str = None,
                 yolo_files: List[str] = []
                 ):
        """
        :param yolo_file:  图像部分修复功能如: (修脸、修手) 检测模型
        TODO: 当前修复模型未支持Lora解析后续支持
        """
        self.pipeline_path = pipeline_path
        self.preset_lora_config = preset_lora_config
        self.lora_root_path = lora_root_path
        self.lcm_file = lcm_file
        self.detectors = [YoloDetetor(yolo_file) for yolo_file in yolo_files]
        self.compile_pipeline = self._init_compile_model()

    def _init_compile_model(self):
        pipeline = AdPipeline.from_pretrained(self.pipeline_path,
                                                             torch_dtype=torch.float16,
                                                             use_safetensors=True, variant="fp16"
                                                             )
        inpaint_pipeline = StableFastCompilePipeline(
            pipeline.inpaint_pipeline, self.preset_lora_config, self.lcm_file
        )
        pipeline.inpaint_pipeline = inpaint_pipeline
        return pipeline

    def generate(self,
                 common: Mapping[str, Any] | None = None,
                 txt2img_only: Mapping[str, Any] | None = None,
                 inpaint_only: Mapping[str, Any] | None = None,
                 images: Image.Image | Iterable[Image.Image] | None = None,
                 mask_dilation: int = 4,
                 mask_blur: int = 4,
                 mask_padding: int = 32,
                 ):
        return self.compile_pipeline(
            common=common,
            txt2img_only=txt2img_only,
            inpaint_only=inpaint_only,
            images=images,
            detectors=self.detectors,
            mask_dilation=mask_dilation,
            mask_blur=mask_blur,
            mask_padding=mask_padding
        )
