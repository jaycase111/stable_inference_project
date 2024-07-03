"""
    SD 线上推理服务入口
"""
import time
import torch
from diffusers import (StableDiffusionPipeline,
                        StableDiffusionXLPipeline,
                       EulerAncestralDiscreteScheduler)
import importlib

kwarg_inputs = dict(
    prompt=
    'a beautiful girl',
    negative_prompt="bad picture",
    height=1024,
    width=1024,
    num_inference_steps=8,
    num_images_per_prompt=2,
)

def load_model():
    model = StableDiffusionXLPipeline.from_pretrained(
        '/root/autodl-tmp/sdxl_download/sdxl-base',
        torch_dtype=torch.float16)

    scheduler = "LCMScheduler"
    scheduler_cls = getattr(importlib.import_module('diffusers'),
                            scheduler)
    model.scheduler = scheduler_cls.from_config(model.scheduler.config)
    model.safety_checker = None
    model.to(torch.device('cuda'))
    return model

model = load_model()


lora = "/root/autodl-tmp/lcm_loras/pytorch_lora_weights.safetensors"
model.load_lora_weights(lora)
model.fuse_lora()

output_image = []
begin = time.time()
for i in range(10):
    model.load_lora_weights("/root/autodl-tmp/loras/cinematic.safetensors")
    output_image = model(**kwarg_inputs).images[0]
output_image.save("output_test.png")
print(f'Inference time: {time.time() - begin:.3f}s')