import time
import torch
import importlib
from diffusers import (StableDiffusionPipeline,
                        StableDiffusionXLPipeline,
                       EulerAncestralDiscreteScheduler)
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)

def load_model():
    model = StableDiffusionXLPipeline.from_pretrained(
        '/root/autodl-tmp/sdxl_download/sdxl-base',
        torch_dtype=torch.float16)

    model.scheduler = EulerAncestralDiscreteScheduler.from_config(
        model.scheduler.config)
    model.safety_checker = None
    model.to(torch.device('cuda'))
    return model

model = load_model()

config = CompilationConfig.Default()
# xformers and Triton are suggested for achieving best performance.
try:
    import xformers
    config.enable_xformers = True
except ImportError:
    print('xformers not installed, skip')
try:
    import triton
    config.enable_triton = True
except ImportError:
    print('Triton not installed, skip')
# CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
# But it can increase the amount of GPU memory used.
# For StableVideoDiffusionPipeline it is not needed.
config.enable_cuda_graph = True


lora = "/root/autodl-tmp/lcm_loras/pytorch_lora_weights.safetensors"

scheduler = "LCMScheduler"
scheduler_cls = getattr(importlib.import_module('diffusers'),
                                scheduler)
model.scheduler = scheduler_cls.from_config(model.scheduler.config)

model.load_lora_weights(lora)
model.fuse_lora()

model.load_lora_weights("/root/autodl-tmp/pytorch_lora_weights-3.safetensors", adapter_name="test")

model = compile(model, config)

kwarg_inputs = dict(
    prompt=
    'a beautiful girl',
    negative_prompt="bad picture",
    height=1024,
    width=1024,
    num_inference_steps=8,
    num_images_per_prompt=2,
)

# NOTE: Warm it up.
# The initial calls will trigger compilation and might be very slow.
# After that, it should be very fast.
for _ in range(3):
    output_image = model(**kwarg_inputs).images[0]

# Let's see it!
# Note: Progress bar might work incorrectly due to the async nature of CUDA.
begin = time.time()
for i in range(10):
    output_image = model(**kwarg_inputs).images[0]
    output_image.save("output.png")
print(f'Inference time: {time.time() - begin:.3f}s')

# Let's view it in terminal!
from sfast.utils.term_image import print_image

print_image(output_image, max_width=80)