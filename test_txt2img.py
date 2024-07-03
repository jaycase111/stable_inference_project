import time
from cache_manager.lora_manager import LoraManager
from pipeline.text2img_pipeline import Text2ImgPipeline


role_lora_tag = "cinematic"

lora_manager = LoraManager()
role_lora_file = lora_manager.query_lora_file(role_lora_tag)

lora_config = {"role": role_lora_file}
lcm_file = "/root/autodl-tmp/lcm_loras/pytorch_lora_weights.safetensors"
pipeline_path = "/root/autodl-tmp/sdxl_download/sdxl-base"
lora_root_path = "/root/autodl-tmp/loras"


pipeline = Text2ImgPipeline(
    pipeline_path=pipeline_path,
    preset_lora_config=lora_config,
    lcm_file=lcm_file,
    lora_root_path=lora_root_path
)

input_ = {"prompt": "1 girl, <lora:cinematic>",
          "height": 1024,
          "width": 1024,
          "num_inference_steps": 8,
          "num_images_per_prompt": 2,

          }

for i in range(3):
    pipeline.generate(input_)
images = []

time_start = time.time()
for i in range(10):
    images = pipeline.generate(input_)
images[0].save('lora_output.png')

print(f'Inference time: {time.time() - time_start:.3f}s')

