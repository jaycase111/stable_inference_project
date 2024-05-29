import torch
from diffusers import AutoPipelineForImage2Image
from pipelines.text2img_pipeline import Text2ImgPipeline
from acceleration.stable_fast_accelerate import StableFastCompilePipeline


class ControlnetText2ImgPipeline(Text2ImgPipeline):
    
    def __init__(self,
                 pipeline_path: str,
                 preset_lora_config: dict,
                 lora_root_path="",
                 lcm_file: str = None,
                 ):
        super().__init__(pipeline_path, preset_lora_config,
                         lora_root_path, lcm_file)
        
    def _init_compile_model(self):
        pipeline = AutoPipelineForImage2Image.from_pretrained(self.pipeline_path,
                                                             torch_dtype=torch.float16,
                                                             use_safetensors=True, variant="fp16"
                                                             )
        self.compile_pipeline = StableFastCompilePipeline(
            pipeline, self.preset_lora_config, self.lcm_file
        )
        return self.compile_pipeline
    
    def generate(self, input: dict):
        """
                :param
                input: 文生图请求参数字典
                常用字段:
                        prompt:                 生成Prompt
                        image:                  controlnet 参考图片
                        height:                 生成图片高度
                        width:                  生成图片宽度
                        num_inference_steps:    推理步数
                        num_images_per_prompt:  生成图像个数
                        guidance_scale:         提示词相关性
                :return:
                """
        return super().generate(input)

