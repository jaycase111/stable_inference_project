import torch
from typing import List
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from cache_manager.lora_manager import LoraManager
from acceleration.stable_fast_accelerate import StableFastCompilePipeline
from prompt_parser.base_prompt_parser import parse_prompt_not_weight, parse_loras, ExtraNetworkParams


class Text2ImgPipeline:

    def __init__(self,
                 pipeline_path: str,
                 preset_lora_config: dict,
                 lora_root_path="",
                 lcm_file: str = None,
                 ):
        """
        :param pipeline_path:             模型Pipeline地址
        :param lcm_file:                  LCM-Lora地址
        :param preset_lora_config:        Lora配置文件
        """

        self.pipeline_path = pipeline_path
        self.preset_lora_config = preset_lora_config
        self.lcm_file = lcm_file
        self.lora_manager = LoraManager(lora_root_path)
        self.compile_pipeline = self._init_compile_model()

    def _init_compile_model(self):
        pipeline = StableDiffusionXLPipeline.from_pretrained(self.pipeline_path,
                                                             torch_dtype=torch.float16,
                                                             )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config)
        pipeline.safety_checker = None
        pipeline.to(torch.device('cuda'))
        compile_pipeline = StableFastCompilePipeline(
            pipeline, self.preset_lora_config, self.lcm_file
        )
        return compile_pipeline

    def _get_lora_tag_and_path(self,
                               lora_list: List[ExtraNetworkParams]):
        """
        :param lora_list:
        :return:  TODO:
                        (1) 当前不支持Lora weight挂载、后续支持
                        (2) 当前一个类型仅支持挂载一个Lora
        """
        tag_lora_map = {}
        for entity in lora_list:
            positional = entity.positional
            lora_name = positional[0]
            weight = positional[1] if len(positional) > 1 else 1.
            print("lora_name and weight: ", lora_name, weight)
            tag = self.lora_manager.query_lora_tag(lora_name)
            file_path = self.lora_manager.query_lora_file(lora_name)
            if not file_path:
                tag_lora_map[tag] = [file_path, weight]
        return tag_lora_map

    def generate(self, input: dict):
        """
        :param
        input: 文生图请求参数字典
        常用字段:
                prompt:                 生成Prompt
                height:                 生成图片高度
                width:                  生成图片宽度
                num_inference_steps:    推理步数
                num_images_per_prompt:  生成图像个数
                guidance_scale:         提示词相关性
        :return:
        """
        prompt = input["prompt"]
        prompt_text = parse_prompt_not_weight(prompt)
        loras = parse_loras(prompt)
        tag_lora_map = self._get_lora_tag_and_path(loras)
        self.compile_pipeline.reset_loras_to_zero()
        for tag, (lora_file, lora_weight) in tag_lora_map.items():
            self.compile_pipeline.load_lora_file(tag, lora_file, lora_weight)
        input["prompt"] = prompt_text
        output_images = self.compile_pipeline.generate(input)
        return output_images
