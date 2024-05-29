import torch

from acceleration.preset_lora import PresetLora
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, LCMScheduler
from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)


def switch_lora(unet, lora, weight=1.):
    # TODO: 支持Lora-权重加载
    state_dict = unet.state_dict()
    unet.load_attn_procs(lora)
    update_state_dict(state_dict, unet.state_dict())
    unet.load_state_dict(state_dict, assign=True)

def reset_loras_to_zero(unet: torch.Module, preset_lora: PresetLora):
    """
    将之前加载的Lora参数 (除LCM以外) 全部置为0
    :return:
    """
    for lora_name in preset_lora.get_support_loras():
        zero_lora_file = preset_lora.get_zero_role_file(lora_name)
        switch_lora(unet, zero_lora_file)


def load_lora_file(
                    preset_lora: PresetLora,
                    unet: torch.Module,
                    tag: str,
                    lora_file: str,
                    weight: float = 1.
                       ):
        """
        推理引擎动态加载Lora
        :param tag:             待加载Lora文件类型
        :param lora_file:       待加载Lora文件
        :param weight:          权重
        TODO: 支持
        :return:
        """
        if tag not in preset_lora.get_support_loras():
            print(f" This {tag} lora type not support compile in stable-fast model")
            return
        switch_lora(unet, lora_file, weight)

def compile_stable_fast_pipeline(pipeline: AutoPipelineForText2Image):
    config = CompilationConfig.Default()
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
    config.enable_cuda_graph = True

    pipeline = compile(pipeline, config)
    return pipeline


def update_state_dict(dst, src):
    for key, value in src.items():
        dst[key].copy_(value)



class StableFastCompilePipeline:

    def __init__(self,
                 base_pipeline: AutoPipelineForText2Image,
                 preset_lora_config: dict,
                 lcm_lora_file: str = None,
                 ):
        # TODO: 当前Lora-weight配置为 LCM: 1、其他Lora为0.8 后续改成配置化
        if lcm_lora_file:
            base_pipeline.scheduler = LCMScheduler.from_config(base_pipeline.scheduler.config)
            base_pipeline.load_lora_weights(lcm_lora_file, adapter_name="lcm")
            lora_name_list, lora_weight_list = ["lcm"], [1.0]
        else:
            lora_name_list, lora_weight_list = [], []

        self.preset_lora = PresetLora(preset_lora_config)

        for lora_name in self.preset_lora.get_support_loras():
            lora_file = self.preset_lora.get_zero_role_file(lora_name)
            base_pipeline.load_lora_weights(lora_file, adapter_name=lora_name)
            lora_name_list.append(lora_name)
            lora_weight_list.append(0.8)

        base_pipeline = base_pipeline.set_adapters(lora_name_list, adapter_weights=lora_weight_list)
        self.pipeline = compile_stable_fast_pipeline(base_pipeline)

    def get_preset_lora(self) -> PresetLora:
        return self.preset_lora

    def load_lora_file(self,
                            tag: str,
                            lora_file: str,
                            weight: float = 1.
                       ):
        """
        推理引擎动态加载Lora
        :param tag:             待加载Lora文件类型
        :param lora_file:       待加载Lora文件
        :param weight:          权重
        TODO: 支持
        :return:
        """
        if tag not in self.preset_lora.get_support_loras():
            print(f" This {tag} lora type not support compile in stable-fast model")
            return
        switch_lora(self.pipeline.unet, lora_file, weight)

    def reset_loras_to_zero(self):
        """
        将之前加载的Lora参数 (除LCM以外) 全部置为0
        :return:
        """
        for lora_name in self.preset_lora.get_support_loras():
            zero_lora_file = self.preset_lora.get_zero_role_file(lora_name)
            switch_lora(self.pipeline.unet, zero_lora_file)

    def reset_lora_to_zero_by_key(self, tag):
        """
        :param tag:  指定重新设置为0的lora类型
        用处: 之前请求挂载的部分Lora本次还可用、只将部分Lora重新设置为0、以消除该部分Lora影响
        :return:
        """
        assert tag in self.preset_lora.get_support_loras(), f"not support {tag} type lora"
        zero_lora_file = self.preset_lora.get_zero_role_file(tag)
        switch_lora(self.pipeline.unet, zero_lora_file)

    def generate(self, input: dict):
        return self.pipeline(**input).images