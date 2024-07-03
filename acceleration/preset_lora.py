import torch
from safetensors.torch import load_file, save_file

def init_save_zeros_safetensors(input_safetensors: str,
                                zero_save_safetensors: str):
    """
    :param input_safetensors:           输入safetensors文件
    :param zero_save_safetensors:       与输入safetensors维度一致、数值全为0的safetensors文件
    input_safetensors: 样例lora
    zero_save_safetensors: 样例初始化全为0Lora、便于Lora切换
    :return:
    """
    print("input_safetensors: ", input_safetensors)
    state_dict = load_file(input_safetensors)
    zero_state_dict = {key:torch.zeros_like(value, dtype=value.dtype) for key, value in state_dict.items()}
    save_file(zero_state_dict, zero_save_safetensors)


class PresetLora:

    zero_role_file = "zero_role_empty.safetensors"

    role: str = "example_role.safetensors"

    """
        推理 动态Lora切换数据类
        当前支持:
            角色Lora
        TODO: 后续支持更多样Lora挂载、推理要求为后续输入同类型的Lora文件、维度必须一致
    """

    def __init__(self,
                 lora_config: dict):
        self.role = lora_config["role"]
        init_save_zeros_safetensors(self.role, self.zero_role_file)

    def get_support_loras(self):
        return ["role"]

    def get_zero_role_file(self, key: str):
        return getattr(self, f"zero_{key}_file")


