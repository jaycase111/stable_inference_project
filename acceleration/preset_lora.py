import torch
import safetensors


def init_save_zeros_safetensors(input_safetensors: str,
                                zero_save_safetensors: str):
    """
    :param input_safetensors:           输入safetensors文件
    :param zero_save_safetensors:       与输入safetensors维度一致、数值全为0的safetensors文件
    :return:
    """
    state_dict = safetensors.torch.load(input_safetensors, map_location=torch.device('cpu'))
    zero_state_dict = {key:torch.zeros_like(value) for key, value in state_dict.items()}
    safetensors.torch.save(zero_state_dict, zero_save_safetensors)


class PresetLora:

    zero_role_file = "zero_role_empty.safetensors"
    zero_colthe_file = "zero_colthe_empty.safetensors"

    role: str = "example_role.safetensors"
    colthe: str = "example_colthe.safetensors"

    """
        推理 动态Lora切换数据类
        当前支持:
            角色Lora
            服装Lora
        TODO: 后续支持更多样Lora挂载、推理要求为后续输入同类型的Lora文件、维度必须一致
    """

    def __init__(self,
                 lora_config: dict):
        self.role = lora_config["role"]
        self.colthe = lora_config["colthe"]

        init_save_zeros_safetensors(self.role, self.zero_role_file)
        init_save_zeros_safetensors(self.colthe, self.zero_colthe_file)

    def get_support_loras(self):
        return ["role", "colthe"]

    def get_zero_role_file(self, key: str):
        return getattr(self, f"zero_{key}_file")


