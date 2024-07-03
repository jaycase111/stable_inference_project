
class Text2ImgPipeline:

    """
        分区SD绘图底层推理加速项目
    """

    def __init__(self,
                 pipeline_path: str,
                 preset_lora_config: dict,
                 lora_root_path="",
                 lcm_file: str = None,
                 ):
        """
        :param pipeline_path:           预训练模型地址
        :param preset_lora_config:      预设置lora配置
        :param lora_root_path:          lora管理地址
        :param lcm_file:                lcm-lora地址
        """
        pass