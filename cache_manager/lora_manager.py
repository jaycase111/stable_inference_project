import os


class LoraManager:

    def __init__(self,
                 lora_root_path: str = "/root/autodl-tmp/loras"):
        """
        :param lora_root_path:  lora角色存储根目录
                                    两级目录: 第一级别目录为lora属性, 如: role、colthe
                                        第二级别目录为真实safetensors 文件
        """
        self.lora_root_path = lora_root_path
        self.tag_files_map = self._read_lora_files()

    def _read_lora_files(self):
        """
            TODO: 定时扫描Lora文件库
        """
        tag_files_map = {}
        lora_tags = os.listdir(self.lora_root_path)
        for tag in lora_tags:
            tag_path = os.path.join(self.lora_root_path, tag)
            tag = tag.split(".")[0]
            tag_files_map[tag] = tag_path
        return tag_files_map

    def query_lora_tag(self, lora_tag: str):
        # TODO 补充真实Lora的属性管理功能
        return "role"

    def query_lora_file(self, lora_tag: str):
        """
        :param lora_tag: lora 标签
        TODO 当前输入lora_file 为 kobe 这种tag、当前比较方式较为简单、后续优化
        """
        if lora_tag in self.tag_files_map.keys():
            return self.tag_files_map[lora_tag]
        else:
            return None

