import os


class LoraManager:

    def __init__(self,
                 lora_root_path: str):
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
            lora_files = [os.path.join(tag_path, file) for file in os.listdir(tag_path)]
            tag_files_map[tag] = lora_files
        return tag_files_map

    def query_lora_file(self, lora_file: str):
        """
        :param lora_file: lora 标签
        :return: 该lora标签对应的tag 以及 file文件
        TODO 当前输入lora_file 为 kobe 这种tag、当前比较方式较为简单、后续优化
        """
        for tag, files in self.tag_files_map.items():
            for file in files:
                if lora_file in file:
                    return tag, file
        return None, None
