"""
    单人基础Prompt语法解析类库
    参考代码: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/extra_networks.py
"""

import re
from collections import defaultdict

"""
    单人Prompt解析
"""

re_extra_net = re.compile(r"<(\w+):([^>]+)>")

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)


class ExtraNetworkParams:
    def __init__(self, items=None):
        self.items = items or []
        self.positional = []
        self.named = {}

        for item in self.items:
            parts = item.split('=', 2) if isinstance(item, str) else [item]
            if len(parts) == 2:
                self.named[parts[0]] = parts[1]
            else:
                self.positional.append(item)

    def __eq__(self, other):
        return self.items == other.items


def parse_prompt_attention(text):
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def parse_prompt(prompt):
    res = defaultdict(list)

    def found(m):
        name = m.group(1)
        args = m.group(2)

        res[name].append(ExtraNetworkParams(items=args.split(":")))

        return ""

    prompt = re.sub(re_extra_net, found, prompt)

    return prompt, res


def parse_prompts(prompts):
    res = []
    extra_data = None

    for prompt in prompts:
        updated_prompt, parsed_extra_data = parse_prompt(prompt)

        if extra_data is None:
            extra_data = parsed_extra_data

        res.append(updated_prompt)

    return res, extra_data


def parse_extra_network_prompts(prompts):
    """
    :param prompts: Prompt列表
    :return:    prompts: 返回Prompts输入中每个prompt抽取的生成Prompt
                extra_network_data: 输入Promps批次中需要加载的网络和Lora
    """
    prompts, extra_network_data = parse_prompts(prompts)
    return prompts, extra_network_data


def parse_prompt_not_weight(prompt: str):
    """
    :param prompt: 文生图Prompt
    :return:       清除Lora标志、以及word权重的干净文本
    """
    prompt_text, _ = parse_prompt(prompt)
    prompt_text_weight = parse_prompt_attention(prompt_text)
    return " ".join([prompt_weight[0] for prompt_weight in prompt_text_weight]).strip()


def parse_loras(prompt: str):
    _, extra_network_data = parse_prompt(prompt)
    lora_data = extra_network_data["lora"] if "lora" in extra_network_data else []
    return lora_data
