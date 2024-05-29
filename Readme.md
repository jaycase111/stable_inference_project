Stable-Diffusion推理框架 基于Diffusers实现


高级特性1、基于Diffusers库重新实现以下功能

    1、文生图                            (DONE)

    2、分区生图                          (DONE)

    3、ControlNet-文生图                 (DONE)

    4、修脸插件                          (DONE)

意义: 基于Diffuser完成对WebUI和ComfyUI基础功能完成解耦、代码更加简洁、且服务分离、便于后续二次开发以及服务流量控制


高级特型2、实现语法解析树

    1、支持原始SD语法解析Prompt、调用上述插件   (DONE)


高级特性3、融合Stable-Fast实现各基础库加速

    1、上述功能结合Stable-Fast推理库实现对比原始Torch实现1倍推理加速     (DONE)

    2、动态加载LCM-Lora、获得推理步数降低、Step 20 降低到 step 8       (DONE)

    3、使用SDXL-Turbo | SDXL-Lighting 轻量级模型推理                (DONE)


高级特性4、实现队列机制

    1、实现队列机制、以便复杂、高并发线上情况使用

    2、优先级队列机制、便于请求插队、实现VIP用户插队机制


TODO:

    (1) 实现队列机制

    (2) 实现接口开发

    (3) 实现简单WEB页面



