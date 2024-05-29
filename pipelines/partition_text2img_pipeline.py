import torch
from tqdm import tqdm
from typing import List, Optional
from cache_manager.lora_manager import LoraManager
from diffusers import AutoPipelineForImage2Image
from acceleration.preset_lora import PresetLora
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from acceleration.stable_fast_accelerate import StableFastCompilePipeline, switch_lora, reset_loras_to_zero, \
    load_lora_file
from prompt_parser.base_prompt_parser import parse_prompt_not_weight, parse_loras, ExtraNetworkParams
from mixdiff import StableDiffusionCanvasPipeline, Text2ImageRegion, Image2ImageRegion, \
    DiffusionRegion, RerollRegion, RerollModes, MaskWeightsBuilder

"""
    分区插件实现参考: https://github.com/albarji/mixture-of-diffusers/blob/master/mixdiff/canvas.py
"""


class StableDiffusionFastCanvasPipeline(StableDiffusionCanvasPipeline):
    """
        当前类为StableDiffusionCanvasPipeline的子类、
            主要优化点为:
                (1) Stable-Fast 推理加速 
                (2) 动态Lora加载
    """

    @torch.no_grad()
    def __call__(
            self,
            canvas_height: int,
            canvas_width: int,
            regions: List[DiffusionRegion],
            regions_loras: List[dict],
            num_inference_steps: Optional[int] = 50,
            seed: Optional[int] = 12345,
            reroll_regions: Optional[List[RerollRegion]] = None,
            cpu_vae: Optional[bool] = False,
            decode_steps: Optional[bool] = False,
            preset_lora: PresetLora = None
    ):
        if reroll_regions is None:
            reroll_regions = []
        batch_size = 1

        if decode_steps:
            steps_images = []

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Split diffusion regions by their kind
        text2image_regions = [region for region in regions if isinstance(region, Text2ImageRegion)]
        text2image_loras = [lora for lora in regions_loras if lora is not None]
        assert len(text2image_loras) == len(
            text2image_regions), "canvas pipeline text2image regions length must equal textLora regions length"
        image2image_regions = [region for region in regions if isinstance(region, Image2ImageRegion)]

        # Prepare text embeddings
        for region in text2image_regions:
            region.tokenize_prompt(self.tokenizer)
            region.encode_prompt(self.text_encoder, self.device)

        # Create original noisy latents using the timesteps
        latents_shape = (batch_size, self.unet.config.in_channels, canvas_height // 8, canvas_width // 8)
        generator = torch.Generator(self.device).manual_seed(seed)
        init_noise = torch.randn(latents_shape, generator=generator, device=self.device)

        # Reset latents in seed reroll regions, if requested
        for region in reroll_regions:
            if region.reroll_mode == RerollModes.RESET.value:
                region_shape = (latents_shape[0], latents_shape[1], region.latent_row_end - region.latent_row_init,
                                region.latent_col_end - region.latent_col_init)
                init_noise[:, :, region.latent_row_init:region.latent_row_end,
                region.latent_col_init:region.latent_col_end] = torch.randn(region_shape,
                                                                            generator=region.get_region_generator(
                                                                                self.device), device=self.device)

        # Apply epsilon noise to regions: first diffusion regions, then reroll regions
        all_eps_rerolls = regions + [r for r in reroll_regions if r.reroll_mode == RerollModes.EPSILON.value]
        for region in all_eps_rerolls:
            if region.noise_eps > 0:
                region_noise = init_noise[:, :, region.latent_row_init:region.latent_row_end,
                               region.latent_col_init:region.latent_col_end]
                eps_noise = torch.randn(region_noise.shape, generator=region.get_region_generator(self.device),
                                        device=self.device) * region.noise_eps
                init_noise[:, :, region.latent_row_init:region.latent_row_end,
                region.latent_col_init:region.latent_col_end] += eps_noise

        # scale the initial noise by the standard deviation required by the scheduler
        latents = init_noise * self.scheduler.init_noise_sigma

        # Get unconditional embeddings for classifier free guidance in text2image regions
        for region in text2image_regions:
            max_length = region.tokenized_prompt.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            region.encoded_prompt = torch.cat([uncond_embeddings, region.encoded_prompt])

        # Prepare image latents
        for region in image2image_regions:
            region.encode_reference_image(self.vae, device=self.device, generator=generator)

        # Prepare mask of weights for each region
        mask_builder = MaskWeightsBuilder(latent_space_dim=self.unet.config.in_channels, nbatch=batch_size)
        mask_weights = [mask_builder.compute_mask_weights(region).to(self.device) for region in text2image_regions]

        # Diffusion timesteps
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # Diffuse each region            
            noise_preds_regions = []

            # text2image regions
            for region, tag_lora_map in zip(text2image_regions, text2image_loras):
                # TODO 清除所有加载Lora
                reset_loras_to_zero(self.unet, preset_lora)

                region_latents = latents[:, :, region.latent_row_init:region.latent_row_end,
                                 region.latent_col_init:region.latent_col_end]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([region_latents] * 2)
                # scale model input following scheduler rules
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                # TODO: 加载Lora
                for tag, (lora_file, lora_weight) in tag_lora_map.items():
                    load_lora_file(preset_lora, self.unet, tag, lora_file, lora_weight)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=region.encoded_prompt)["sample"]
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_region = noise_pred_uncond + region.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_preds_regions.append(noise_pred_region)

            # Merge noise predictions for all tiles
            noise_pred = torch.zeros(latents.shape, device=self.device)
            contributors = torch.zeros(latents.shape, device=self.device)
            # Add each tile contribution to overall latents
            for region, noise_pred_region, mask_weights_region in zip(text2image_regions, noise_preds_regions,
                                                                      mask_weights):
                noise_pred[:, :, region.latent_row_init:region.latent_row_end,
                region.latent_col_init:region.latent_col_end] += noise_pred_region * mask_weights_region
                contributors[:, :, region.latent_row_init:region.latent_row_end,
                region.latent_col_init:region.latent_col_end] += mask_weights_region
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            noise_pred = torch.nan_to_num(
                noise_pred)  # Replace NaNs by zeros: NaN can appear if a position is not covered by any DiffusionRegion

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Image2Image regions: override latents generated by the scheduler
            for region in image2image_regions:
                influence_step = self.get_latest_timestep_img2img(num_inference_steps, region.strength)
                # Only override in the timesteps before the last influence step of the image (given by its strength)
                if t > influence_step:
                    timestep = t.repeat(batch_size)
                    region_init_noise = init_noise[:, :, region.latent_row_init:region.latent_row_end,
                                        region.latent_col_init:region.latent_col_end]
                    region_latents = self.scheduler.add_noise(region.reference_latents, region_init_noise, timestep)
                    latents[:, :, region.latent_row_init:region.latent_row_end,
                    region.latent_col_init:region.latent_col_end] = region_latents

            if decode_steps:
                steps_images.append(self.decode_latents(latents, cpu_vae))

        # scale and decode the image latents with vae
        image = self.decode_latents(latents, cpu_vae)
        output = StableDiffusionXLPipelineOutput(images=image)
        return output


class PartitionText2ImgPipeline:

    def __init__(self,
                 pipeline_path: str,
                 preset_lora_config: dict,
                 lora_root_path="",
                 lcm_file: str = None,
                 ):
        self.pipeline_path = pipeline_path
        self.preset_lora_config = preset_lora_config
        self.lcm_file = lcm_file
        self.lora_manager = LoraManager(lora_root_path)

        self.compile_pipeline, self.preset_lora = self._init_compile_model()

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
            tag, file_path = self.lora_manager.query_lora_file(lora_name)
            if not tag:
                tag_lora_map[tag] = [file_path, weight]
        return tag_lora_map

    def _init_compile_model(self):
        pipeline = StableDiffusionFastCanvasPipeline.from_pretrained(self.pipeline_path,
                                                                     torch_dtype=torch.float16,
                                                                     use_safetensors=True, variant="fp16"
                                                                     )

        # TODO: 测试stable-fast 的加速编译可否直接使用
        compile_pipeline = StableFastCompilePipeline(
            pipeline, self.preset_lora_config, self.lcm_file
        )
        pipeline.unet = compile_pipeline.unet
        return pipeline, compile_pipeline.get_preset_lora()

    def _get_regional_loras(self,
                            regions: List[DiffusionRegion]):
        prompt_list = [region.prompt if hasattr(region, "prompt") else None for region in regions]
        lora_extra_list = [parse_loras(prompt) if not prompt else None for prompt in prompt_list]
        lora_map_list = [self._get_lora_tag_and_path(lora_extra) if not lora_extra else None for lora_extra in
                         lora_extra_list]
        return lora_map_list

    def _get_region_prompt_text(self,
                                regions: List[DiffusionRegion]):
        regions_prompt = []
        for region in regions:
            if isinstance(region, Text2ImageRegion):
                prompt = parse_prompt_not_weight(region.prompt)
                region.prompt = prompt
            else:
                pass
            regions_prompt.append(region)
        return regions_prompt

    def generate(self,
                 canvas_height: int,
                 canvas_width: int,
                 regions: List[DiffusionRegion],
                 num_inference_steps: Optional[int] = 50,
                 seed: Optional[int] = 12345,
                 reroll_regions: Optional[List[RerollRegion]] = None,
                 decode_steps: Optional[bool] = False
                 ):
        """
        :param canvas_height:
        :param canvas_width:
        :param regions:
        :param num_inference_steps:
        :param seed:
        :param reroll_regions:
        :param decode_steps:
        :return:
        当前使用 Stable-Fast 框架加速、不再支持CPU推理、现废止cpu_vae 参数
        """
        region_loras = self._get_regional_loras(regions)
        regions = self._get_region_prompt_text(regions)
        return self.compile_pipeline(
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            regions=regions,
            regions_loras=region_loras,
            num_inference_steps=num_inference_steps,
            seed=seed,
            reroll_regions=reroll_regions,
            # cpu_vae = cpu_vae,
            decode_steps=decode_steps,
            preset_lora=self.preset_lora
        )
