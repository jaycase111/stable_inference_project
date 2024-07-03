import time
from diffusers import LMSDiscreteScheduler
from thrid_party.mixdiff.canvas import StableDiffusionXLCanvasPipeline, Text2ImageRegion

# Creater scheduler and model (similar to StableDiffusionPipeline)
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipeline = StableDiffusionXLCanvasPipeline.from_pretrained("/root/autodl-tmp/sdxl_download/sdxl-base", scheduler=scheduler, use_auth_token=True).to("cuda:0")

# Mixture of Diffusers generation
time_start = time.time()
image = pipeline(
    canvas_height=640,
    canvas_width=1408,
    regions=[
        Text2ImageRegion(0, 640, 0, 640, guidance_scale=8,
            prompt=f"A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
        Text2ImageRegion(0, 640, 384, 1024, guidance_scale=8,
            prompt=f"A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
        Text2ImageRegion(0, 640, 768, 1408, guidance_scale=8,
            prompt=f"An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
    ],
    num_inference_steps=50,
    seed=7178915308,
)["sample"][0]

print(f'Inference time: {time.time() - time_start:.3f}s')
image.save("partition.png")


# from PIL import Image
# from diffusers import LMSDiscreteScheduler, DiffusionPipeline
# from diffusers.pipelines.pipeline_utils import Image2ImageRegion, Text2ImageRegion, preprocess_image
#
#
# # Load and preprocess guide image
# iic_image = preprocess_image(Image.open("input_image.png").convert("RGB"))
#
# # Creater scheduler and model (similar to StableDiffusionPipeline)
# scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
# pipeline = DiffusionPipeline.from_pretrained("/root/autodl-tmp/sdxl_download/sdxl-base", scheduler=scheduler).to("cuda:0", custom_pipeline="mixture_canvas")
# pipeline.to("cuda")
#
# # Mixture of Diffusers generation
# output = pipeline(
#     canvas_height=800,
#     canvas_width=352,
#     regions=[
#         Text2ImageRegion(0, 800, 0, 352, guidance_scale=8,
#             prompt=f"best quality, masterpiece, WLOP, sakimichan, art contest winner on pixiv, 8K, intricate details, wet effects, rain drops, ethereal, mysterious, futuristic, UHD, HDR, cinematic lighting, in a beautiful forest, rainy day, award winning, trending on artstation, beautiful confident cheerful young woman, wearing a futuristic sleeveless dress, ultra beautiful detailed  eyes, hyper-detailed face, complex,  perfect, model,  textured,  chiaroscuro, professional make-up, realistic, figure in frame, "),
#         Image2ImageRegion(352-800, 352, 0, 352, reference_image=iic_image, strength=1.0),
#     ],
#     num_inference_steps=100,
#     seed=5525475061,
# )["images"][0]