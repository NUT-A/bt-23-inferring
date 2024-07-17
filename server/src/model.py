import os
from pydantic import BaseModel
import requests
from tqdm import tqdm

import torch
import diffusers
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler
from onediff.infer_compiler import oneflow_compile

class PipelineParams(BaseModel):
    num_inference_steps: int
    width: int
    height: int
    guidance_scale: float
    negative_prompt: str

# Define the request body model
class GenerateRequest(BaseModel):
    prompt: str
    seed: int
    pipeline_params: PipelineParams
    timeout: float
    model_name: str
    pipeline_type: str

class AnimeModel:
    def __init__(self, model_path="checkpoints/AnimeV3.safetensors"):
        self.model_path = model_path
        self.pipe = AnimeModel.get_pipe(model_path)

    @staticmethod
    def download_checkpoint(download_url, checkpoint_file):
        if os.path.exists(checkpoint_file):
            print(f"{checkpoint_file} already exists. Skipping download.")
            return

        folder, filename = os.path.split(checkpoint_file)
        os.makedirs(folder, exist_ok=True)
        with requests.get(download_url, stream=True) as response:
            total_size = int(response.headers.get("content-length", 0))
            with open(checkpoint_file, "wb") as file_stream, tqdm(
                desc=checkpoint_file,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file_stream.write(data)
                    progress_bar.update(len(data))

        print("Download completed successfully.")

    @staticmethod
    def download_anime_model(model_path):
        AnimeModel.download_checkpoint("https://civitai.com/api/download/models/173961", model_path)

    @staticmethod
    def optimise_torch():
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

    @staticmethod
    def get_scheduler(config):
        return DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True
        )

    @staticmethod
    def optimise_model(pipe):
        pipe.unet = oneflow_compile(pipe.unet)
        pipe.vae = oneflow_compile(pipe.vae)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

    @staticmethod
    def get_pipe(model_path: str):
        AnimeModel.download_anime_model(model_path)
        
        vae = AutoencoderTiny.from_pretrained(
            'madebyollin/taesdxl',
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to('cuda')

        txt2img_pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
            model_path,
            use_safetensors=True,
            vae=vae,
            torch_dtype=torch.float16
        )

        txt2img_pipe.scheduler = AnimeModel.get_scheduler(
            txt2img_pipe.scheduler.config
        )

        txt2img_pipe.to("cuda")
        
        AnimeModel.optimise_model(txt2img_pipe)
        
        return txt2img_pipe

    @staticmethod
    def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
        if step_index == int(pipe.num_timesteps * 0.4):
            callback_kwargs['prompt_embeds'] = callback_kwargs['prompt_embeds'].chunk(2)[-1]
            callback_kwargs['add_text_embeds'] = callback_kwargs['add_text_embeds'].chunk(2)[-1]
            callback_kwargs['add_time_ids'] = callback_kwargs['add_time_ids'].chunk(2)[-1]
            pipe._guidance_scale = 0.0

        return callback_kwargs
    
    def preheat_model(self):
        print("Preheating model...")
        self.infer(
            "a beautiful anime girl", 420, PipelineParams(
                num_inference_steps=20,
                width=1024,
                height=1024,
                guidance_scale=7.0,
                negative_prompt="out of frame, nude, duplicate, watermark, signature, mutated, text, blurry, worst quality, low quality, artificial, texture artifacts, jpeg artifacts"
            )
        )
        
        print("Model preheated")

    def infer(self, prompt: str, seed: int, pipeline_params: PipelineParams):
        generator = torch.manual_seed(seed)
        
        return self.pipe(
            generator=generator,
            prompt=prompt,
            negative_prompt=pipeline_params.negative_prompt,
            num_inference_steps=int(pipeline_params.num_inference_steps * 0.8),
            callback_on_step_end=AnimeModel.callback_dynamic_cfg,
            callback_on_step_end_tensor_inputs=['prompt_embeds', 'add_text_embeds', 'add_time_ids'],
            guidance_scale=pipeline_params.guidance_scale,
            width=pipeline_params.width,
            height=pipeline_params.height,
            num_images_per_prompt=1
        ).images[0]