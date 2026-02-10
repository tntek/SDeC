import torch
from typing import Optional
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler, EDMDPMSolverMultistepScheduler
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from scipy.spatial.distance import cdist
import numpy as np
import unet.pipeline_stable_diffusion_xl as pipeline_stable_diffusion_xl
from torch.fft import fftn, fftshift, ifftn, ifftshift
from typing import Optional, Tuple

from unet.unet import UNet2DConditionModel

from train import get_final_embedding

from diffusers import DDIMScheduler
from diffusers.utils import logging
import time
logging.disable_progress_bar()

def load_pipe_from_path(model_path, device, torch_dtype, variant):
    model_name = model_path.split('/')[-1]
    if model_path.split('/')[-1] == 'playground-v2.5-1024px-aesthetic':
        scheduler = EDMDPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype, variant=variant,)
    else:
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype, variant=variant,)
        #scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    if model_path.split('/')[-1] == 'Juggernaut-X-v10' or model_path.split('/')[-1] == 'Juggernaut-XI-v11':
        variant = None

    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype, variant=variant,)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=torch_dtype, variant=variant,)
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", torch_dtype=torch_dtype, variant=variant,)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype, variant=variant,)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=torch_dtype, variant=variant,)
    unet_new = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype, variant=variant,)
    
    pipe = pipeline_stable_diffusion_xl.StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
                      text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet_new,
        scheduler=scheduler,
    )
    pipe.to(device)

    return pipe, model_name

def get_generator(seed=42):
    return torch.Generator("cuda").manual_seed(seed)

def movement_gen_story_slide_windows_compel(id_prompt, frame_prompts, pipe,compel,device, window_length, seed,save_dir, verbose=True):  
    import os
   
    story_images = []

    
    for index, movement in enumerate(frame_prompts):

        gen_propmts = f'{id_prompt} {movement}'
        
        generate = torch.Generator().manual_seed(seed)

        if hasattr(pipe.text_encoder, 'reset'):
            pipe.text_encoder.reset()
        if hasattr(pipe.text_encoder_2, 'reset'):
            pipe.text_encoder_2.reset()
        prompt_embeds, pooled_prompt_embeds = compel(gen_propmts)

        embedding=get_final_embedding(pipe,id_prompt,movement,compel,device)


        logging.disable_progress_bar()
        images = pipe(
            generator=generate,
            prompt_embeds=embedding,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=50,
        ).images
        story_images.append(images[0])
        images[0].save(os.path.join(save_dir, f'{id_prompt} {movement}.jpg'))
    image_array_list = [np.array(pil_img) for pil_img in story_images]

    # Concatenate images horizontally
    story_image = np.concatenate(image_array_list, axis=1)
    story_image = Image.fromarray(story_image.astype(np.uint8))

    story_image.save(os.path.join(save_dir, f'story_image_{id_prompt}.jpg'))

    return story_images, story_image



   