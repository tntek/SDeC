import os
import torch
import random
import diffusers
import torch.utils
import unet.utils as utils
import argparse
from datetime import datetime

from compel import Compel, ReturnedEmbeddingsType
diffusers.utils.logging.set_verbosity_error()


def generate_images(pipe, compel,id_prompt, frame_prompt_list, device,save_dir, window_length, seed,verbose=True):
    images, story_image = utils.movement_gen_story_slide_windows_compel(id_prompt,frame_prompt_list,pipe,compel,device,window_length,seed,save_dir=save_dir)

    return images, story_image


def main(device, model_path, save_dir, id_prompt, frame_prompt_list, precision, seed, window_length):
    pipe, _ = utils.load_pipe_from_path(model_path, device, torch.float16 if precision == "fp16" else torch.float32, precision)
    
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        
        requires_pooled=[False, True]
)

    images, story_image = generate_images(pipe,compel,id_prompt, frame_prompt_list, device,save_dir, window_length, seed)

    return images, story_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a specific device.")
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for computation (e.g., cuda:0, cpu)')
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', help='Path to the model')

    parser.add_argument('--project_base_path', type=str, default='.', help='Path to save the generated images')
    parser.add_argument('--id_prompt', type=str, default="A leprechaun", help='Initial prompt for image generation')
    parser.add_argument('--frame_prompt_list', type=str, nargs='+', default=[
     "in a clover field",
    "mending a shoe",
   "inside a hollow oak tree",
   "sharing a pint",
   "at a village fair",
   "hiding a pot of gold"
    ], help='List of frame prompts')

    parser.add_argument('--precision', type=str, choices=["fp16", "fp32"], default="fp16", help='Model precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--window_length', type=int, default=10, help='Window length for story generation')
    parser.add_argument('--save_padding', type=str, default='test', help='Padding for save directory')
    parser.add_argument('--random_seed', action='store_true', help='Use random seed')
    parser.add_argument('--json_path', type=str,)
    
    args = parser.parse_args()
    if args.random_seed:
        args.seed = random.randint(0, 1000000)

    current_time = datetime.now().strftime("%Y%m%d%H")
    current_time_ = datetime.now().strftime("%M%S")
    save_dir = os.path.join(args.project_base_path, f'result/{current_time}/{current_time_}_{args.save_padding}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    
   
    if args.json_path is None:
        main(args.device, args.model_path, save_dir, args.id_prompt, args.frame_prompt_list, args.precision, args.seed, args.window_length)
    else:
        import json
        with open(args.json_path, "r") as file:
            data = json.load(file)

        combinations = data["combinations"]

        for combo in combinations:
            main(args.device, args.model_path, save_dir, combo['id_prompt'], combo['frame_prompt_list'], args.precision, args.seed, args.window_length)


