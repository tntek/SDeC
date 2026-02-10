import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import random
import numpy as np
import os
from datetime import datetime
import time
from compel import Compel, ReturnedEmbeddingsType


def tokenize_with_sdxl(prompt, tokenizer, max_len=77, pad=True):
    tokens = tokenizer(
        prompt,
        padding="max_length" if pad else False,
        max_length=max_len,
        truncation=True,
        return_tensors="pt"
    )
    token_ids = tokens["input_ids"][0]
    words = tokenizer.convert_ids_to_tokens(token_ids)
    return token_ids, words

def embed_with_sdxl(prompt, pipe, device):
    with torch.no_grad():
        outputs = pipe.encode_prompt(prompt=prompt)[0].float().to(device)
        return outputs  # [1, 77, 2048]

def find_subsequence_index(sub_ids, full_ids):
    for i in range(len(full_ids) - len(sub_ids) + 1):
        if torch.equal(full_ids[i:i+len(sub_ids)], sub_ids):
            return list(range(i, i + len(sub_ids)))
    return []


class TrainableSVDAmplifier(nn.Module):
    def __init__(self, diff):
        super().__init__()
        # SVD分解时转为float32
        U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)
        self.register_buffer('U', U)
        self.register_buffer('Vh', Vh)
        self.original_S=S.clone()
        self.S = nn.Parameter(S.clone())

    def forward(self):
        S_full = self.S

        amplified_diff = (self.U * S_full.unsqueeze(0)) @ self.Vh
        return amplified_diff


def train_svd_and_get_embedding(pipe,full_prompt, id_prompt, frame_prompt, compel, device, num_steps=200, lr=1e-1):
    full_ids, full_tokens = tokenize_with_sdxl(full_prompt, pipe.tokenizer, pad=True)
    id_ids, id_tokens = tokenize_with_sdxl(id_prompt, pipe.tokenizer, pad=False)
    id_ids = id_ids[:-1]
    id_token_len = pipe.tokenizer(id_prompt, return_tensors="pt")["input_ids"].shape[1]  # 包含</s>

    frame_ids, frame_tokens = tokenize_with_sdxl(frame_prompt, pipe.tokenizer, pad=False)
    frame_ids = frame_ids[:-1]
    frame_token_idxs = find_subsequence_index(frame_ids, full_ids)

    text_embed, _ = compel(full_prompt)
    empty_embed, _ = compel("")
    frame_embed, _ = compel(frame_prompt)

    text_embed = text_embed.float()
    empty_embed = empty_embed.float()
    frame_embed = frame_embed.float()

    id_token_len = pipe.tokenizer(id_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
    frame_token_len = pipe.tokenizer(frame_prompt, return_tensors="pt")["input_ids"].shape[1] - 1

    max_len = max(id_token_len, frame_token_len)
    min_len = min(id_token_len, frame_token_len)
    e = text_embed[0, :id_token_len]        # id区域
  

    amplifier = TrainableSVDAmplifier(e).to(device)
    optimizer = torch.optim.Adam(amplifier.parameters(), lr=lr)
    for step in tqdm(range(num_steps), desc="[Train]"):
        optimizer.zero_grad()
        amplified_diff = amplifier()
        amplified_embed = amplified_diff

        embedding = text_embed.clone()
        embedding[0, :id_token_len] = amplified_embed

        if step < num_steps //4:
            loss = 100*F.mse_loss(
                embedding[0, 1:min_len],
                frame_embed[0, 1:min_len]
            )

            loss_id = torch.tensor(0.0, device=embedding.device)
        else:
            loss_align = F.mse_loss(
                    embedding[0, 1:min_len],
                    frame_embed[0, 1:min_len]
                )
            loss_id = F.mse_loss(amplified_embed, e)
            loss = loss_align + 10*loss_id
   
        loss.backward()

        optimizer.step()

    embedding = text_embed.clone()
    embedding[0, :id_token_len] =  amplifier()
    S_diff = amplifier.S - amplifier.original_S

    return S_diff

def svd_amplify_tokens_groupwise(
    pipe,
    compel,
    id_prompt,
    full_prompt,
    S_diff,
    max_amp=2,
    eps=1e-6
) -> torch.Tensor:
   
    id_token_len = pipe.tokenizer(id_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
    text_embed, _ = compel(full_prompt)
    text_embed = text_embed.clone()
    e =text_embed[0,:id_token_len]
    e=e.float()
    U, S, Vh = torch.linalg.svd(e ,full_matrices=False)  

    S_new = S.clone()

    delta = torch.abs(S_diff) + eps       
    inv_delta = 1.0 / delta                        
    inv_delta_norm = inv_delta / inv_delta.max()  
    
    alpha_vec = 1.0 + inv_delta_norm * (max_amp - 1.0)  

    S_new = S_new * alpha_vec

    amplified_diff = (U * S_new.unsqueeze(0)) @ Vh  
    amplified_embed = amplified_diff      

    text_embed[0, :id_token_len] = amplified_embed
    return text_embed


def get_final_embedding(pipe,id_prompt,frame_prompt,compel,device):
    full_prompt = id_prompt + " " + frame_prompt
    S_diff = train_svd_and_get_embedding(pipe,full_prompt, id_prompt, frame_prompt, compel, device, num_steps=100, lr=10)
    embedding=svd_amplify_tokens_groupwise(pipe,compel,id_prompt,full_prompt,S_diff)
    return embedding

