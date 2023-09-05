# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from diffusion.models.imagen import build_imagen


@pytest.mark.parametrize('stage', [1, 2, 3])
def test_imagen_forward(stage):
    device='cuda'
    model = build_imagen(t5_name='google/t5-v1_1-small', stage=stage) # imagen unet wont run on cpu due to pixelshuffle
    
    batch_size = 1
    H = 8
    W = 8
    t5_small_emb_size = 512

    image = torch.randn(batch_size, 3, H, W, device=device)
    caption = torch.randint(low=0, high=128, size=(
        batch_size,
        t5_small_emb_size,
    ), dtype=torch.long, device=device)
    batch = {'image': image, 'captions': caption}
    model.to(device)
    output, target, timesteps = model(batch)
    assert output.shape == image.shape
    assert target.shape == image.shape



@pytest.mark.parametrize('guidance_scale', [0.0, 3.0])
@pytest.mark.parametrize('negative_prompt', [None, 'so cool'])
@pytest.mark.parametrize('stage', [1, 2, 3])
def test_imagen_generate(guidance_scale, negative_prompt, stage):
    # fp16 vae does not run on cpu
    device = 'cuda'
    model = build_imagen(t5_name='google/t5-v1_1-small', stage=stage) # imagen unet wont run on cpu due to pixelshuffle
    model.to(device)
    output = model.generate(
        prompt='a cool doge',
        negative_prompt=negative_prompt,
        num_inference_steps=1,
        num_images_per_prompt=1,
        height=8,
        width=8,
        guidance_scale=guidance_scale,
        lowres_conditional_images=torch.randn(size=(1, 3, 8, 8)),
        progress_bar=False
    )
    assert output.shape == (1, 3, 8, 8)
