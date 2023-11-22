# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from diffusion.models.models import stable_diffusion_2, stable_diffusion_xl


def test_model_forward():
    # fp16 vae does not run on cpu
    model = stable_diffusion_2(pretrained=False, fsdp=False, encode_latents_in_fp16=False)
    batch_size = 1
    H = 8
    W = 8
    image = torch.randn(batch_size, 3, H, W)
    latent = torch.randn(batch_size, 4, H // 8, W // 8)
    caption = torch.randint(low=0, high=128, size=(
        batch_size,
        77,
    ), dtype=torch.long)
    batch = {'image': image, 'captions': caption}
    output, target, _ = model(batch)  # model.forward generates the unet output noise or v_pred target.
    assert output.shape == latent.shape
    assert target.shape == latent.shape


@pytest.mark.parametrize('guidance_scale', [0.0, 3.0])
@pytest.mark.parametrize('negative_prompt', [None, 'so cool'])
def test_model_generate(guidance_scale, negative_prompt):
    # fp16 vae does not run on cpu
    model = stable_diffusion_2(pretrained=False, fsdp=False, encode_latents_in_fp16=False)
    output = model.generate(
        prompt='a cool doge',
        negative_prompt=negative_prompt,
        num_inference_steps=1,
        num_images_per_prompt=1,
        height=8,
        width=8,
        guidance_scale=guidance_scale,
        progress_bar=False,
    )
    assert output.shape == (1, 3, 8, 8)

@pytest.mark.parametrize('use_e5', [True, False])
def test_model_forward_sdxl(use_e5):
    model = stable_diffusion_xl(use_e5=use_e5, pretrained=False, fsdp=False, encode_latents_in_fp16=False, clip_qkv=None)
    batch_size = 1
    H = 32
    W = 32
    image = torch.randn(batch_size, 3, H, W)
    latent = torch.randn(batch_size, 4, H // 8, W // 8)
    caption = torch.randint(low=0, high=128, size=(
        batch_size,
        77,
    ), dtype=torch.long)
    randfloat = torch.randn(batch_size, 1)
    caption = torch.stack([caption, caption], dim=1)
    batch = {'image': image,
            'captions': caption,
            'cond_original_size': randfloat,
            'cond_crops_coords_top_left': randfloat,
            'cond_target_size': randfloat}
    output, target, _ = model(batch)  # model.forward generates the unet output noise or v_pred target.
    assert output.shape == latent.shape
    assert target.shape == latent.shape