# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from diffusion.models.models import stable_diffusion_2
from transformers import PretrainedConfig
from diffusers import UNet2DConditionModel


@pytest.mark.parametrize('unet_name', ['sd2', 'uvit-huge-ps2'])
def test_model_forward(unet_name):
    # fp16 vae does not run on cpu
    model = stable_diffusion_2(unet_name=unet_name, fsdp=False, encode_latents_in_fp16=False)
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

@pytest.mark.parametrize('unet_name', ['sd2', 'uvit-huge-ps2'])
def test_model_eval_forward(unet_name):
    # fp16 vae does not run on cpu
    model = stable_diffusion_2(unet_name=unet_name, fsdp=False, encode_latents_in_fp16=False)
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
    unet_out, targets, _, generated_images = model.eval_forward(batch)  # model.forward generates the unet output noise or v_pred target.
    assert unet_out.shape == latent.shape
    assert targets.shape == latent.shape
    assert generated_images == image.shape


@pytest.mark.parametrize('unet_name', ['sd2', 'uvit-huge-ps2'])
@pytest.mark.parametrize('text_encoder_name_or_path', ['stabilityai/stable-diffusion-2-base', 'e5-base'])
@pytest.mark.parametrize('guidance_scale', [0.0, 3.0])
@pytest.mark.parametrize('negative_prompt', [None, 'so cool'])
def test_model_generate(unet_name, guidance_scale, negative_prompt, text_encoder_name_or_path):
    # fp16 vae does not run on cpu
    model = stable_diffusion_2(unet_name=unet_name, text_encoder_name_or_path=text_encoder_name_or_path,
                               fsdp=False, encode_latents_in_fp16=False)
    output = model.generate(
        prompt='a cool doge',
        negative_prompt=negative_prompt,
        num_inference_steps=1,
        num_images_per_prompt=1,
        height=32,
        width=32,
        guidance_scale=guidance_scale,
        progress_bar=False,
    )
    assert output.shape == (1, 3, 32, 32)

def test_unet_e5():
    config = PretrainedConfig.get_config_dict(pretrained_model_name_or_path='stabilityai/stable-diffusion-2-base',
                                                subfolder='unet')
    unet = UNet2DConditionModel(**config[0])
    mock_e5_output = torch.randn(2, 6, 1024) # bs, num tokens, embedding_dim conditioning
    mock_timesteps = torch.randint(0, 1000, (2,)) # randn 1-1000, batch_Size
    mock_noised_latents = torch.randn(2, 4, 8, 8) # bs, 8, 8, 4
    output = unet(mock_noised_latents, mock_timesteps, mock_e5_output)['sample']
    assert output.shape == (2, 4, 8, 8)

