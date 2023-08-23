# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion models in pixel space."""

from typing import Literal, List, Optional
import copy

import torch
from composer.devices import DeviceGPU

from composer.models import ComposerModel
from torchmetrics import MeanSquaredError, Metric
from tqdm.auto import tqdm
from diffusers import DDIMScheduler, DDPMScheduler
from imagen_pytorch import BaseUnet64, SRUnet256, SRUnet1024
from transformers import T5Tokenizer, T5EncoderModel
from resize_right import resize


def imagen(t5_name: str = 'google/t5-v1_1-base', unet='base', prediction_type='epsilon', stage=1):
    """Discrete pixel diffusion training setup.

    Args:
        t5_name (str, optional): Name of the t5 model to load. 
            see https://huggingface.co/docs/transformers/model_doc/t5v1.1 for mode list.
            Defaults to 'google/t5-v1_1-base'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'.
            Defaults to 'epsilon'.
    """
    # Create a pixel space unet
    # Get the T5 text encoder and tokenizer:
    text_encoder = T5EncoderModel.from_pretrained(t5_name)
    tokenizer = T5Tokenizer.from_pretrained(t5_name)
    embed_dim = text_encoder.config.d_model # pull embedding dim from hf config
    lowres_noise_scheduler = None


    unets =  {1: BaseUnet64(embed_dim=embed_dim), 
              2: SRUnet256(embed_dim=embed_dim, lowres_cond=True),
              3: SRUnet1024(embed_dim=embed_dim, lowres_cond=True)}
    
    noise_schedulers = {1: DDPMScheduler(num_train_timesteps=1000,
                                    beta_schedule='squaredcos_cap_v2',
                                    prediction_type=prediction_type,
                                    thresholding=True),
                        2: DDPMScheduler(num_train_timesteps=1000,
                                    beta_schedule='squaredcos_cap_v2',
                                    prediction_type=prediction_type,
                                    thresholding=True),
                        3: DDPMScheduler(beta_schedule='linear',
                                    beta_start=1e-4,
                                    beta_end=0.02,
                                    thresholding=True,
                                    prediction_type=prediction_type)
                        }
    
    unet = unets[stage]
    noise_scheduler = noise_schedulers[stage]
    inference_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                        beta_start=1e-4,
                                        beta_end=0.02,
                                        beta_schedule='linear',
                                        prediction_type=prediction_type,
                                        thresholding=True)
    
    if stage > 1: 
        is_superres = True
        # create noise scheduler for noise augmentation

        
    # Create the pixel space diffusion model
    model = Imagen(unet,
                    text_encoder,
                    tokenizer,
                    noise_scheduler,
                    is_superres=is_superres,
                    inference_scheduler=inference_scheduler,
                    prediction_type=prediction_type,
                    train_metrics=[MeanSquaredError()],
                    val_metrics=[MeanSquaredError()])

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
    return model



class Imagen(ComposerModel):
    """Pixel space diffusion model.

    Args:
        model (torch.nn.Module): Model to use as the core diffusion model.
        text_encoder (torch.nn.Module): HuggingFace CLIP or LLM text enoder.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for
            text_encoder. For a `CLIPTextModel` this will be the
            `CLIPTokenizer` from HuggingFace transformers.
        scheduler (diffusers.SchedulerMixin or diffusion.Scheduler): Scheduler to use for diffusion during training.
        inference_scheduler (diffusers.SchedulerMixin or diffusion.schedulers.ContinuousTimeScheduler): Scheduler to
            use for diffusion during inference. If `None`, defaults to `scheduler`.
        continuous_time (bool): Whether to use continuous time diffusion. Default: `False`.
        input_key (str):  The name of the inputs in the dataloader batch. Default: `image`.
        conditioning_key (str): The name of the conditioning inputs in the dataloader batch. Default: `captions`.
        prediction_type (str): The type of prediction to use. Must be one of 'sample', 'epsilon', or 'v_prediction'.
            Default: `epsilon`.
        train_metrics (List[torchmetrics.Metric]): List of metrics to use during training.
            Default: `[torchmetrics.MeanSquaredError]`.
        val_metrics (List[torchmetrics.Metric]): List of metrics to use during validation.
            Default: `[torchmetrics.MeanSquaredError]`.
        val_seed (int): Random seed to use for validation. Default: `1138`.
    """

    def __init__(self,
                 model,
                 text_encoder,
                 tokenizer,
                 scheduler,
                 is_superres:bool=False,
                 noise_augmentation_level:float = 0.2,
                 inference_scheduler=None,
                 continuous_time: bool = False,
                 input_key: str = 'image',
                 conditioning_key: str = 'captions',
                 prediction_type: str = 'epsilon',
                 train_metrics: Optional[List] = None,
                 val_metrics: Optional[List] = None,
                 val_seed: int = 1138):
        super().__init__()
        self.model = model
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.is_superres = is_superres
        self.inference_scheduler = inference_scheduler if inference_scheduler is not None else scheduler
        self.noise_augmentation_level = noise_augmentation_level
        self.continuous_time = continuous_time
        self.input_key = input_key
        self.conditioning_key = conditioning_key
        if prediction_type not in ['sample', 'epsilon', 'v_prediction']:
            raise ValueError(f'prediction type must be one of sample, epsilon, or v_prediction. Got {prediction_type}')
        self.prediction_type = prediction_type
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.val_seed = val_seed

        # freeze text_encoder training
        self.text_encoder.requires_grad_(False)

    def forward(self, batch, generator=None):
        inputs, conditioning = batch[self.input_key], batch[self.conditioning_key]
        batch_size = inputs.shape[0]
        # Encode the conditioning
        conditioning = self.text_encoder(conditioning)[0]
        # Sample the diffusion timesteps, either discrete or continuous
        if self.continuous_time:
            timesteps = self.scheduler.t_max * torch.rand(batch_size, device=inputs.device, generator=generator)
        else:
            timesteps = torch.randint(0,
                                      len(self.scheduler), (batch_size,),
                                      device=inputs.device,
                                      generator=generator)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(inputs)
        noised_inputs = self.scheduler.add_noise(inputs, noise, timesteps)
        # Generate the targets
        if self.prediction_type == 'epsilon':
            targets = noise
        elif self.prediction_type == 'sample':
            targets = inputs
        elif self.prediction_type == 'v_prediction':
            targets = self.scheduler.get_velocity(inputs, noise, timesteps)
        else:
            raise ValueError(
                f'prediction type must be one of sample, epsilon, or v_prediction. Got {self.prediction_type}')
        
        # Forward through the model
        if self.is_superres:
            # resize inputs for superres 64 -> 256 for stage 2, 256 -> 1024 for stage 3
            lowres_conditional_images = resize_image_to(inputs, inputs.shape[-1]*4, pad_mode='reflect')

            # add noise augmentation here, we add noise at a fixed timestep in the scheduler. 
            # for a scheduler with 1k steps and a noise level of 0.2 we would set the timestep to 200.
            lowres_aug_timesteps = torch.full((batch_size,), 
                                              int(len(self.scheduler) * self.noise_augmentation_level), 
                                             device=inputs.device, dtype=torch.long)
            lowres_noise = torch.randn_like(lowres_conditional_images) 
            self.scheduler.add_noise(lowres_conditional_images, lowres_noise, lowres_aug_timesteps)

            # pass through superres unet
            return self.model(x=noised_inputs, time=timesteps, text_embeds=conditioning,
                             lowres_cond_img=lowres_conditional_images, lowres_aug_time=lowres_aug_timesteps), targets, timesteps

        else:
            return self.model(x=noised_inputs, time=timesteps, text_embeds=conditioning), targets, timesteps
        # resize inputs for superres + add noise augmentation here

    def loss(self, outputs, batch):
        return torch.nn.functional.mse_loss(outputs[0], outputs[1])

    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs
        # Create rng and fix the seed for eval
        generator = torch.Generator(device=self.model.device)
        generator = generator.manual_seed(self.val_seed)
        # Get model outputs
        model_out, targets, timesteps = self.forward(batch, generator=generator)
        return model_out, targets, timesteps

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, list):
            metrics_dict = {metric.__class__.__name__: metric for metric in metrics}
        elif isinstance(metrics, dict):
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric
        else:
            raise NotImplementedError(f'Metrics type {metrics.__class__.__name__} not supported.')
        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, MeanSquaredError):
            metric.update(outputs[0], outputs[1])
        else:
            raise NotImplementedError(f'Metric {metric.__class__.__name__} not implemented.')

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[list] = None,
        negative_prompt: Optional[list] = None,
        tokenized_prompts: Optional[torch.LongTensor] = None,
        tokenized_negative_prompts: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        height: int = 64,
        width: int = 64,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 3.0,
        num_images_per_prompt: Optional[int] = 1,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = True,
    ):
        """Generates image from noise.

        Performs the backward diffusion process, each inference step takes
        one forward pass through the unet.

        Args:
            prompt (str or List[str]): The prompt or prompts to guide the image generation.
            negative_prompt (str or List[str]): The prompt or prompts to guide the
                image generation away from. Ignored when not using guidance
                (i.e., ignored if guidance_scale is less than 1).
                Must be the same length as list of prompts. Default: `None`.
            tokenized_prompts (torch.LongTensor): Optionally pass pre-tokenized prompts instead
                of string prompts. Default: `None`.
            tokenized_negative_prompts (torch.LongTensor): Optionally pass pre-tokenized negative
                prompts instead of string prompts. Default: `None`.
            prompt_embeds (torch.FloatTensor): Optionally pass pre-tokenized prompts instead
                of string prompts. If both prompt and prompt_embeds
                are passed, prompt_embeds will be used. Default: `None`.
            negative_prompt_embeds (torch.FloatTensor): Optionally pass pre-embedded negative
                prompts instead of string negative prompts. If both negative_prompt and
                negative_prompt_embeds are passed, prompt_embeds will be used.  Default: `None`.
            height (int, optional): The height in pixels of the generated image.
                Default: `64`.
            width (int, optional): The width in pixels of the generated image.
                Default: `64`.
            num_inference_steps (int): The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense
                of slower inference. Default: `50`.
            guidance_scale (float): Guidance scale as defined in
                Classifier-Free Diffusion Guidance. guidance_scale is defined as w of equation
                2. of Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
                Higher guidance scale encourages to generate images that are closely linked
                to the text prompt, usually at the expense of lower image quality.
                Default: `3.0`.
            num_images_per_prompt (int): The number of images to generate per prompt.
                 Default: `1`.
            progress_bar (bool): Wether to use the tqdm progress bar during generation.
                Default: `True`.
            seed (int): Random seed to use for generation. Set a seed for reproducible generation.
                Default: `None`.
        """
        # Create rng for the generation
        device = self.model.device
        rng_generator = torch.Generator(device=device)
        if seed:
            rng_generator = rng_generator.manual_seed(seed)  # type: ignore

        do_classifier_free_guidance = guidance_scale > 1.0  # type: ignore

        text_embeddings = self._prepare_text_embeddings(prompt, tokenized_prompts, prompt_embeds, num_images_per_prompt)
        batch_size = len(text_embeddings)  # len prompts * num_images_per_prompt
        # classifier free guidance + negative prompts
        # negative prompt is given in place of the unconditional input in classifier free guidance
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ([''] * (batch_size // num_images_per_prompt))  # type: ignore
            unconditional_embeddings = self._prepare_text_embeddings(negative_prompt, tokenized_negative_prompts,
                                                                     negative_prompt_embeds, num_images_per_prompt)
            # concat uncond + prompt
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])

        # prepare for diffusion generation process
        images = torch.randn((batch_size, 3, height, width), device=device, generator=rng_generator)

        self.inference_scheduler.set_timesteps(num_inference_steps)
        # scale the initial noise by the standard deviation required by the scheduler
        images = images * self.inference_scheduler.init_noise_sigma

        # backward diffusion process
        for t in tqdm(self.inference_scheduler.timesteps, disable=not progress_bar):
            if do_classifier_free_guidance:
                model_input = torch.cat([images] * 2)
            else:
                model_input = images

            model_input = self.inference_scheduler.scale_model_input(model_input, t)
            # get model's predicted output
            model_output = self.model(model_input, t, encoder_hidden_states=text_embeddings).sample

            if do_classifier_free_guidance:
                # perform guidance. Note this is only techincally correct for prediction_type 'epsilon'
                pred_uncond, pred_text = model_output.chunk(2)
                model_output = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            images = self.inference_scheduler.step(model_output, t, images, generator=rng_generator)['prev_sample']

        # Rescale to (0, 1)
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.detach()  # (batch*num_images_per_prompt, channel, h, w)

    def _prepare_text_embeddings(self, prompt, tokenized_prompts, prompt_embeds, num_images_per_prompt):
        """Tokenizes and embeds prompts if needed, then duplicates embeddings to support multiple generations per prompt."""
        device = self.text_encoder.device
        if prompt_embeds is None:
            if tokenized_prompts is None:
                tokenized_prompts = self.tokenizer(prompt,
                                                   padding='max_length',
                                                   max_length=self.tokenizer.model_max_length,
                                                   truncation=True,
                                                   return_tensors='pt').input_ids
            text_embeddings = self.text_encoder(tokenized_prompts.to(device))[0]  # type: ignore
        else:
            text_embeddings = prompt_embeds

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)  # type: ignore
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return text_embeddings


# from https://github.com/AssemblyAI-Examples/MinImagen/blob/main/minimagen/helpers.py#L6
def exists(val) -> bool:
    """
    Checks to see if a value is not `None`
    """
    return val is not None

def resize_image_to(image: torch.tensor,
                    target_image_size: int,
                    clamp_range: tuple = None,
                    pad_mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'reflect'
                    ) -> torch.tensor:
    """
    Resizes image to desired size.

    :param image: Images to resize. Shape (b, c, s, s)
    :param target_image_size: Edge length to resize to.
    :param clamp_range: Range to clamp values to. Tuple of length 2.
    :param pad_mode: `constant`, `edge`, `reflect`, `symmetric`.
        See [TorchVision documentation](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.pad.html) for additional details
    :return: Resized image. Shape (b, c, target_image_size, target_image_size)
    """
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    scale_factors = target_image_size / orig_image_size
    out = resize(image, scale_factors=scale_factors, pad_mode=pad_mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out