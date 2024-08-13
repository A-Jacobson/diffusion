
import torch
from torch import nn
from typing import Tuple
from composer import ComposerModel
import open_clip
import numpy as np
from open_clip.loss import SigLipLoss
from composer.utils import dist
from composer.utils.file_helpers import get_file


class CLIP(ComposerModel):
    def __init__(self, model_name: str, pretrained:str=None, loss=None,
                 force_quick_gelu: bool = False,
                 force_custom_text: bool = False,
                 force_patch_dropout: float | None = None,
                 force_image_size: int | Tuple[int, int] | None = None,
                **model_kwargs):
        super().__init__()
        self.config = {}
        self.config['model_name'] = model_name
        self.config['pretrained'] = pretrained
        self.config['force_quick_gelu'] = force_quick_gelu
        self.config['force_custom_text'] = force_custom_text
        self.config['force_patch_dropout'] = force_patch_dropout
        self.config['force_image_size'] = force_image_size
        self.loss = loss
        if isinstance(loss, SigLipLoss):
            model_kwargs["init_logit_scale"] = np.log(10)  # different from CLIP
            model_kwargs["init_logit_bias"] = -10
        self.model, self.transforms_train, self.transforms_val = (
            open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                force_quick_gelu=force_quick_gelu,
                force_custom_text=force_custom_text,
                force_patch_dropout=force_patch_dropout,
                force_image_size=force_image_size,
                output_dict=True,
                **model_kwargs,
            )
        )

        self.tokenizer = open_clip.get_tokenizer(model_name=model_name)
        self.loss_function = loss

    @property
    def text_encoder_dim(self):
        768

    def forward(self, batch):
        return self.model(batch["image"], batch["text"]) 

    def loss(self, outputs, batch):
        return self.loss_function(**outputs, output_dict=False)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def encode_text(self, text, normalize:bool=False):
        return self.model.encode_text(text, normalize=normalize)

# class CLIPTextEncoder():

#     def __init__(self, transformer):

#     def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
#         if pool_type == 'first':
#             pooled, tokens = x[:, 0], x[:, 1:]
#         elif pool_type == 'last':
#             pooled, tokens = x[:, -1], x[:, :-1]
#         elif pool_type == 'argmax':
#             # take features from the eot embedding (eot_token is the highest number in each sequence)
#             assert text is not None
#             pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
#         else:
#             pooled = tokens = x

#         return pooled, tokens

#     def encode_text(self, text, normalize: bool = False):
#         cast_dtype = self.transformer.get_cast_dtype()

#         x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

#         x = x + self.positional_embedding.to(cast_dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x, attn_mask=self.attn_mask)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
#         x, _ = text_global_pool(x, text, self.text_pool_type)
#         if self.text_projection is not None:
#             if isinstance(self.text_projection, nn.Linear):
#                 x = self.text_projection(x)
#             else:
#                 x = x @ self.text_projection

#         return F.normalize(x, dim=-1) if normalize else x



def load_clip_text_encoder(load_path: str,
                     local_path: str = '/tmp/clip_weights.pt',
                     torch_dtype=None):
    """Function to load an clip from a composer checkpoint without the loss weights.

    Will also load the latent statistics if the statistics tracking callback was used.

    Args:
        load_path (str): Path to the composer checkpoint. Can be a local folder, URL, or composer object store.
        local_path (str): Local path to save the autoencoder weights to. Default: `/tmp/autoencoder_weights.pt`.
        torch_dtype (torch.dtype): Torch dtype to cast the weights to. Default: `None`.

    Returns:
        text_encoder (CLIP): AutoEncoder model with weights loaded from the checkpoint.
        tokenizer (Dict[str, Union[list, float]]): Dictionary of latent statistics if present, else `None`.
    """
    # Download the autoencoder weights and init them
    if dist.get_local_rank() == 0:
        get_file(path=load_path, destination=local_path, overwrite=True)
    with dist.local_rank_zero_download_and_wait(local_path):
        # Load the autoencoder weights from the state dict
        state_dict = torch.load(local_path, map_location='cpu')
    # Get the config from the state dict and init the model using it
    config = state_dict['state']['model']['config']
    model = CLIP(**config)
    # Need to clean up the state dict to remove loss and metrics.
    cleaned_state_dict = {}
    for key in list(state_dict['state']['model'].keys()):
        if key.split('.')[0] == 'model':
            cleaned_key = '.'.join(key.split('.')[1:])
            cleaned_state_dict[cleaned_key] = state_dict['state']['model'][key]
    # Load the cleaned state dict into the model
    model.load_state_dict(cleaned_state_dict, strict=True)
    
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    
    tokenizer = model.tokenizer
    text_encoder = model.model.transformer
    return text_encoder, tokenizer
