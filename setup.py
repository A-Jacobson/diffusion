# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion package setup."""

from setuptools import find_packages, setup

install_requires = [
    'composer>=0.16.0',
    'mosaicml-streaming>=0.4.0,<1.0',
    'hydra-core>=1.2',
    'hydra-colorlog>=1.1.0',
    'diffusers[torch]==0.16.0',
    'transformers[torch]==4.29.2',
    'wandb==0.15.4',
    'xformers==0.0.16',
    'triton==2.0.0',
    'torchmetrics[image]==0.11.3',
    'clean-fid',
    'clip@git+https://github.com/openai/CLIP.git'
]
extras_require = {}


extras_require['imagen'] = {
    'resize_right==0.0.2',
    'imagen_pytorch==1.25.6',
    'einops_exts==0.0.4'
}

extras_require['dev'] = {
    'pre-commit>=2.18.1,<3',
    'pytest==7.3.0',
    'coverage[toml]==7.2.2',
    'pyarrow==11.0.0',
}

extras_require['all'] = set(dep for deps in extras_require.values() for dep in deps)

setup(
    name='diffusion',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
