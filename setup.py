import os
import sys

from setuptools import find_packages, setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aimev2"))

# find the version
version_file = os.path.join(
    os.path.dirname(__file__), "aimev2", "configs", "version", "default.yaml"
)
with open(version_file, "r") as f:
    version_text = f.read()
__version__ = version_text.split(":")[-1].strip()

assert sys.version_info.major == 3, (
    "This repo is designed to work with Python 3."
    + "Please install it before proceeding."
)

setup(
    name="aimev2",
    author="Xingyuan Zhang",
    author_email="xingyuan.zhang@argmax.ai",
    packages=find_packages(),
    version=__version__,
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "einops",
        "dm_control",
        "mujoco",
        "gym",
        "matplotlib",
        "tensorboard",
        "tqdm",
        "moviepy",
        "imageio==2.27",
        "hydra-core",
        "timm==0.6.12",
    ],
    url="https://github.com/argmax-ai/aime-v2",
)
