# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/facebookresearch/segment-anything

from setuptools import find_packages, setup
from setuptools.command.install import install
from tempfile import NamedTemporaryFile
from os import unlink
import subprocess
import sys

class CustomInstall(install):
    def run(self):
        with NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write('numpy>=2.0.0\n')
            f.write('nvidia-cublas-cu12>=12.4.5.8\n')
            f.write('nvidia-cuda-cupti-cu12>=12.4.127\n')
            f.write('nvidia-cuda-nvrtc-cu12>=12.4.127\n')
            f.write('nvidia-cuda-runtime-cu12>=12.4.127\n')
            f.write('nvidia-cudnn-cu12>=9.1.0.70\n')
            f.write('nvidia-cufft-cu12>=11.2.1.3\n')
            f.write('nvidia-curand-cu12>=10.3.5.147\n')
            constraints_file = f.name
        try:
            subprocess.check_call([
                sys.executable, 
                '-m', 
                'pip', 
                'install', 
                'monai', 
                '--constraint', 
                constraints_file
            ])
        finally:
            unlink(constraints_file)
        install.run(self)

setup(
    name="medsam",
    version="0.0.1",
    author="Jun Ma",
    python_requires=">=3.9",
    install_requires=["matplotlib", "scikit-image", "SimpleITK>=2.2.1", "nibabel", "tqdm", "scipy", "ipympl", "opencv-python", "jupyterlab", "ipywidgets"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
    cmdclass={
        'install': CustomInstall,
    },
)
