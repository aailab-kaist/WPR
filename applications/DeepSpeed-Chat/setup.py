#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# setup.py: install script for deepspeed_chat
"""
to install deepspeed_chat and its dependencies for development work,
run this cmd from the root directory:
    pip install -e .
"""
#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import setuptools

setuptools.setup(
    name="deepspeed-chat",
    version="0.1",
    url="https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat",
    include_package_data=True,
    packages=setuptools.find_packages(include=["dschat"]),

    install_requires=[
        "deepspeed==0.13.1",
        "accelerate==1.0.1",
        "tensorboard",
        "transformers==4.41.0",
        "datasets==3.1.0",
        "huggingface-hub==0.30.1",
        "sentencepiece>=0.1.97",
        "protobuf==3.20.3",
        "numpy==1.24.4",
        "scipy",
        "pyext",
        "nvidia-ml-py",
    ],

    extras_require={
        "azureml": [
            "azure-ml-component",
            "azureml-core",
        ],
    },
)