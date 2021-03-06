
# Original Code
# https://github.com/PyTables/PyTables/blob/master/setup.py
# See https://github.com/graykode/matorage/blob/0.1.0/NOTICE
# modified by TaeHwan Jung(@graykode)

# Copyright 2020-present Tae Hwan Jung
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
from packaging import version
from setuptools import find_packages

logger = logging.getLogger(__name__)

def find_name(name="matorage"):
    """If "--name-with-python-version" is on the command line then
    append "-pyX.Y" to the base name"""
    if "--name-with-python-version" in sys.argv:
        name += "-py%i.%i" % (sys.version_info[0], sys.version_info[1])
        sys.argv.remove("--name-with-python-version")
    return name

def get_requirements(file='requirements.txt'):
    # Fetch the requisites
    with open(file) as f:
        requirements = f.read().splitlines()
    return requirements

def get_setuptools():
    setuptools_kwargs = {}

    setuptools_kwargs['zip_safe'] = False
    setuptools_kwargs['install_requires'] = get_requirements()

    # Detect packages automatically.
    setuptools_kwargs['packages'] = find_packages(exclude=['*.bench'])

    return setuptools_kwargs

def check_torch_tf_version():
    try:
        USE_TF = os.environ.get("USE_TF", "AUTO").upper()
        USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
        if USE_TORCH in ("1", "ON", "YES", "AUTO") and USE_TF not in ("1", "ON", "YES"):
            import torch
            import torchvision

            assert hasattr(torch, "__version__") and version.parse(
                torch.__version__
            ) >= version.parse("1.0.0")
            assert hasattr(torchvision, "__version__") and version.parse(
                torch.__version__
            ) >= version.parse("0.2.0")

            _torch_available = True  # pylint: disable=invalid-name
            logger.info("PyTorch version {} available.".format(torch.__version__))
            logger.info("PyTorch Vision version {} available.".format(torchvision.__version__))
        else:
            logger.info("Disabling PyTorch because USE_TF is set")
            _torch_available = False
    except ImportError:
        _torch_available = False  # pylint: disable=invalid-name

    try:
        USE_TF = os.environ.get("USE_TF", "AUTO").upper()
        USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

        if USE_TF in ("1", "ON", "YES", "AUTO") and USE_TORCH not in ("1", "ON", "YES"):
            import tensorflow as tf
            import tensorflow_io as tfio

            assert hasattr(tf, "__version__") and version.parse(
                tf.__version__
            ) >= version.parse("2.2")
            assert hasattr(tfio, "__version__") and version.parse(
                tfio.__version__
            ) >= version.parse("0.13")
            _tf_available = True  # pylint: disable=invalid-name
            logger.info("TensorFlow version {} available.".format(tf.__version__))
            logger.info("TensorFlow IO version {} available.".format(tfio.__version__))
        else:
            logger.info("Disabling Tensorflow because USE_TORCH is set")
            _tf_available = False
    except (ImportError, AssertionError):
        _tf_available = False  # pylint: disable=invalid-name
