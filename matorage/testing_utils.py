# Original Code
# https://github.com/huggingface/transformers/blob/master/src/transformers/testing_utils.py
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

import unittest

from matorage.utils import _tf_available, _torch_available


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not _torch_available:
        test_case = unittest.skip("test requires PyTorch")(test_case)
    return test_case


def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow.

    These tests are skipped when TensorFlow isn't installed.

    """
    if not _tf_available:
        test_case = unittest.skip("test requires TensorFlow")(test_case)
    return test_case
