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

import torch
import unittest
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from tests.test_model import ModelTest

from matorage.model.config import ModelConfig
from matorage.model.torch.manager import ModelManager
from matorage.testing_utils import require_torch

@require_torch
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@require_torch
class TorchModelTest(ModelTest, unittest.TestCase):

    def test_torch_saver(self, save_to_json_file=False):
        self.model_config = ModelConfig(
            **self.storage_config,
            model_name='test_torch_saver',
            additional={
                "framework" : "pytorch"
            }
        )
        if save_to_json_file:
            self.model_config_file = 'model_config_file.json'
            self.model_config.to_json_file(self.model_config_file)

        self.model_manager = ModelManager(
            config=self.model_config
        )

        model = Model()
        self.model_manager.save({
            "step" : 0
        }, model)

    def test_saver_from_json_file(self):

        self.test_torch_saver(save_to_json_file=True)

        self.model_config = None
        self.model_manager = None

        self.model_config = ModelConfig.from_json_file(self.model_config_file)

        self.model_manager = ModelManager(
            config=self.model_config
        )

        model = Model()
        self.model_manager.save({
            "step": 0
        }, model)