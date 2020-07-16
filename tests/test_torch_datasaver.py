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
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tests.test_data import DataTest

from matorage.data.config import DataConfig
from matorage.data.saver import DataSaver
from matorage.data.attribute import DataAttribute
from matorage.testing_utils import require_torch

@require_torch
class TorchDataSaverTest(DataTest, unittest.TestCase):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def test_mnist_saver(self):
        dataset = datasets.MNIST(
            '/tmp/data/mnist',
            train=True,
            download=True,
            transform=self.transform
        )

        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name='test_mnist_saver',
            attributes=[
                DataAttribute('image', 'uint8', (28, 28), itemsize=32),
                DataAttribute('target', 'uint8', (1), itemsize=32)
            ]
        )
        self.data_saver = DataSaver(
            config=self.data_config
        )

        train_loader = DataLoader(dataset, batch_size=60, num_workers=8)
        for (image, target) in tqdm(train_loader):
            self.data_saver({
                'image': image,
                'target': target
            })
        self.data_saver.disconnect()