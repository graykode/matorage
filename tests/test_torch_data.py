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
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from tests.test_data import DataTest

from matorage.data.config import DataConfig
from matorage.data.saver import DataSaver
from matorage.data.attribute import DataAttribute
from matorage.testing_utils import require_torch

class TorchDataset(Dataset):
    def __init__(self):
        self.x_data = torch.tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], dtype=torch.uint8)
        self.y_data = torch.tensor([0, 1], dtype=torch.uint8)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

@require_torch
class TorchDataTest(DataTest, unittest.TestCase):

    def test_torch_saver(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name='test_torch_saver',
            additional={
                "framework" : "pytorch"
            },
            attributes=[
                DataAttribute('image', 'uint8', (2, 2), itemsize=32),
                DataAttribute('target', 'uint8', (1), itemsize=32)
            ]
        )
        self.data_saver = DataSaver(
            config=self.data_config
        )

        _dataset = TorchDataset()
        loader = DataLoader(_dataset)
        for (image, target) in tqdm(loader):
            self.data_saver({
                'image': image,
                'target': target
            })
        self.data_saver.disconnect()

    def test_torch_loader(self):
        from matorage.torch import MTRDataset

        self.test_torch_saver()

        dataset = MTRDataset(config=self.data_config)
        loader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=True)

        for batch_idx, (image, target) in enumerate(tqdm(loader)):
            pass

    def test_torch_index(self):
        from matorage.torch import MTRDataset

        self.test_torch_saver()

        dataset = MTRDataset(config=self.data_config, index=True)

        assert torch.equal(dataset[0][0], torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8))
        assert dataset[0][1] == 0