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
import json
import torch
import unittest
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from tests.test_data import DataTest

from matorage.data.config import DataConfig
from matorage.data.saver import DataSaver
from matorage.data.attribute import DataAttribute
from matorage.testing_utils import require_torch


@require_torch
class TorchDataTest(DataTest, unittest.TestCase):
    def test_torch_saver(self, data_config=None, save_to_json_file=False):
        if data_config is None:
            self.data_config = DataConfig(
                **self.storage_config,
                dataset_name="test_torch_saver",
                additional={"framework": "pytorch"},
                attributes=[
                    DataAttribute("image", "uint8", (2, 2), itemsize=32),
                    DataAttribute("target", "uint8", (1), itemsize=32),
                ]
            )
        else:
            self.data_config = data_config

        if save_to_json_file:
            self.data_config_file = "data_config_file.json"
            self.data_config.to_json_file(self.data_config_file)

        self.data_saver = DataSaver(config=self.data_config)

        self.data_saver(
            {
                "image": np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                "target": np.asarray([0, 1]),
            }
        )
        self.data_saver.disconnect()

    def test_torch_loader(self):
        from matorage.torch import Dataset

        self.test_torch_saver()

        self.dataset = Dataset(config=self.data_config)
        loader = DataLoader(self.dataset, batch_size=64, num_workers=8, shuffle=True)

        for batch_idx, (image, target) in enumerate(tqdm(loader)):
            pass

    def test_torch_loader_with_compressor(self):
        from matorage.torch import Dataset

        data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_torch_loader_with_compressor",
            additional={"framework": "pytorch"},
            compressor={"complevel": 4, "complib": "zlib"},
            attributes=[
                DataAttribute("image", "uint8", (2, 2), itemsize=32),
                DataAttribute("target", "uint8", (1), itemsize=32),
            ]
        )

        self.test_torch_saver(data_config=data_config)

        self.dataset = Dataset(config=data_config)
        loader = DataLoader(self.dataset, batch_size=64, num_workers=8, shuffle=True)

        for batch_idx, (image, target) in enumerate(tqdm(loader)):
            pass

    def test_torch_index(self):
        from matorage.torch import Dataset

        self.test_torch_saver()

        dataset = Dataset(config=self.data_config, index=True)

        assert torch.equal(
            dataset[0][0], torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
        )
        assert torch.equal(dataset[0][1], torch.tensor([0], dtype=torch.uint8))

    def test_torch_index_with_compressor(self):
        from matorage.torch import Dataset

        data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_torch_index_with_compressor",
            additional={"framework": "pytorch"},
            compressor={"complevel": 4, "complib": "zlib"},
            attributes=[
                DataAttribute("image", "uint8", (2, 2), itemsize=32),
                DataAttribute("target", "uint8", (1), itemsize=32),
            ]
        )

        self.test_torch_saver(data_config=data_config)

        dataset = Dataset(config=self.data_config, index=True)

        assert torch.equal(
            dataset[0][0], torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
        )
        assert torch.equal(dataset[0][1], torch.tensor([0], dtype=torch.uint8))

    def test_saver_from_json_file(self):

        self.test_torch_saver(save_to_json_file=True)

        self.data_config = None
        self.data_saver = None

        self.data_config = DataConfig.from_json_file(self.data_config_file)

        self.data_saver = DataSaver(config=self.data_config)

        self.data_saver(
            {
                "image": np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                "target": np.asarray([0, 1]),
            }
        )
        self.data_saver.disconnect()

    def test_loader_from_json_file(self):
        from matorage.torch import Dataset

        self.test_torch_saver(save_to_json_file=True)

        self.data_config = None

        self.data_config = DataConfig.from_json_file(self.data_config_file)

        self.dataset = Dataset(config=self.data_config)
        loader = DataLoader(self.dataset, batch_size=64, num_workers=8, shuffle=True)

        for batch_idx, (image, target) in enumerate(tqdm(loader)):
            pass

    def test_torch_not_clear(self):
        from matorage.torch import Dataset

        self.test_torch_loader()

        if os.path.exists(self.dataset.cache_path):
            with open(self.dataset.cache_path) as f:
                _pre_file_mapper = json.load(f)

        self.dataset = Dataset(config=self.data_config, clear=False)

        if os.path.exists(self.dataset.cache_path):
            with open(self.dataset.cache_path) as f:
                _next_file_mapper = json.load(f)

        self.assertEqual(_pre_file_mapper, _next_file_mapper)

def suite():
    return unittest.TestSuite(unittest.makeSuite(TorchDataTest))


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
