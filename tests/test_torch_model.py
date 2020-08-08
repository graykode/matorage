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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    def test_torchmodel_saver(self, model_config=None, save_to_json_file=False):
        if model_config is None:
            self.model_config = ModelConfig(
                **self.storage_config,
                model_name="test_torchmodel_saver",
                additional={"framework": "pytorch"}
            )
        else:
            self.model_config = model_config

        if save_to_json_file:
            self.model_config_file = "model_config_file.json"
            self.model_config.to_json_file(self.model_config_file)

        self.model_manager = ModelManager(config=self.model_config)

        model = Model()
        self.model_manager.save(model, step=0)

    def test_torchmodel_saver_from_json_file(self):

        self.test_torchmodel_saver(save_to_json_file=True)

        self.model_config = None
        self.model_manager = None

        self.model_config = ModelConfig.from_json_file(self.model_config_file)

        self.model_manager = ModelManager(config=self.model_config)

        model = Model()
        self.model_manager.save(model, step=0)

    def test_torchmodel_loader(self):

        self.test_torchmodel_saver()

        model = Model()
        self.model_manager.load(model, step=0)

    def test_torchmodel_loader_with_compressor(self):

        model_config = ModelConfig(
            **self.storage_config,
            model_name="test_torchmodel_loader_with_compressor",
            additional={"framework": "pytorch"},
            compressor={"complevel": 4, "complib": "zlib"}
        )

        self.test_torchmodel_saver(model_config=model_config)

        self.model_manager = ModelManager(config=self.model_config)

        model = Model()
        self.model_manager.load(model, step=0)

    def test_torchmodel_layer_loader(self):

        self.test_torchmodel_saver()

        self.model_manager = ModelManager(config=self.model_config)

        self.model_manager.load("f.weight", step=0)

    @unittest.skip("skip")
    def test_mnist_eval(self, model, device):
        test_dataset = datasets.MNIST(
            "/tmp/data", train=False, transform=self.transform
        )
        test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for image, target in test_loader:
                image, target = image.to(device), target.to(device)
                output = model(image)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        return correct

    def test_mnist_reloaded(self):
        import torch.optim as optim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset = datasets.MNIST(
            "/tmp/data", train=True, download=True, transform=self.transform
        )

        model = Model().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4)

        for batch_idx, (image, target) in enumerate(tqdm(train_loader)):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        self.model_config = ModelConfig(
            **self.storage_config,
            model_name="testmodel",
            additional={"version": "1.0.1"}
        )
        self.model_manager = ModelManager(config=self.model_config)

        self.model_manager.save(model, epoch=1)

        pretrained_model = Model().to(device)
        correct = self.test_mnist_eval(model=pretrained_model, device=device)

        self.model_manager.load(pretrained_model, epoch=1)
        pretrained_correct = self.test_mnist_eval(model=pretrained_model, device=device)

        assert correct < pretrained_correct


def suite():
    return unittest.TestSuite(unittest.makeSuite(TorchModelTest))


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
