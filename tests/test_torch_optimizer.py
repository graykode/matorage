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
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tests.test_optimizer import OptimizerTest

from matorage.optimizer.config import OptimizerConfig
from matorage.optimizer.torch.manager import OptimizerManager
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
class TorchOptimizerTest(OptimizerTest, unittest.TestCase):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "/tmp/data", train=True, download=True, transform=transform
    )

    def test_optimizer_saver(self, optimizer_config=None, save_to_json_file=False):

        model = Model().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(self.train_dataset, batch_size=64, num_workers=4)

        for batch_idx, (image, target) in enumerate(tqdm(train_loader)):
            image, target = image.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if optimizer_config is None:
            self.optimizer_config = OptimizerConfig(
                **self.storage_config,
                optimizer_name="testoptimizer",
                additional={"version": "1.0.1"}
            )
        else:
            self.optimizer_config = optimizer_config

        if save_to_json_file:
            self.optimizer_config_file = "optimizer_config_file.json"
            self.optimizer_config.to_json_file(self.optimizer_config_file)

        self.optimizer_manager = OptimizerManager(config=self.optimizer_config)
        self.optimizer_manager.save(optimizer)

    def test_optimizer_saver_from_json_file(self):

        self.test_optimizer_saver(save_to_json_file=True)

        self.optimizer_config = None
        self.optimizer_manager = None

        self.optimizer_config = OptimizerConfig.from_json_file(
            self.optimizer_config_file
        )

        self.optimizer_manager = OptimizerManager(config=self.optimizer_config)

        model = Model().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(self.train_dataset, batch_size=64, num_workers=4)

        for batch_idx, (image, target) in enumerate(tqdm(train_loader)):
            image, target = image.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        self.optimizer_manager = OptimizerManager(config=self.optimizer_config)
        self.optimizer_manager.save(optimizer)

    def test_optimizer_loader(self):

        self.test_optimizer_saver()

        model = Model().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        self.optimizer_manager.load(optimizer, step=938)

    def test_optimizer_scheduler_saver(self):
        from torch.optim.lr_scheduler import StepLR

        model = Model().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1.0)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=134, gamma=0.99)

        train_loader = DataLoader(self.train_dataset, batch_size=64, num_workers=4)

        for batch_idx, (image, target) in enumerate(tqdm(train_loader)):
            image, target = image.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

        self.optimizer_config = OptimizerConfig(
            **self.storage_config,
            optimizer_name="testoptimizerwithscheduler",
            additional={"version": "1.0.1"}
        )
        self.optimizer_manager = OptimizerManager(config=self.optimizer_config)
        self.optimizer_manager.save(optimizer, scheduler)

        return scheduler

    def test_optimizer_scheduler_loader(self):
        from torch.optim.lr_scheduler import StepLR

        _scheduler = self.test_optimizer_scheduler_saver()

        model = Model()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        self.optimizer_manager.load_with_scheduler(optimizer, scheduler, step=938)
        self.assertEqual(_scheduler.state_dict(), scheduler.state_dict())

def suite():
    return unittest.TestSuite(unittest.makeSuite(TorchOptimizerTest))


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
