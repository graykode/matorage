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

import time
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from matorage import *
from matorage.torch import *

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

def train(model, train_loader, optimizer, criterion, device):
    for batch_idx, (image, target) in enumerate(tqdm(train_loader)):
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model().to(device)
    if args.train:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        traindata_config = DataConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            dataset_name='mnist',
            additional={
                "mode": "train",
                "framework": "pytorch"
            },
        )
        train_dataset = Dataset(config=traindata_config, clear=True)
        train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)

    if args.test:
        testdata_config = DataConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            dataset_name='mnist',
            additional={
                "mode": "test",
                "framework": "pytorch"
            },
        )
        test_dataset = Dataset(config=testdata_config, clear=True)
        test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)

    start = time.time()

    for epoch in range(5):
        if args.train:
            train(model, train_loader, optimizer, criterion, device)
        if args.test:
            test(model, test_loader, device)

    end = time.time()
    print(end - start)