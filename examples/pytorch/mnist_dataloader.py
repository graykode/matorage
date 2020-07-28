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

def test(model, device):

    testdata_config = DataConfig(
        endpoint='127.0.0.1:9000',
        access_key='minio',
        secret_key='miniosecretkey',
        dataset_name='mnist',
        additional={
            "mode": "test"
        },
    )
    test_dataset = MTRDataset(config=testdata_config, clear=True)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--test', action='store_true', default=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        model = Model().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    traindata_config = DataConfig(
        endpoint='127.0.0.1:9000',
        access_key='minio',
        secret_key='miniosecretkey',
        dataset_name='mnist',
        additional={
            "mode": "train"
        },
    )
    train_dataset = MTRDataset(config=traindata_config, clear=True)

    start = time.time()

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    for epoch in range(5):
        for batch_idx, (image, target) in enumerate(train_loader):
            image, target = image.to(device), target.to(device)
            if args.train:
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 20 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx * len(image), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
        if args.test:
            test(model, device)

    end = time.time()
    print(end - start)