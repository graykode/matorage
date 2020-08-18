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
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from matorage import *


def traindata_save(dataset):
    traindata_config = DataConfig(
        endpoint="127.0.0.1:9000",
        access_key="minio",
        secret_key="miniosecretkey",
        dataset_name="mnist",
        additional={"mode": "train", "framework": "pytorch"},
        attributes=[
            DataAttribute("image", "float32", (1, 28, 28)),
            DataAttribute("target", "int64", (1)),
        ],
    )

    traindata_saver = DataSaver(config=traindata_config, refresh=True)

    train_loader = DataLoader(dataset, batch_size=60, num_workers=8)
    for (image, target) in tqdm(train_loader):
        traindata_saver({"image": image, "target": target})
    traindata_saver.disconnect()


def testdata_save(dataset):
    testdata_config = DataConfig(
        endpoint="127.0.0.1:9000",
        access_key="minio",
        secret_key="miniosecretkey",
        dataset_name="mnist",
        additional={"mode": "test", "framework": "pytorch"},
        attributes=[
            DataAttribute("image", "float32", (1, 28, 28)),
            DataAttribute("target", "int64", (1)),
        ],
    )

    testdata_saver = DataSaver(config=testdata_config, refresh=True)

    test_loader = DataLoader(dataset, batch_size=60, num_workers=8)
    for (image, target) in tqdm(test_loader):
        testdata_saver({"image": image, "target": target})
    testdata_saver.disconnect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    start = time.time()

    if args.train:
        train_dataset = datasets.MNIST(
            "/tmp/data", train=True, download=True, transform=transform
        )
        traindata_save(train_dataset)
    if args.test:
        test_dataset = datasets.MNIST("/tmp/data", train=False, transform=transform)
        testdata_save(test_dataset)

    end = time.time()
    print(end - start)
