# [matorage](https://matorage.readthedocs.io/en/latest)

<p align="center">
<a href="https://travis-ci.com/github/graykode/matorage"><img alt="Build Status" src="https://travis-ci.com/graykode/matorage.svg?branch=master"></a>
<a href="https://matorage.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/matorage/badge/?version=latest"></a>
<a href="https://github.com/graykode/matorage/blob/master/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/matorage/"><img alt="PyPI" src="https://img.shields.io/pypi/v/matorage"></a>
<a href="https://pepy.tech/project/matorage"><img alt="Downloads" src="https://static.pepy.tech/badge/matorage"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

**An efficient way to store/load and manage dataset, model and optimizer for deep learning with matorage!**

Matorage is tensor(multidimensional matrix) object storage manager for deep learning framework(Pytorch, Tensorflow V2, Keras).

## Features

- Boilerplated data pipeline for dataset, model and optimizer.
- High performance on tensor storage

**For researchers who need to focus on model training**:

- Support storing data in pre-processed Tensor(multidimensional matrix), eliminate training time.
- Reduce storage space through multiple compression methods.
- Manage data and models while training

**For AI Developer who need to focus on creating data pipeline:**

- Concurrency data save & load
- Compatible with object storage such as MinIO, S3
- Generate pipeline from user endpoints data.

## Quick Start with Pytorch Example

For an example of tensorflow, refer to the detailed document.
If you want to see the full code, see below

- [Pytorch Mnist Example](examples/pytorch/mnist)
- [Tensorflow Mnist Example](examples/tensorflow/mnist)
- [SQuAD 1.1/2.0 Example](examples/pytorch/squad)

- Content
    - [0. Install matorage with pip](https://github.com/graykode/matorage#0-install-matorage-with-pip)
    - [1. Set up Minio Server with docker](https://github.com/graykode/matorage#1-set-up-minio-server-with-docker)
    - [2. Save pre-processed dataset](https://github.com/graykode/matorage#2-save-pre-processed-dataset)
    - [3. Load dataset from matorage](https://github.com/graykode/matorage#3-load-dataset-from-matorage)
    - [4. Save & Load Model when training](https://github.com/graykode/matorage#4-save--load-model-when-training)
    - [5. Save & Load Optimizer when training](https://github.com/graykode/matorage#5-save--load-optimizer-when-training)
- [Unittest](https://github.com/graykode/matorage#unittest)

#### 0. Install matorage with pip

```bash
$ pip install matorage
```


#### 1. Set up Minio Server with docker

quick start with NAS(network access storage) using docker
It can be managed through the web through the address http://127.0.0.1:9000/, and security is managed through ``MINIO_ACCESS_KEY`` and ``MINIO_SECRET_KEY``.

```bash
$ mkdir ~/shared # create nas storage folder
$ docker run -it -p 9000:9000 \
    --restart always -e \
    "MINIO_ACCESS_KEY=minio" -e \
    "MINIO_SECRET_KEY=miniosecretkey" \
    -v ~/shared:/container/vol \
    minio/minio gateway nas /container/vol
```


#### 2. Save pre-processed dataset

First, create a ``DataConfig`` by importing matorage.
This is an example of pre-processing mnist and storing it in distributed storage.
``additional`` is freely in the form of a dict, and records the shape and type of tensor to be stored in ``attributes``.

```python
from matorage import DataConfig

traindata_config = DataConfig(
    endpoint='127.0.0.1:9000',
    access_key='minio',
    secret_key='miniosecretkey',
    dataset_name='mnist',
    additional={
        "mode": "train",
        "framework" : "pytorch",
        ...
        "blah" : "blah"
    },
    attributes=[
        ('image', 'float32', (1, 28, 28)),
        ('target', 'int64', (1))
    ]
)
```

Now do a simple pre-processing and save the data.

```python
from matorage import DataSaver

traindata_saver = DataSaver(config=traindata_config)
train_loader = DataLoader(dataset, batch_size=60, num_workers=8)
for (image, target) in tqdm(train_loader):
    # image shape : torch.Size([64, 1, 28, 28])
    # target shape : torch.Size([64])
    traindata_saver({
        'image': image,
        'target': target
    })
traindata_saver.disconnect()
```


#### 3. Load dataset from matorage

Now fetch data iteratively from storage with the same config as the saved dataset when training.

```python
from matorage.torch import Dataset

train_dataset = Dataset(config=traindata_config, clear=True)
train_loader = DataLoader(
    train_dataset, batch_size=64, num_workers=8, shuffle=True
)

for batch_idx, (image, target) in enumerate(tqdm(train_loader)):
    image, target = image.to(device), target.to(device)
```

Only an index can be fetched through lazy load.

```python
train_dataset = Dataset(config=traindata_config, clear=True)
print(train_dataset[0], len(train_dataset))
```


#### 4. Save & Load Model when training

During training, you can save and load models of specific steps or epochs in distributed storage through inmemory.
First, make the model config the same as the dataset.

```python
from matorage import ModelConfig
from matorage.torch import ModelManager

model_config = ModelConfig(
    endpoint='127.0.0.1:9000',
    access_key='minio',
    secret_key='miniosecretkey',
    model_name='mnist_simple_training',
    additional={
        "version" : "1.0.1",
        ...
        "blah" : "blah"
    }
)

model_manager = ModelManager(config=model_config)
print(model_manager.get_metadata)
model_manager.save(model, epoch=1)
print(model_manager.get_metadata)
```

When an empty model is loaded with specific steps or epochs, the appropriate weight is filled into the model.

```python
print(model.state_dict())
model_manager.load(model, epoch=1)
print(model.state_dict())
# load a layer weight.
print(model_manager.load('net1.0.weight', step=0))
```


#### 5. Save & Load Optimizer when training

Save and load of optimizer is similar to managing model.

```python
from matorage import OptimizerConfig
from matorage.torch import OptimizerManager

optimizer_config = OptimizerConfig(
    endpoint='127.0.0.1:9000',
    access_key='minio',
    secret_key='miniosecretkey',
    optimizer_name='adam',
    additional={
        "model" : "1.0.1",
        ...
        "blah" : "blah"
    }
)

optimizer_manager = OptimizerManager(config=optimizer_config)
print(optimizer_manager.get_metadata)
# The optimizer contains information about the step.
optimizer_manager.save(optimizer)
print(optimizer_manager.get_metadata)
```

When an empty optimizer is loaded with specific steps, the appropriate weight is filled into the optimizer.

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer_manager.load(optimizer, step=938)
```


### Unittest
```bash
$ git clone https://github.com/graykode/matorage && cd matorage
$ python -m tests.test_suite
```


### Framework Requirement

- torch(>=1.0.0), torchvision(>=0.2.2)
- tensorflow(>=2.2), tensorflow_io(>=0.13)

### Author

[Tae Hwan Jung(@graykode)](https://github.com/graykode/matorage>)
We are looking for a contributor.
