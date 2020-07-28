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
import tensorflow as tf
from tensorflow.keras import datasets

from matorage import *
from matorage.tensorflow import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--test', action='store_true', default=True)
    args = parser.parse_args()

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

    train_loader = tf.data.Dataset.from_tensor_slices(train_dataset.get_objectnames)
    train_loader = train_loader.interleave(lambda filename: tf.data.Dataset.from_generator(
        train_dataset,
        (tf.float32, tf.int64),
        (tf.TensorShape([28, 28]), tf.TensorShape([1])),
        args=(filename,)), cycle_length=4).batch(64)

    for batch_idx, (image, target) in enumerate(tqdm(train_loader, total=60000//64)):
        print(image.shape, target.shape)
        pass

    end = time.time()
    print(end - start)