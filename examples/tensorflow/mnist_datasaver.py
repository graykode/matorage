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

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def traindata_save(dataset):
    train_image, train_target = dataset
    traindata_config = DataConfig(
        endpoint='127.0.0.1:9000',
        access_key='minio',
        secret_key='miniosecretkey',
        dataset_name='mnist',
        additional={
            "mode": "train",
            "framework": "tensorflow"
        },
        attributes=[
            DataAttribute('image', 'float32', (28, 28)),
            DataAttribute('target', 'int64', (1))
        ]
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = train_dataset.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(60).prefetch(tf.data.experimental.AUTOTUNE)

    traindata_saver = DataSaver(config=traindata_config)
    for (image, target) in tqdm(train_dataset, total=60000//60):
        traindata_saver({
            'image': image,
            'target': target
        })
    traindata_saver.disconnect()

def testdata_save(dataset):
    testdata_config = DataConfig(
        endpoint='127.0.0.1:9000',
        access_key='minio',
        secret_key='miniosecretkey',
        dataset_name='mnist',
        additional={
            "mode": "test",
            "framework": "tensorflow"
        },
        attributes=[
            DataAttribute('image', 'float32', (28, 28)),
            DataAttribute('target', 'int64', (1))
        ]
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    test_dataset = test_dataset.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(60).prefetch(tf.data.experimental.AUTOTUNE)

    testdata_saver = DataSaver(config=testdata_config)
    for (image, target) in tqdm(test_dataset, total=10000 // 60):
        testdata_saver({
            'image': image,
            'target': target
        })
    testdata_saver.disconnect()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tensorflow V2 MNIST Example')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--test', action='store_true', default=True)
    args = parser.parse_args()

    train_dataset, test_dataset = datasets.mnist.load_data()

    start = time.time()

    if args.train:
        traindata_save(train_dataset)
    if args.test:
        testdata_save(test_dataset)

    end = time.time()
    print(end - start)