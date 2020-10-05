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
    return tf.cast(image, tf.float32) / 255.0, label


def data_save(dataset, evaluate=False):
    data_config = DataConfig(
        endpoint="127.0.0.1:9000",
        database="127.0.0.1:5432",
        access_key="minio",
        secret_key="miniosecretkey",
        dataset_name="mnist",
        additional={
            "mode": "train" if not evaluate else "test",
            "framework": "tensorflow",
        },
        attributes=[("image", "float32", (28, 28)), ("target", "int64", (1)),],
    )

    total_dataset = len(dataset[0])
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(60).prefetch(tf.data.experimental.AUTOTUNE)

    data_saver = DataSaver(config=data_config, refresh=True)
    for (image, target) in tqdm(dataset, total=total_dataset // 60):
        data_saver({"image": image, "target": target})
    data_saver.disconnect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tensorflow V2 MNIST Example")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    train_dataset, test_dataset = datasets.mnist.load_data()

    start = time.time()

    if args.train:
        data_save(train_dataset, evaluate=False)
    if args.test:
        data_save(test_dataset, evaluate=True)

    end = time.time()
    print(end - start)
