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

# https://www.tensorflow.org/tutorials/quickstart/advanced

import time
import argparse
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

from matorage import *
from matorage.tensorflow import *

model = Sequential(
    [
        layers.Reshape((28 * 28,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)
model.build(input_shape=(None, 28 * 28))
optimizer = optimizers.Adam(lr=0.01)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")


@tf.function
def train_step(image, target):
    with tf.GradientTape() as tape:
        output = model(image)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=output, labels=tf.one_hot(target, depth=10)
        )
        loss = tf.reduce_sum(loss) / 64

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(target, output)


@tf.function
def test_step(images, labels):
    output = model(images)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=output, labels=tf.one_hot(target, depth=10)
    )

    test_loss(loss)
    test_accuracy(labels, output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tensorflow V2 MNIST Example")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    storage_config = {
        "endpoint": "127.0.0.1:9000",
        "database": "127.0.0.1:5432",
        "access_key": "minio",
        "secret_key": "miniosecretkey",
        "secure": False,
    }

    if args.train:
        traindata_config = DataConfig(
            **storage_config,
            dataset_name="mnist",
            additional={"mode": "train", "framework": "tensorflow"},
        )
        train_dataset = Dataset(
            config=traindata_config, shuffle=True, batch_size=64, clear=True
        )

        model_config = ModelConfig(
            **storage_config,
            model_name="mnist_example",
            additional={"framework": "tensorflow"},
        )
        model_manager = ModelManager(config=model_config)

        optimizer_config = OptimizerConfig(
            **storage_config,
            optimizer_name="mnist_example",
            additional={"framework": "tensorflow"},
        )
        optimizer_manager = OptimizerManager(config=optimizer_config)

    if args.test:
        testdata_config = DataConfig(
            **storage_config,
            dataset_name="mnist",
            additional={"mode": "test", "framework": "tensorflow"},
        )
        test_dataset = Dataset(config=testdata_config, batch_size=64, clear=True)

    epochs = 5
    template = "Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"

    start = time.time()

    for e in range(5):
        if args.train:
            for batch_idx, (image, target) in enumerate(
                tqdm(train_dataset.dataloader, total=60000 // 64)
            ):
                train_step(image, target)
            model_manager.save(model, epoch=(e + 1))
            optimizer_manager.save(optimizer)
        if args.test:
            for batch_idx, (image, target) in enumerate(
                tqdm(test_dataset.dataloader, total=10000 // 64)
            ):
                test_step(image, target)
        print(
            template.format(
                e + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )

    end = time.time()
    print(end - start)
