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

import copy
import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tests.test_optimizer import OptimizerTest

from matorage.optimizer.config import OptimizerConfig
from matorage.optimizer.tensorflow.v2.manager import OptimizerManager
from matorage.testing_utils import require_tf


@require_tf
class TFOptimizerTest(OptimizerTest, unittest.TestCase):
    def create_model(self):
        model = tf.keras.models.Sequential(
            [
                keras.layers.Dense(512, activation="relu", input_shape=(784,)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10),
            ]
        )
        model.compile(
            optimizer="adam",
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model

    def test_optimizer_saver(self, optimizer_config=None, save_to_json_file=False):

        (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
        train_labels = train_labels[:1000]
        train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0

        self.model = self.create_model()
        self.model.fit(train_images, train_labels, epochs=1)

        if optimizer_config is None:
            self.optimizer_config = OptimizerConfig(
                **self.storage_config,
                optimizer_name="test_optimizer_saver",
                additional={"framework": "tensorflow"}
            )
        else:
            self.optimizer_config = optimizer_config

        if save_to_json_file:
            self.optimizer_config_file = "model_config_file.json"
            self.optimizer_config.to_json_file(self.optimizer_config_file)

        self.optimizer_manager = OptimizerManager(config=self.optimizer_config)

        self.optimizer_manager.save(self.model.optimizer)

    def test_tfmodel_saver_from_json_file(self):

        self.test_optimizer_saver(save_to_json_file=True)

        self.optimizer_config = None
        self.optimizer_manager = None

        self.optimizer_config = OptimizerConfig.from_json_file(
            self.optimizer_config_file
        )

        self.optimizer_manager = OptimizerManager(config=self.optimizer_config)

        self.optimizer_manager.save(self.model.optimizer)

    def test_optimizer_loader(self):

        (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
        train_labels = train_labels[:1000]
        train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0

        self.model = self.create_model()
        self.model.fit(train_images, train_labels, epochs=1)
        before_optim_weight = copy.deepcopy(self.model.optimizer.get_weights())

        self.test_optimizer_saver()

        self.optimizer_manager.load(self.model.optimizer, step=32)
        after_optim_weight = self.model.optimizer.get_weights()

        assert len(before_optim_weight) == len(after_optim_weight)
        for i in range(1, len(after_optim_weight)):
            assert not np.array_equal(after_optim_weight, before_optim_weight)


def suite():
    return unittest.TestSuite(unittest.makeSuite(TFOptimizerTest))


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
