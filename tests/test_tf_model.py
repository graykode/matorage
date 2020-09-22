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

import gc
import unittest
import tensorflow as tf
from tensorflow import keras
from unittest.case import SkipTest

from tests.test_model import ModelTest

from matorage.model.config import ModelConfig
from matorage.model.tensorflow.v2.manager import ModelManager
from matorage.testing_utils import require_tf


@require_tf
class TFModelTest(ModelTest, unittest.TestCase):
    flag = False

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

    def test_tfmodel_saver(self, model_config=None, save_to_json_file=False):

        self.model = self.create_model()

        if model_config is None:
            self.model_config = ModelConfig(
                **self.storage_config,
                model_name="test_tfmodel_saver",
                additional={"framework": "tensorflow"}
            )
        else:
            self.model_config = model_config

        if save_to_json_file:
            self.model_config_file = "model_config_file.json"
            self.model_config.to_json_file(self.model_config_file)

        self.model_manager = ModelManager(config=self.model_config)

        self.model_manager.save(self.model, step=0)

    def test_tfmodel_saver_from_json_file(self):

        self.test_tfmodel_saver(save_to_json_file=True)

        self.model_config = None
        self.model_manager = None

        self.model_config = ModelConfig.from_json_file(self.model_config_file)

        self.model_manager = ModelManager(config=self.model_config)

        self.model_manager.save(self.model, step=0)

def suite():
    return unittest.TestSuite(unittest.makeSuite(TFModelTest))


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
