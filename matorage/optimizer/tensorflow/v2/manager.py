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

_KB = 1024
"""The size of a Kilobyte in bytes"""

_MB = 1024 * _KB
"""The size of a Megabyte in bytes"""

import io
import ast
import h5py
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from matorage.optimizer.manager import Manager


class OptimizerManager(Manager):

    """
        Optimizer Manager Tensorflow classes. This class overrides ``Manager``.

        Note:
            Unlike Dataset, optimizer weight is loaded entirely into cpu memory.
            Therefore, the `HDF5_CORE` driver using the memory option is default setting.

        Args:
            config (:obj:`matorage.OptimizerConfig`, **require**):
                A OptimizerConfig instance object
            num_worker_threads (:obj:`int`, optional, defaults to `4`):
                Number of backend storage worker to upload or download.
            multipart_upload_size (:obj:`integer`, optional, defaults to `5 * 1024 * 1024`):
                size of the incompletely uploaded object.
                You can sync files faster with `multipart upload in MinIO. <https://github.com/minio/minio-py/blob/master/minio/api.py#L1795>`_
                This is because MinIO clients use multi-threading, which improves IO speed more
                efficiently regardless of Python's Global Interpreter Lock(GIL).

        Examples::

            from matorage.tensorflow import OptimizerManager
            optimizer_config = OptimizerConfig(
                endpoint='127.0.0.1:9000',
                access_key='minio',
                secret_key='miniosecretkey',
                optimizer_name='testoptimizer',
                additional={
                    "version" : "1.0.1"
                }
            )

            optimizer_manager = OptimizerManager(config=optimizer_config)
            optimizer_manager.save(model.optimizer, step=100)

        """

    def __init__(self, config, num_worker_threads=4, multipart_upload_size=5 * _MB):
        super(OptimizerManager, self).__init__(
            config, num_worker_threads, multipart_upload_size
        )

    def _get_step(self, optimizer):
        if len(optimizer.weights) > 0:
            return optimizer.weights[0].numpy()
        else:
            return None

    def _set_metadata(self, metadata, optimizer, step):
        assert isinstance(optimizer, OptimizerV2)

        metadata["optimizer"].update(
            {
                str(step): {
                    "framework": "tensorflow",
                    "config": str(optimizer.get_config()),
                    "param_groups": [],
                }
            }
        )

    def _set_scheduler(self, metadata, scheduler, step):
        raise NotImplementedError()

    def _save_optimizer(self, step, optimizer):
        assert isinstance(optimizer, OptimizerV2)
        assert isinstance(self.config.metadata, dict)
        assert str(step) in self.config.metadata["optimizer"]

        for i in range(1, len(optimizer.weights)):
            param_name = optimizer.weights[i].name
            param_value = optimizer.weights[i].numpy()

            self._save_param(step, group="", name=param_name, weight=param_value)

            self.config.metadata["optimizer"][str(step)]["param_groups"].append(
                param_name
            )

    def _load_optimizer(self, step, layers, optimizer):
        assert isinstance(optimizer, OptimizerV2)
        assert isinstance(self.config.metadata, dict)

        step = str(step)
        if step not in self.config.metadata["optimizer"]:
            raise KeyError(
                "Available only in {}".format(
                    list(self.config.metadata["optimizer"].keys())
                )
            )

        _ordered_weight = {}
        for layer in layers:
            name = layer.object_name
            layer_image = self._client.get_object(
                bucket_name=self.config.bucket_name, object_name=name
            ).read()

            layer_image = h5py.File(io.BytesIO(layer_image), "r")

            name = name[name.find("/") + 1 :]
            _ordered_weight[name] = layer_image[self.type][:]

        new_weight = [np.array(int(step))]
        for param_name in self.config.metadata["optimizer"][str(step)]["param_groups"]:
            new_weight.append(_ordered_weight[param_name])

        _metadata_config = ast.literal_eval(
            self.config.metadata["optimizer"][str(step)]["config"]
        )
        optimizer.from_config(_metadata_config)
        optimizer.set_weights(new_weight)

    def save(self, optimizer):
        """
        save weight of optimizer

        Args:
            optimizer (:obj:`tf.keras.optimizers.Optimizer`, **require**):
                Tensorflow optimizer.

        Examples::

            >>> model = Model()
            # model training...
            >>> optimizer_manager.save(model.optimizer)

        """
        super(OptimizerManager, self).save(optimizer)

    def load(self, optimizer, step):
        """
        load weight of optimizer

        Args:
            optimizer (:obj:`tf.keras.optimizers.Optimizer`, **require**):
                Tensorflow optimizer.
            step (:obj:`integer`, **require**):
                optimizer step.

        Examples::

            >>> model = Model()
            >>> optimizer_manager.load(model.optimizer, step=938)

        """
        super(OptimizerManager, self).load(optimizer, step)
