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

import os
import io
import h5py
import numpy as np
import tensorflow as tf
from collections import OrderedDict, defaultdict
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import __version__ as keras_version
from tensorflow.python.keras.saving.hdf5_format import preprocess_weights_for_loading

from matorage.model.manager import Manager

class ModelManager(Manager):

    """
    Model Manager Tensorflow classes. This class overrides ``Manager``.

    .. code-block:: python

        from matorage.tensorflow import ModelManager

        model_config = ModelConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            model_name='testmodel',
            additional={
                "version" : "1.0.1"
            }
        )

        model_manager = ModelManager(config=model_config)

        model = Sequential([
            layers.Dense(10)
        ])
        model.build(input_shape=(None, 5))

        model_manager.save(model, step=100)

    Note:
        Unlike Dataset, model weight is loaded entirely into cpu memory.
        Therefore, the `HDF5_CORE` driver using the memory option is default setting.

    Args:
        config (:obj:`matorage.ModelConfig`, **require**):
            A ModelConfig instance object
        num_worker_threads (:obj:`int`, optional, defaults to `4`):
            Number of backend storage worker to upload or download.
        multipart_upload_size (:obj:`integer`, optional, defaults to `5 * 1024 * 1024`):
            size of the incompletely uploaded object.
            You can sync files faster with `multipart upload in MinIO. <https://github.com/minio/minio-py/blob/master/minio/api.py#L1795>`_
            This is because MinIO clients use multi-threading, which improves IO speed more
            efficiently regardless of Python's Global Interpreter Lock(GIL).

    """

    def __init__(self, config, num_worker_threads=4, multipart_upload_size=5 * _MB):
        super(ModelManager, self).__init__(config, num_worker_threads, multipart_upload_size)

    def _save_model(self, model_folder, model):
        for layer in model.weights:
            self._save_layer(model_folder, layer.name, layer.numpy())

    def _load_model(self, model_folder, layers, model):
        weight = OrderedDict()

        if isinstance(model, str):
            keys = [model]
        else:
            keys = [w.name for w in model.weights]

        for layer in layers:
            name = layer.object_name
            if name.find('/') > -1:
                name = name[name.find('/') + 1:]

            if name in keys:
                layer_image = self._client.get_object(
                    bucket_name=self.config.bucket_name,
                    object_name=f"{model_folder}/{name}"
                ).read()

                layer_image = h5py.File(io.BytesIO(layer_image), 'r')
                weight[name] = tf.convert_to_tensor(layer_image[self.type][:])

        if isinstance(model, str):
            return weight
        else:
            self._load_state_dict(
                model=model,
                weight_dict=weight
            )

    def _load_state_dict(self, model, weight_dict):
        original_keras_version = keras_version
        original_backend = K.backend()

        weight_value_tuples = []
        for k, layer in enumerate(model.layers):
            weight_names = [l.name for l in layer.weights]
            if len(weight_names) == 0:
                continue
            weight_values = [np.asarray(weight_dict[weight_name]) for weight_name in weight_names]

            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            weight_values = preprocess_weights_for_loading(
                layer, weight_values, original_keras_version, original_backend)

            if len(weight_values) != len(symbolic_weights):
                raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                                 '" in the current model) was found to '
                                 'correspond to layer ' + layer.name + ' in the save file. '
                                                                 'However the new layer ' + layer.name + ' expects ' +
                                 str(len(symbolic_weights)) +
                                 ' weights, but the saved weights have ' +
                                 str(len(weight_values)) + ' elements.')
            weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)