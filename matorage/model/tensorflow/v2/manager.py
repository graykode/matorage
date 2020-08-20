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

    Examples::

        from matorage import ModelConfig
        from matorage.tensorflow import ModelManager
        from tensorflow.keras import layers, Sequential

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
            Sequential([
                layers.Dense(10),
                layers.ReLU()
            ]),
            Sequential([
                layers.Dense(10),
                layers.ReLU()
            ])
        ])
        model.build(input_shape=(None, 5))

        model_manager.save(model, step=100)

    """

    def __init__(self, config, num_worker_threads=4, multipart_upload_size=5 * _MB):
        super(ModelManager, self).__init__(
            config, num_worker_threads, multipart_upload_size
        )

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
            if name.find("/") > -1:
                name = name[name.find("/") + 1 :]

            if name in keys:
                layer_image = self._client.get_object(
                    bucket_name=self.config.bucket_name,
                    object_name=f"{model_folder}/{name}",
                ).read()

                layer_image = h5py.File(io.BytesIO(layer_image), "r")
                weight[name] = tf.convert_to_tensor(layer_image[self.type][:])

        if isinstance(model, str):
            return weight
        else:
            self._load_state_dict(model=model, weight_dict=weight)

    def _load_state_dict(self, model, weight_dict):
        original_keras_version = keras_version
        original_backend = K.backend()

        weight_value_tuples = []
        for k, layer in enumerate(model.layers):
            weight_names = [l.name for l in layer.weights]
            if len(weight_names) == 0:
                continue
            weight_values = [
                np.asarray(weight_dict[weight_name]) for weight_name in weight_names
            ]

            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            weight_values = preprocess_weights_for_loading(
                layer, weight_values, original_keras_version, original_backend
            )

            if len(weight_values) != len(symbolic_weights):
                raise ValueError(
                    "Layer #"
                    + str(k)
                    + ' (named "'
                    + layer.name
                    + '" in the current model) was found to '
                    "correspond to layer " + layer.name + " in the save file. "
                    "However the new layer "
                    + layer.name
                    + " expects "
                    + str(len(symbolic_weights))
                    + " weights, but the saved weights have "
                    + str(len(weight_values))
                    + " elements."
                )
            weight_value_tuples += zip(symbolic_weights, weight_values)
        K.batch_set_value(weight_value_tuples)

    def save(self, model, **kwargs):
        """
        save weight of model

        Args:
            model (:obj:`tf.keras.Model`, **require**):
                Tensorflow model (``tf.keras.Model`` type)
            kwargs (:obj:`**kwargs`, optional):
                metadata about step or epoch for model.

        Examples::

            >>> model_manager.save(model, step=0)
            >>> model_manager.save(model, epoch=0)
            >>> model_manager.save(model, epoch=0, step=0)

        """
        super(ModelManager, self).save(model, **kwargs)

    def load(self, model, **kwargs):
        """
        load weight of model

        Args:
            model (:obj:`tf.keras.Model` or `string`, **require**):
                Tensorflow model(``tf.keras.Model`` type) or layer name(string type).
            kwargs (:obj:`**kwargs`, optional):
                metadata about step or epoch for model.

        Returns:
            :obj:`None or OrderedDict`: If ``model`` is tensorflow model, weight is loaded into the model and return None.
            however, If it is a string type with the name of the layer, it returns the weight of the OrderedDict type.

        Examples::

            # Load entire model
            >>> model_manager.load(model, step=0)

            # Load sub-layer model
            >>> submodel = Sequential([
            >>>   Sequential([
            >>>     layers.Dense(10),
            >>>     layers.ReLU()
            >>>   ])
            >>> ])
            >>> submodel.build(input_shape=(None, 5))
            >>> model_manager.load(submodel, step=0)

            >>> model_manager.load('sequential/dense/kernel:0', step=0)
            OrderedDict([('sequential/dense/kernel:0',
            <tf.Tensor: shape=(5, 784), dtype=float32, numpy=
            array([[ 0.08160326,  0.00161414, -0.00507049, ...,  0.02965256,
                    -0.07447692,  0.02029154],
                   [-0.06808375,  0.0112161 , -0.0640984 , ..., -0.05060118,
                    -0.03650254,  0.01808494],
                   [ 0.00063588,  0.00848304, -0.01014224, ...,  0.0616277 ,
                    -0.05507688,  0.02844934],
                   [-0.00206905,  0.04553737,  0.03098481, ..., -0.05891491,
                     0.0705805 , -0.03912991],
                   [ 0.04252511, -0.04907732, -0.07053198, ...,  0.00260394,
                     0.07418892, -0.0714546 ]], dtype=float32)>)])

        """
        return super(ModelManager, self).load(model, **kwargs)
