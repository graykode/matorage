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
import torch
from collections import OrderedDict

from matorage.model.manager import Manager

class ModelManager(Manager):

    """
    Model Manager Pytorch classes. This class overrides ``Manager``.

    .. code-block:: python

        from matorage.torch import ModelManager

        model_config = ModelConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            model_name='testmodel',
            additional={
                "version" : "1.0.1"
            }
        )

        model_manager = ModelManager(config=model_config, inmemory=False)

        import torch.nn as nn
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.f = nn.Linear(5, 10)
            def forward(self, x):
                return self.f(x)

        model_manager.save({ "step" :100 }, model)

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
        for name, weight in model.state_dict().items():
            self._save_layer(model_folder, name, weight.numpy())

    def _load_model(self, model_folder, layers, model):
        weight = OrderedDict()

        if isinstance(model, str):
            keys = [model]
        else:
            keys = list(model.state_dict().keys())

        for layer in layers:
            name = os.path.basename(layer.object_name)
            if name in keys:
                layer_image = self._client.get_object(
                    bucket_name=self.config.bucket_name,
                    object_name=f"{model_folder}/{name}"
                ).read()

                layer_image = h5py.File(io.BytesIO(layer_image), 'r')
                weight[name] = torch.from_numpy(layer_image[self.type][:])

        if isinstance(model, str):
            return weight
        else:
            model.load_state_dict(weight)
            return model