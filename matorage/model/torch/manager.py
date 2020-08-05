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

    Args:
        config (:obj:`matorage.ModelConfig`, **require**):
            A ModelConfig instance object

        inmemory (:obj:`boolean`, optional, defaults to `False`):
            If you use this value as `True`, then you can use `HDF5_CORE driver <https://support.hdfgroup.org/HDF5/doc/TechNotes/VFL.html#TOC1>`_
            so the temporary file for uploading or downloading to backend storage,
            such as MinIO, is not stored on disk but is in the memory.
            Keep in mind that using memory is fast because it doesn't use disk IO, but it's not always good.
            If default option(False), then `HDF5_SEC2` driver will be used on posix OS(or `HDF5_WINDOWS` in Windows).

    """

    def __init__(self, config, inmemory=False, num_worker_threads=4, multipart_upload_size=5 * _MB):
        super(ModelManager, self).__init__(config, inmemory, num_worker_threads, multipart_upload_size)

    def _save_model(self, model_folder, model):
        for name, weight in model.state_dict().items():
            self._save_layer(model_folder, name, weight.numpy())