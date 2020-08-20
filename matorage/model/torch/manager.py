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

        model_manager = ModelManager(config=model_config)

        import torch.nn as nn
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.net1 = nn.Sequential(
                    nn.Linear(5, 10),
                    nn.ReLU(),
                )
                self.net2 = nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU(),
                )
            def forward(self, x):
                x = self.net1(x)
                x = self.net2(x)
                return x

        model = Model()
        model_manager.save(model, step=100)


    """

    def __init__(self, config, num_worker_threads=4, multipart_upload_size=5 * _MB):
        super(ModelManager, self).__init__(
            config, num_worker_threads, multipart_upload_size
        )

    def _save_model(self, model_folder, model):
        for name, weight in model.state_dict().items():
            self._save_layer(model_folder, name, weight.cpu().numpy())

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
                    object_name=f"{model_folder}/{name}",
                ).read()

                layer_image = h5py.File(io.BytesIO(layer_image), "r")
                weight[name] = torch.from_numpy(layer_image[self.type][:])

        if isinstance(model, str):
            return weight
        else:
            model.load_state_dict(weight)

    def save(self, model, **kwargs):
        """
        save weight of model

        Args:
            model (:obj:`torch.nn.Module`, **require**):
                Pytorch model (``torch.nn.Module`` type)
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
            model (:obj:`torch.nn.Module` or `string`, **require**):
                Pytorch model(``torch.nn.Module`` type) or layer name(string type).
            kwargs (:obj:`**kwargs`, optional):
                metadata about step or epoch for model.

        Returns:
            :obj:`None or OrderedDict`: If ``model`` is pytorch model, weight is loaded into the model and return None.
            however, If it is a string type with the name of the layer, it returns the weight of the OrderedDict type.

        Examples::

            # Load entire model
            >>> model = Model()
            >>> model_manager.load(model, step=0)
            >>> model
            Model(
              (net1): Sequential(
                (0): Linear(in_features=5, out_features=10, bias=True)
                (1): ReLU()
              )
              (net2): Sequential(
                (0): Linear(in_features=10, out_features=5, bias=True)
                (1): ReLU()
              )
            )

            # Load sub-layer model
            >>> class SubModel(nn.Module):
            >>>   def __init__(self):
            >>>     super(SubModel, self).__init__()
            >>>     self.net1 = nn.Sequential(
            >>>         nn.Linear(5, 10),
            >>>         nn.ReLU(),
            >>>     )

            >>> submodel = SubModel()
            >>> model_manager.load(submodel, step=0)
            >>> model
            Model(
              (net1): Sequential(
                (0): Linear(in_features=5, out_features=10, bias=True)
                (1): ReLU()
              )
            )

            # Load from layer name
            >>> model_manager.load('net1.0.weight', step=0)
            OrderedDict([('net1.0.weight',
            tensor([[ 0.2679, -0.2147, -0.1927, -0.3263,  0.0930],
                [ 0.0144,  0.2935,  0.3614, -0.0493, -0.3772],
                [ 0.4101, -0.1864,  0.1076, -0.3900,  0.3613],
                [-0.2831,  0.3692,  0.3367,  0.2491, -0.2971],
                [-0.3019,  0.1682, -0.3951,  0.1528,  0.1778],
                [-0.1593,  0.3315, -0.2286,  0.1294,  0.2087],
                [-0.3394, -0.2706,  0.1515,  0.0357, -0.4252],
                [ 0.2555, -0.4435, -0.3353,  0.2096, -0.3741],
                [ 0.3950, -0.2630, -0.1730,  0.1393,  0.3678],
                [ 0.3065, -0.0095,  0.0988,  0.4294,  0.3338]]))])

        """
        return super(ModelManager, self).load(model, **kwargs)
