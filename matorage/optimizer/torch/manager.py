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
import h5py
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from torch.optim.optimizer import Optimizer

from matorage.optimizer.manager import Manager


class OptimizerManager(Manager):

    """
    Optimizer Manager Pytorch classes. This class overrides ``Manager``.

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

        from matorage.torch import OptimizerManager
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
        optimizer = optim.Adam(Model().parameters(), lr=0.01)
        optimizer_manager.save(optimizer, step=100)

    """

    def __init__(self, config, num_worker_threads=4, multipart_upload_size=5 * _MB):
        super(OptimizerManager, self).__init__(
            config, num_worker_threads, multipart_upload_size
        )

    def _get_step(self, optimizer):
        state = optimizer.state_dict()["state"]
        if state:
            step = list(state.values())[0]["step"]
            return step
        else:
            return None

    def _set_metadata(self, metadata, optimizer, step):
        assert isinstance(optimizer, Optimizer)
        optimizer = optimizer.state_dict()
        metadata["optimizer"].update(
            {
                str(step): {
                    "framework": "pytorch",
                    "param_groups": optimizer["param_groups"],
                }
            }
        )

    def _set_scheduler(self, metadata, scheduler, step):
        assert isinstance(scheduler, dict)
        metadata["scheduler"].update({
            str(step): scheduler
        })

    def _save_optimizer(self, step, optimizer):
        assert isinstance(optimizer, Optimizer)
        state = optimizer.state_dict()["state"]
        for param_name, param_dict in state.items():
            for param_dict_key, param_dict_value in param_dict.items():
                if torch.is_tensor(param_dict_value):
                    param_dict_value = param_dict_value.cpu().numpy()
                elif isinstance(param_dict_value, int):
                    param_dict_value = np.asarray([param_dict_value])
                self._save_param(
                    step, group=param_name, name=param_dict_key, weight=param_dict_value
                )

    def _load_optimizer(self, step, layers, optimizer):
        assert isinstance(optimizer, Optimizer)

        weight = OrderedDict()
        step = str(step)

        if step not in self.config.metadata["optimizer"]:
            raise KeyError(
                "Available only in {}".format(
                    list(self.config.metadata["optimizer"].keys())
                )
            )
        weight["param_groups"] = self.config.metadata["optimizer"][step]["param_groups"]
        weight["state"] = defaultdict(lambda: defaultdict())

        for layer in layers:
            name = layer.object_name
            layer_image = self._client.get_object(
                bucket_name=self.config.bucket_name, object_name=name
            ).read()

            layer_image = h5py.File(io.BytesIO(layer_image), "r")
            step, param_name, param_dict_key = name.split("/")
            if layer_image[self.type][:].shape == (1,):
                value = layer_image[self.type][0]
            else:
                value = torch.from_numpy(layer_image[self.type][:])
            weight["state"][param_name][param_dict_key] = value

        optimizer.load_state_dict(weight)

    def save(self, optimizer, scheduler=None):
        """
        save weight of optimizer

        Args:
            optimizer (:obj:`torch.optim`, **require**):
                Pytorch optimizer.

        Examples::

            >>> model = Model()
            >>> optimizer = optim.Adam(model.parameters(), lr=0.01)
            # model training...
            >>> optimizer_manager.save(optimizer)

        """
        if scheduler:
            scheduler = scheduler.state_dict()
        super(OptimizerManager, self).save(optimizer, scheduler=scheduler)

    def load(self, optimizer, step):
        """
        load weight of optimizer

        Args:
            optimizer (:obj:`torch.optim`, **require**):
                Pytorch optimizer.
            step (:obj:`integer`, **require**):
                optimizer step.

        Examples::

            >>> optimizer_manager = OptimizerManager(config=optimizer_config)
            >>> optimizer = optim.Adam(model.parameters(), lr=0.01)
            >>> optimizer_manager.load(optimizer, step=938)

        """
        super(OptimizerManager, self).load(optimizer, step)

    def load_with_scheduler(self, optimizer, scheduler, step):
        """
        load weight of optimizer and scheduler

        Args:
            optimizer (:obj:`torch.optim`, **require**):
                Pytorch optimizer.
            scheduler (:obj:`torch.optim.lr_scheduler`, **require**):
                Pytorch scheduler.
            step (:obj:`integer`, **require**):
                optimizer step.

        Examples::

            >>> optimizer_manager = OptimizerManager(config=optimizer_config)
            >>> optimizer = optim.Adam(model.parameters(), lr=0.01)
            >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            >>> optimizer_manager.load_with_scheduler(optimizer, scheduler, step=938)

        """
        super(OptimizerManager, self).load(optimizer, step)

        step = str(step)
        if step in self.config.metadata["scheduler"]:
            scheduler.load_state_dict(self.config.metadata["scheduler"][step])
        else:
            raise KeyError(
                "Available only in {}".format(
                    list(self.config.metadata["scheduler"].keys())
                )
            )