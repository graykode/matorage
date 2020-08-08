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

import os
import json
import tables
import hashlib
import tempfile
from minio import Minio

from matorage.nas import NAS
from matorage.utils import check_nas, logger
from matorage.uploader import Uploader

_KB = 1024
"""The size of a Kilobyte in bytes"""

_MB = 1024 * _KB
"""The size of a Megabyte in bytes"""

class Manager(object):
    type='optimizer'

    def __init__(self, config, num_worker_threads=4, multipart_upload_size=5 * _MB):
        self.config = config
        self.num_worker_threads = num_worker_threads
        self.multipart_upload_size = multipart_upload_size

        self._client = Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        ) if not check_nas(self.config.endpoint) else NAS(self.config.endpoint)

        self._uploader = Uploader(
            client=self._client,
            bucket=self.config.bucket_name,
            num_worker_threads=self.num_worker_threads,
            multipart_upload_size=self.multipart_upload_size,
            inmemory=True
        )

    def _uploader_closing(self):
        self._uploader.join_queue()

        _metadata_file = tempfile.mktemp('metadata.json')
        with open(_metadata_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(
                    self.config.metadata,
                    indent=4
                ) + "\n"
            )

        self._client.fput_object(
            bucket_name=self.config.bucket_name,
            object_name='metadata.json',
            file_path=_metadata_file
        )
        os.remove(_metadata_file)

    def _save_with_clear(self, step, optimizer, overwrite=False):
        if overwrite:
            objects = self._client.list_objects(
                bucket_name=self.config.bucket_name,
                prefix=f"{step}/"
            )
            for obj in objects:
                self._client.remove_object(
                    bucket_name=self.config.bucket_name,
                    object_name=obj.object_name
                )

        # saving optimizer
        self._save_optimizer(step, optimizer)
        self._uploader_closing()

    def _save_param(self, step, group, name, weight):
        _local_file = tempfile.mktemp(f"{name}.h5")

        _file = tables.open_file(
            _local_file, 'w',
            driver='H5FD_CORE',
            driver_core_backing_store=False
        )
        _file.create_carray(
            '/', self.type, obj=weight,
            filters=tables.Filters(**self.config.compressor)
        )

        if group:
            self._uploader.set_queue(
                local_file=_file.get_file_image(),
                remote_file=f"{step}/{group}/{name}"
            )
        else:
            self._uploader.set_queue(
                local_file=_file.get_file_image(),
                remote_file=f"{step}/{name}"
            )
        _file.close()

    def save(self, optimizer):
        """
        save weight of optimizer

        .. code-block:: python

            model = Model()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            optimizer_manager.save(optimizer, step=0)

        Args:
        optimizer (:obj:`torch.optim or tf.keras.optimizers`, **require**):
            Pytorch or Tensorflow optimizer.

        Returns:
            :obj: `None`:
        """
        if not self._client.bucket_exists(self.config.bucket_name):
            self._client.make_bucket(self.config.bucket_name)

        step = self._get_step(optimizer)
        if not step:
            logger.error("{} {} step({})is not exist".format(
                self.config.optimizer_name,
                self.config.additional, str(step)
            ))
            return

        if step in self.config.metadata["optimizer"]:
            logger.warn("{} {} is already exist, so optimizer will be overwrited.".format(
                self.config.optimizer_name, str(self.config.additional)
            ))
            self._save_with_clear(step, optimizer, overwrite=True)
        else:
            self._set_metadata(
                metadata=self.config.metadata,
                optimizer=optimizer,
                step=step
            )
            self._save_with_clear(step, optimizer)

    def load(self, optimizer, step):
        """
        load weight of optimizer

        .. code-block:: python

            >>> model = Model()
            >>> optimizer = optim.Adam(model.parameters(), lr=0.01)
            >>> optimizer_manager.load(optimizer, step=938)
            >>> print(optimizer)
            >>>

        Args:
        optimizer (:obj:`torch.optim or tf.keras.optimizers`, **require**):
            Pytorch or Tensorflow optimizer.
        step (:obj:`integer`, **require**):
            optimizer step.

        Returns:
            :obj: `None or OrderedDict`: If ``optimizer`` is pytorch or tensorflow optimizer type,
            weight is loaded into the optimizer and return None.
            however, If it is a string type with the name of the layer, it returns the weight of the OrderedDict type.
        """

        layers = self._client.list_objects(
            bucket_name=self.config.bucket_name,
            prefix=f"{step}/",
            recursive=True
        )

        return self._load_model(step, layers, optimizer)

    @property
    def get_optimizer(self):
        return self.config.metadata["optimizer"]