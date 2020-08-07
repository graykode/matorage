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
    type='model'

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

    def set_default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError

    def _uploader_closing(self):
        self._uploader.join_queue()

        _metadata_file = tempfile.mktemp('metadata.json')
        with open(_metadata_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(
                    self.config.metadata,
                    indent=4,
                    sort_keys=True,
                    default=self.set_default
                ) + "\n"
            )

        self._client.fput_object(
            bucket_name=self.config.bucket_name,
            object_name='metadata.json',
            file_path=_metadata_file
        )
        os.remove(_metadata_file)

    def _save_with_clear(self, model_folder, model, overwrite=False):
        if overwrite:
            objects = self._client.list_objects(
                bucket_name=self.config.bucket_name,
                prefix=f"{model_folder}/"
            )
            for obj in objects:
                self._client.remove_object(
                    bucket_name=self.config.bucket_name,
                    object_name=obj.object_name
                )

        # saving model
        self._save_model(model_folder, model)
        self._uploader_closing()

    def _save_layer(self, model_folder, name, weight):
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

        self._uploader.set_queue(
            local_file=_file.get_file_image(),
            remote_file=f"{model_folder}/{name}"
        )
        _file.close()

    def save(self, model, **kwargs):
        """
        save weight of model

        .. code-block:: python

            model = Model()
            model_manager.save(model, step=0)

        Args:
        model (:obj:`model or string`, **require**):
            Pytorch, Tensorflow model type or layer name string type.

        Returns:
            :obj: `None`:
        """
        if not self._client.bucket_exists(self.config.bucket_name):
            self._client.make_bucket(self.config.bucket_name)

        if not isinstance(kwargs, dict):
            metadata = 0
        else:
            metadata = kwargs

        model_folder = self._hashmap_transfer(metadata)

        if model_folder in self.config.metadata["model"]:
            logger.warn("{} {} is already exist, so model will be overwrited.".format(
                self.config.model_name, str(self.config.additional)
            ))
            self._save_with_clear(model_folder, model, overwrite=True)
        else:
            self.config.metadata["model"].update({model_folder : metadata})
            self._save_with_clear(model_folder, model)

    def load(self, model, **kwargs):
        """
        load weight of model

        .. code-block:: python

            >>> model = Model()
            >>> pretrained_model = model_manager.save(model, step=0)
            >>> print(pretrained_model)
            >>> Model(
                  (f): Linear(in_features=5, out_features=10, bias=True)
                )

            >>> weight = model_manager.save('fc1.weight', step=0)
            >>> print(weight)
            >>> OrderedDict([('fc1.weight', tensor([[ 0.2679, -0.2147, -0.1927, -0.3263,  0.0930],
                [ 0.0144,  0.2935,  0.3614, -0.0493, -0.3772],
                [ 0.4101, -0.1864,  0.1076, -0.3900,  0.3613],
                [-0.2831,  0.3692,  0.3367,  0.2491, -0.2971],
                [-0.3019,  0.1682, -0.3951,  0.1528,  0.1778],
                [-0.1593,  0.3315, -0.2286,  0.1294,  0.2087],
                [-0.3394, -0.2706,  0.1515,  0.0357, -0.4252],
                [ 0.2555, -0.4435, -0.3353,  0.2096, -0.3741],
                [ 0.3950, -0.2630, -0.1730,  0.1393,  0.3678],
                [ 0.3065, -0.0095,  0.0988,  0.4294,  0.3338]]))])

        Args:
        model (:obj:`model or string`, **require**):
            Pytorch, Tensorflow model type or layer name string type.

        Returns:
            :obj: `None or OrderedDict`: If ``model`` is pytorch or tensorflow model type, weight is loaded into the model and return None.
            however, If it is a string type with the name of the layer, it returns the weight of the OrderedDict type.
        """
        if not isinstance(kwargs, dict):
            metadata = 0
        else:
            metadata = kwargs

        model_folder = self._hashmap_transfer(metadata)

        layers = self._client.list_objects(
            bucket_name=self.config.bucket_name,
            prefix=f"{model_folder}/",
            recursive=True
        )

        return self._load_model(model_folder, layers, model)

    def _hashmap_transfer(self, metadata):
        """
        Get unikey object folder with `metadata` of model.

        Returns:
            :obj: `str`:
        """
        if isinstance(metadata, int):
            metadata = str(metadata)
        if not isinstance(metadata, str) and not isinstance(metadata, dict):
            raise ValueError("metadata {} is empty or not str and dict type".format(metadata))

        key = json.dumps(metadata, indent=4, sort_keys=True)
        return hashlib.md5(key.encode('utf-8')).hexdigest()