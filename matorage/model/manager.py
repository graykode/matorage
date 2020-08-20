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
    type = "model"

    def __init__(self, config, num_worker_threads=4, multipart_upload_size=5 * _MB):
        self.config = config
        self.num_worker_threads = num_worker_threads
        self.multipart_upload_size = multipart_upload_size

        self._client = (
            Minio(
                endpoint=self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                region=self.config.region,
            )
            if not check_nas(self.config.endpoint)
            else NAS(self.config.endpoint)
        )

        self._uploader = Uploader(
            client=self._client,
            bucket=self.config.bucket_name,
            num_worker_threads=self.num_worker_threads,
            multipart_upload_size=self.multipart_upload_size,
            inmemory=True,
        )

    def _uploader_closing(self):
        self._uploader.join_queue()

        _metadata_file = tempfile.mktemp("metadata.json")
        with open(_metadata_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(self.config.metadata, indent=4, sort_keys=True) + "\n"
            )

        self._client.fput_object(
            bucket_name=self.config.bucket_name,
            object_name="metadata.json",
            file_path=_metadata_file,
        )
        os.remove(_metadata_file)

    def _save_with_clear(self, model_folder, model, overwrite=False):
        if overwrite:
            objects = self._client.list_objects(
                bucket_name=self.config.bucket_name, prefix=f"{model_folder}/"
            )
            for obj in objects:
                self._client.remove_object(
                    bucket_name=self.config.bucket_name, object_name=obj.object_name
                )

        # saving model
        self._save_model(model_folder, model)
        self._uploader_closing()

    def _save_layer(self, model_folder, name, weight):
        _local_file = tempfile.mktemp(f"{name}.h5")

        _file = tables.open_file(
            _local_file, "w", driver="H5FD_CORE", driver_core_backing_store=False
        )
        _file.create_carray(
            "/", self.type, obj=weight, filters=tables.Filters(**self.config.compressor)
        )

        self._uploader.set_queue(
            local_file=_file.get_file_image(), remote_file=f"{model_folder}/{name}"
        )
        _file.close()

    def save(self, model, **kwargs):
        if not self._client.bucket_exists(self.config.bucket_name):
            self._client.make_bucket(
                self.config.bucket_name, location=self.config.region
            )

        if not isinstance(kwargs, dict):
            metadata = 0
        else:
            metadata = kwargs

        model_folder = self._hashmap_transfer(metadata)

        if model_folder in self.config.metadata["model"]:
            logger.info(
                "{} {} is already exist, so model will be overwrited.".format(
                    self.config.model_name, str(self.config.additional)
                )
            )
            self._save_with_clear(model_folder, model, overwrite=True)
        else:
            self.config.metadata["model"].update({model_folder: metadata})
            self._save_with_clear(model_folder, model)

        logger.info("model with {} is saved".format(str(metadata)))

    def load(self, model, **kwargs):
        if not isinstance(kwargs, dict):
            metadata = 0
        else:
            metadata = kwargs

        model_folder = self._hashmap_transfer(metadata)

        layers = self._client.list_objects(
            bucket_name=self.config.bucket_name,
            prefix=f"{model_folder}/",
            recursive=True,
        )

        logger.info("model with {} is loaded".format(str(metadata)))
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
            raise ValueError(
                "metadata {} is empty or not str and dict type".format(metadata)
            )

        key = json.dumps(metadata, indent=4, sort_keys=True)
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    @property
    def get_metadata(self):
        """
        Get all models according to metadata(ex. step, epoch)

        Returns:
            :obj:`dict`: model of metadata

        Examples::

            >>> model_manager.save(model, step=100)
            >>> model_manager.save(model, step=200)
            >>> model_manager.get_metadata
            {
                'additional': {'version': '1.0.1'},
                'compressor': {'complevel': 0, 'complib': 'zlib'},
                'endpoint': '127.0.0.1:9000',
                'model':
                {
                    'ad44168f1343bc77b4d9ad6f1fef50b6': {'step': 100},
                    'af0677ecf0d15d17d10204be9ff2f9f5': {'step': 200}
                },
                'model_name': 'testmodel'
            }

        """
        return self.config.metadata
