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
import atexit
import tempfile
from minio import Minio
from os.path import expanduser

from matorage.nas import NAS
from matorage.utils import logger, check_nas
from matorage.data.downloader import DataDownloader

class MTRData(object):

    def __init__(self, config, num_worker_threads=4, clear=True, inmemory=False, cache_folder_path='~/.matorage'):
        self.config = config
        self.attribute = self._set_attribute()

        # Storage configuration
        self.num_worker_threads = num_worker_threads

        self.download = False if (config.batch_atomic or inmemory) else True
        self.clear = False if not self.download else clear

        self.cache_folder_path = expanduser(cache_folder_path)
        if not os.path.exists(self.cache_folder_path):
            os.makedirs(self.cache_folder_path)

        self.cache_path = f"{os.path.join(self.cache_folder_path, self.config.bucket_name)}.json"
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                self._object_file_mapper = json.load(f)
        else:
            self._object_file_mapper = {}

        self.reindexer = self._merge_metadata(
            bucket_name=self.config.bucket_name
        )
        self.end_indices = list(self.reindexer.keys())

        if self.download:
            self._init_download()
        self.open_files = {}

        assert len(self._object_file_mapper) == len(self.reindexer)
        atexit.register(self._exit)

    def _create_client(self):
        return Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        ) if not check_nas(self.config.endpoint) else NAS(self.config.endpoint)

    def _init_download(self):
        """
        Download all object from bucket with multi thread.
        cache to `_object_file_mapper` downloaded object paths.

        Returns:
            :obj: `None`:
        """
        _client = self._create_client()
        _downloader = DataDownloader(
            client=_client,
            bucket=self.config.bucket_name,
            num_worker_threads=self.num_worker_threads
        )

        _remote_files = list(self.reindexer.values())
        for _remote_file in _remote_files:
            _local_file = tempfile.mktemp(_remote_file)
            if _remote_file not in self._object_file_mapper:
                self._object_file_mapper[_remote_file] = _local_file
                _downloader.set_queue(local_file=_local_file, remote_file=_remote_file)
        _downloader.join_queue()

        if not os.path.exists(self.cache_path):
            with open(self.cache_path, "w") as f:
                json.dump(self._object_file_mapper, f)
            logger.info('All {} {} datasets are downloaded done.'.format(
                self.config.dataset_name, str(self.config.additional)
            ))

    def _exit(self):
        """
        Close all opened files and remove.

        Returns:
            :obj: `None`:
        """

        for _file in list(self.open_files.values()):
            if _file["file"].isopen:
                _file["file"].close()

        if self.clear:
            for _local_file in list(self._object_file_mapper.values()):
                if os.path.exists(_local_file):
                    os.remove(_local_file)
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)

    def _merge_metadata(self, bucket_name):
        """
        merge splited metadatas to a one file.

        Returns:
            :obj:`dict` : last end indexes with filename
            {
                3335: 'tmpajivq0tw0923909106de4222.h5',
                6670: 'tmp1g5zxyl0576b788d259844d1.h5',
                10005: 'tmpqnkklb9u27395376c94d4c14.h5'
            }

        """
        client = self._create_client()
        objects = client.list_objects(
            bucket_name,
            prefix='metadata/'
        )

        total_index = []
        for obj in objects:
            metadata = client.get_object(
                bucket_name,
                object_name=obj.object_name
            )
            local_indexer = json.loads(metadata.read().decode('utf-8'))["indexer"]
            total_index.extend(list(local_indexer.values()))

        reindexer = {}
        for _index in total_index:
            key = list(reindexer.keys())[-1] + _index["length"] if reindexer else _index["length"]
            reindexer[key] = _index["name"]

        return reindexer

    def _set_attribute(self):
        """
        Set `attribute` dictionary.

        Returns:
            :obj:`dict` : attribute
            {
                'image': {'shape': (28, 28), 'type': 'uint8'},
                'target': {'shape': (1,), 'type': 'uint8'}
            }
        """
        _attributes = {}
        _metadata_attributes = self.config.metadata.attributes
        for _attr in _metadata_attributes:
            _attributes[_attr.name] = {
                "shape" : _attr.shape,
                "type" : str(_attr.type.type)
            }
        return _attributes