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
import io
import json
import atexit
import bisect
import tempfile
from minio import Minio
from os.path import expanduser

from matorage.nas import NAS
from matorage.utils import logger, check_nas
from matorage.downloader import Downloader


class MTRData(object):
    r"""Parent Dataset class for Tensorflow and Pytorch Dataset

        This class is customized for the dataset of the Tensorflow, so it is operated by the following procedure.
        1. The `_object_file_mapper` manages the minio object as key and the downloaded local path as value.
            {'tmpv7sy5_1fff7845eccd874068.h5': '/tmp/tmpja6wo221tmpv7sy5_1fff7845eccd874068.h5'}
            When minio object is downloaded, it is recorded in _object_file_maper.
        2. We read `_object_file_mapper` and download only new objects that are not there.
        3. if Tensorflow v2(2.2.0>=), we use `tfio.IODataset.from_hdf5` and parallel `interleave` more fast

        Args:
            config (:obj:`matorage.DataConfig`, `require`):
            num_worker_threads :obj:`int`, `optional`, defaults to `4`):
                    number of backend storage worker to upload or download.
            clear (:obj:`boolean`, `optional`, defaults to `True`):
                Delete all files stored on the local storage after the program finishes.
            cache_folder_path (:obj:`str`, `optional`, defaults to `~/.matorage`):
                cached folder path to check which files are downloaded complete.
            index (:obj:`boolean`, `optional`, defaults to `False`):
                setting for index mode.
    """

    def __init__(
        self,
        config,
        num_worker_threads=4,
        clear=True,
        cache_folder_path="~/.matorage",
        index=False,
    ):
        self.config = config
        self.attribute = self._set_attribute()

        # Storage configuration
        self.num_worker_threads = num_worker_threads
        self.clear = clear
        self.index = index

        self._check_bucket()

        # merge all metadatas and load in memory.
        self.merged_indexer, self.merged_filetype = self._merge_metadata()
        self.end_indices = list(self.merged_indexer.keys())

        self._clients = {}

        if not self.index:
            # cache object which is downloaded.
            if not check_nas(self.config.endpoint):
                self._caching(cache_folder_path=cache_folder_path)
            else:
                self._object_file_mapper = {}

            # download all object in /tmp folder
            self._init_download()

            atexit.register(self._exit)

    def _caching(self, cache_folder_path):
        self.cache_folder_path = expanduser(cache_folder_path)
        if not os.path.exists(self.cache_folder_path):
            os.makedirs(self.cache_folder_path)

        self.cache_path = (
            f"{os.path.join(self.cache_folder_path, self.config.bucket_name)}.json"
        )
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                self._object_file_mapper = json.load(f)
        else:
            self._object_file_mapper = {}

    def _check_bucket(self):
        _client = self._create_client()
        if not _client.bucket_exists(self.config.bucket_name):
            raise ValueError(
                "dataset {} with {} is not exist".format(
                    self.config.dataset_name, str(self.config.additional)
                )
            )

    def _create_client(self):
        return (
            Minio(
                endpoint=self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                region=self.config.region
            )
            if not check_nas(self.config.endpoint)
            else NAS(self.config.endpoint)
        )

    def _find_object(self, index):
        """
        find filename by index with binary search algorithm(indexes had been sorted).

        Returns:
            :obj:`str`: filename for index
        """
        _key_idx = bisect.bisect_right(self.end_indices, index)
        _key = self.end_indices[_key_idx]
        _last_key = self.end_indices[_key_idx - 1] if _key_idx else 0
        _relative_index = index - _last_key
        return self.merged_indexer[_key], _relative_index

    def _get_item_with_inmemory(self, idx):
        import h5py

        _pid = os.getpid()
        if _pid not in self._clients:
            self._clients[_pid] = self._create_client()

        _objectname, _relative_index = self._find_object(idx)
        _file_image = (
            self._clients[_pid]
            .get_object(self.config.bucket_name, object_name=_objectname)
            .read()
        )
        _file_image = h5py.File(io.BytesIO(_file_image), "r")

        return_tensor = {}
        for _attr_name in list(self.attribute.keys()):
            try:
                return_tensor[_attr_name] = self._reshape_convert_tensor(
                    numpy_array=_file_image[_attr_name][_relative_index],
                    attr_name=_attr_name,
                )
            except:
                raise IOError("Crash on concurrent read")

        return list(return_tensor.values())

    def _init_download(self):
        """
        Download all object from bucket with multi thread.
        cache to `_object_file_mapper` downloaded object paths.

        """
        _client = self._create_client()
        _downloader = Downloader(
            client=_client,
            bucket=self.config.bucket_name,
            num_worker_threads=self.num_worker_threads,
        )

        _remote_files = list(self.merged_indexer.values()) + list(self.merged_filetype)
        for _remote_file in _remote_files:
            if not check_nas(self.config.endpoint):
                _local_file = tempfile.mktemp(_remote_file)
                if _remote_file not in self._object_file_mapper:
                    self._object_file_mapper[_remote_file] = _local_file
                    _downloader.set_queue(
                        local_file=_local_file, remote_file=_remote_file
                    )
            else:
                if _remote_file not in self._object_file_mapper:
                    self._object_file_mapper[_remote_file] = os.path.join(
                        self.config.endpoint, self.config.bucket_name, _remote_file
                    )
        _downloader.join_queue()

        assert len(self._object_file_mapper) == (len(self.merged_indexer) + len(self.merged_filetype))

        if not check_nas(self.config.endpoint) and not os.path.exists(self.cache_path):
            with open(self.cache_path, "w") as f:
                json.dump(self._object_file_mapper, f)
            logger.info(
                "All {} {} datasets are downloaded done.".format(
                    self.config.dataset_name, str(self.config.additional)
                )
            )

    def _exit(self):
        """
        Close all opened files and remove.

        """

        if self.clear and not check_nas(self.config.endpoint):
            for _local_file in list(self._object_file_mapper.values()):
                if os.path.exists(_local_file):
                    os.remove(_local_file)
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)

    def _merge_metadata(self):
        """
        merge splited metadatas to a one file.
        +) merge dataset of filetype list

        Returns:
            :obj:`dict` : last end indexes with filename
            {
                3335: 'tmpajivq0tw0923909106de4222.h5',
                6670: 'tmp1g5zxyl0576b788d259844d1.h5',
                10005: 'tmpqnkklb9u27395376c94d4c14.h5'
            }

        """
        client = self._create_client()
        objects = client.list_objects(self.config.bucket_name, prefix="metadata/")

        total_index = []
        filetypes = []
        for obj in objects:
            metadata = client.get_object(
                self.config.bucket_name, object_name=obj.object_name
            )
            metadata = json.loads(metadata.read().decode("utf-8"))
            local_indexer = metadata["indexer"]
            total_index.extend(list(local_indexer.values()))
            filetypes.extend(metadata["filetype"])

        reindexer = {}
        for _index in total_index:
            key = (
                list(reindexer.keys())[-1] + _index["length"]
                if reindexer
                else _index["length"]
            )
            reindexer[key] = _index["name"]

        return reindexer, filetypes

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
                "shape": _attr.shape,
                "type": str(_attr.type.type),
            }
        return _attributes

    @property
    def get_filetype_list(self):
        """
        Get list of filetype dataset in bucket of ``DataConfig``

        Returns:
            :obj: `list`: list of key of filetype dataset in bucket of ``DataConfig``
        """
        return self.merged_filetype

    def get_filetype_from_key(self, filename):
        """
        Download filetype dataset from key in bucket of ``DataConfig``

        Args:
            filename (:obj:`string`):
                file name

        Returns:
            :obj: `string`: local path of downloaded file in bucket of ``DataConfig``
        """

        assert filename in self._object_file_mapper
        return self._object_file_mapper[filename]