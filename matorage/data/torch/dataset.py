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
import time
import torch
import atexit
import tables
import bisect
import tempfile
from minio import Minio
from os.path import expanduser
from multiprocessing import Manager, Barrier

from torch.utils.data import Dataset

from matorage.nas import NAS
from matorage.utils import check_nas
from matorage.data.downloader import DataDownloader
from matorage.utils import logger

class MTRDataset(Dataset):
    r"""MTRDataset class for Pytorch Dataset

        This class is customized for the dataset of the PyTorch, so it is operated by the following procedure.
        1. The `_object_file_mapper` manages the minio object as key and the downloaded local path as value.
            {'tmpv7sy5_1fff7845eccd874068.h5': '/tmp/tmpja6wo221tmpv7sy5_1fff7845eccd874068.h5'}
            When minio object is downloaded, it is recorded in _object_file_maper.
        2. We read `_object_file_mapper` and download only new objects that are not there.
        3. `__getitem__` brings numpy data in local data from data index.

        Args:
            config (:obj:`matorage.config.MTRConfig`, `require`):
            num_worker_threads :obj:`int`, `optional`, defaults to `4`):
                    number of backend storage worker to upload or download.
            clear (:obj:`boolean`, `optional`, defaults to `True`):
                Delete all files stored on the local storage after the program finishes.

        HDF5 Options
            inmemory (:obj:`bool`, `optional`, defaults to `False`):
                If you use this value as `True`, then you can use `HDF5_CORE` driver (https://support.hdfgroup.org/HDF5/doc/TechNotes/VFL.html#TOC1)
                so the temporary file for uploading or downloading to backend storage,
                such as MinIO, is not stored on disk but is in the memory.
                Keep in mind that using memory is fast because it doesn't use disk IO, but it's not always good.
                If default option(False), then `HDF5_SEC2` driver will be used on posix OS(or `HDF5_WINDOWS` in Windows).

    """
    def _init_download(self):
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

        with open(self.cache_path, "w") as f:
            json.dump(self._object_file_mapper, f)

    def _pre_open_files(self):
        """
        pre-open file for each processes.
        This function call from individuallly all processes.
        Because in Pytorch Multi Processing of DataLoader use `fork` mode.

        Returns:
            :None
        """
        for _remote, _local in self._object_file_mapper.items():
            _file = tables.open_file(_local, 'r')
            self.open_files[_remote] = {
                "file" : _file,
                "attr_names" : list(_file.get_node("/")._v_children.keys())
            }

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
            logger.info('All dataset downloaded done.')
        self.open_files = {}

        assert len(self._object_file_mapper) == len(self.reindexer)
        atexit.register(self._exit)

    def _exit(self):
        for _file in list(self.open_files.values()):
            _file["file"].close()
        if self.clear:
            for _local_file in list(self._object_file_mapper.values()):
                if os.path.exists(_local_file):
                    os.remove(_local_file)
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)

    def __len__(self):
        return self.end_indices[-1]

    def __getitem__(self, idx):
        if not self.open_files:
            self._pre_open_files()

        _objectname, _relative_index = self._find_object(idx)
        if _objectname in self._object_file_mapper:
            _open_file = self.open_files[_objectname]
            _file = _open_file["file"]
            _attr_names = _open_file["attr_names"]

            return_tensor = {}
            for _attr_name in _attr_names:
                try:
                    return_tensor[_attr_name] = self._reshape_convert_tensor(
                        numpy_array=_file.root[_attr_name][_relative_index],
                        attr_name=_attr_name
                    )
                    if list(return_tensor[_attr_name].size()) == [1]:
                        return_tensor[_attr_name] = return_tensor[_attr_name].item()
                except:
                    raise IOError("Crash on concurrent read")

            return list(return_tensor.values())
        else:
            raise ValueError("objectname({}) is not exist in {}".format(
                _objectname, self._object_file_mapper
            ))

    def _create_client(self):
        return Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        ) if not check_nas(self.config.endpoint) else NAS(self.config.endpoint)

    def _find_object(self, index):
        """
        find filename by index with binary search algorithm(indexes had been sorted).

        Returns:
            :obj:`str`: filename for index
        """
        _key_idx = bisect.bisect_right(self.end_indices, index)
        _key = self.end_indices[_key_idx]
        _last_key = self.end_indices[_key_idx - 1] if _key_idx else 0
        _relative_index = (index - _last_key)
        return self.reindexer[_key], _relative_index

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

    def _reshape_convert_tensor(self, numpy_array, attr_name):
        """
        Reshape numpy tensor and convert from numpy to torch tensor.
        In matorage dataset save in 2D (bz, N) shape to cpu L1 cache manage dataset fast.
        Therefore, this function restores the shape for the user to use.

        Returns:
            :obj:`torch.tensor`
        """
        _shape = self.attribute[attr_name]["shape"]
        numpy_array = numpy_array.reshape(_shape)
        tensor = torch.from_numpy(numpy_array)
        return tensor