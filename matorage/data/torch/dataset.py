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
import tables
import bisect
import tempfile
from minio import Minio
from multiprocessing import Manager

from torch.utils.data import Dataset

from matorage.nas import NAS
from matorage.utils import check_nas

class MTRDataset(Dataset):
    r"""MTRDataset class for Pytorch Dataset

        Note:


        Args:
            config (:obj:`matorage.config.MTRConfig`, `require`):
            download (:obj:`boolean`, `optional`, defaults to `True`):
                download h5 file in local temp folder.
                if using `batch_atomic` or `inmemory` options, then set it `False`
                Setting this option allows you to occupy a lot of local storage volumes, but speeds up
                by not downloadingthem again twice for the next epoch or unused index.
            clear (:obj:`boolean`, `optional`, defaults to `True`):
                Delete all files stored on the local storage after the program finishes.

    """
    def __init__(self, config, inmemory=False, clear=True):
        self.config = config
        self.download = False if (config.batch_atomic or inmemory) else True
        self.clear = False if not self.download else clear

        self.reindexer = self._merge_metadata(
            bucket_name=self.config.bucket_name
        )
        self.ends = list(self.reindexer.keys())

        self._clients = {}

        # To avoid processes race condition.
        _manager = Manager()
        self._worker_job = dict()
        self._object_file_mapper = _manager.dict()

        self.num_worker = -1

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, idx):
        if self.num_worker == -1:
            self._set_num_worker()
            self._balance_worker()
            self._create_clients()

        _worker_info = torch.utils.data.get_worker_info()
        if _worker_info is None:
            worker_id = 0
        else:
            worker_id = _worker_info.id

        _objectname, _index = self._find_object(idx)

        # memoization : if _filename is in local file or in-memory return it.
        if _objectname in self._object_file_mapper:
            self._get_from_local_file_or_memory()
            return 0

        print(worker_id, _objectname, self._object_file_mapper)
        while _objectname not in self._object_file_mapper and \
                self._worker_job[_objectname] != worker_id:
            time.sleep(0.00001)

        if self.download:
            if _objectname not in self._object_file_mapper:
                _local_filename = tempfile.mktemp(_objectname)
                self._clients[worker_id].fget_object(
                    bucket_name=self.config.bucket_name,
                    object_name=_objectname,
                    file_path=_local_filename
                )
                self._object_file_mapper[_objectname] = _local_filename
                print(worker_id, _objectname, 'done')

            # to read file, blocking until download finish.

            return 0
        else:
            # if not set `batch_atomic` first find in lru cache.
            _object = self._clients[worker_id].get_object(
                bucket_name=self.config.bucket_name,
                object_name=_filename
            )

            return 0

    def _balance_worker(self):
        _files = list(self.reindexer.values())
        _idx = 0
        for _file in _files:
            if _idx == self.num_worker:
                _idx = 0
            self._worker_job[_file] = _idx

    def _create_client(self):
        return Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        ) if not check_nas(self.config.endpoint) else NAS(self.config.endpoint)

    def _create_clients(self):
        # create minio client once in a process to avoid thread unsafe
        for w in range(self.num_worker):
             self._clients[w] = self._create_client()

    def _find_object(self, index):
        """
        find filename by index with binary search algorithm(indexes had been sorted).

        Returns:
            :obj:`str`: filename for index
        """
        _key_idx = bisect.bisect_right(self.ends, index)
        _key = self.ends[_key_idx]
        _last_key = self.ends[_key_idx - 1] if _key_idx else 0
        _relative_index = (index - _last_key)
        return self.reindexer[_key], _relative_index

    def _get_from_local_file_or_memory(self):
        pass

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

    def _set_num_worker(self):
        _worker_info = torch.utils.data.get_worker_info()
        if _worker_info is None:
            self.num_worker = 0
        else:
            self.num_worker = _worker_info.num_workers