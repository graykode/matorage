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
import bisect
import tempfile
from minio import Minio

from torch.utils.data import Dataset

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
    def __init__(self, config, download=True, clear=True):
        self.config = config
        self.download = False if config.batch_atomic or config.inmemory else download
        self.clear = clear

        self.reindexer = self._merge_metadata(
            bucket_name=self.config.bucket_name
        )
        self.ends = list(self.reindexer.keys())

        self._clients = {}

        self.cnt = 0

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, idx):
        _filename, _index = self._fine_file(idx)
        _local_filename = tempfile.mktemp(_filename)

        # create minio client once in a process to avoid thread unsafe
        _pid = os.getpid()
        if _pid not in self._clients:
            self._clients[_pid] = self._create_client()

        if self.download and not os.path.exists(_local_filename):
            self._clients[_pid].fget_object(
                bucket_name=self.config.bucket_name,
                object_name=_filename,
                file_path=_local_filename
            )
            # to read file, blocking until download finish.

        else:
            _object = self._clients[_pid].get_object(
                bucket_name=self.config.bucket_name,
                object_name=_filename
            )
            # read in-memory

        return 0

    def _create_client(self):
        return Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        )

    def _fine_file(self, index):
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