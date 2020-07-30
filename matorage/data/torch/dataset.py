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
import torch
import tables
import bisect
import h5py
import tempfile
from torch.utils.data import Dataset

from matorage.data.data import MTRData

class MTRDataset(Dataset, MTRData):
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

    def __init__(self, config, num_worker_threads=4, clear=True, cache_folder_path='~/.matorage'):
        super(MTRDataset, self).__init__(config, num_worker_threads, clear, cache_folder_path)
        self.end_indices = list(self.merged_indexer.keys())
        self.open_files = {}

        self._clients = {}

    def __len__(self):
        return self.end_indices[-1]

    def __getitem__(self, idx):
        if self.download:
            return self._get_item_with_download(idx)
        else:
            return self._get_item_with_inmemory(idx)

    def _get_item_with_inmemory(self, idx):
        _pid = os.getpid()
        if _pid not in self._clients:
            self._clients[_pid] = self._create_client()

        _objectname, _relative_index = self._find_object(idx)
        _file_image = self._clients[_pid].get_object(
            self.config.bucket_name,
            object_name=_objectname
        ).read()
        _file_image = h5py.File(io.BytesIO(_file_image),'r')

        return_tensor = {}
        for _attr_name in list(self.attribute.keys()):
            try:
                return_tensor[_attr_name] = self._reshape_convert_tensor(
                    numpy_array=_file_image[_attr_name][_relative_index],
                    attr_name=_attr_name
                )
                if list(return_tensor[_attr_name].size()) == [1]:
                    return_tensor[_attr_name] = return_tensor[_attr_name].item()
            except:
                raise IOError("Crash on concurrent read")

        return list(return_tensor.values())

    def _get_item_with_download(self, idx):
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
        return self.merged_indexer[_key], _relative_index

    def _exit(self):
        """
        Close all opened files and remove.

        Returns:
            :obj: `None`:
        """
        super(MTRDataset, self)._exit()
        for _file in list(self.open_files.values()):
            if _file["file"].isopen:
                _file["file"].close()

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

    def _pre_open_files(self):
        """
        pre-open file for each processes.
        This function call from individuallly all processes.
        Because in Pytorch Multi Processing of DataLoader use `fork` mode.

        Returns:
            :None
        """
        _driver, _driver_core_backing_store = self._set_driver()
        for _remote, _local in self._object_file_mapper.items():
            _file = tables.open_file(
                _local, 'r',
                driver=_driver,
                driver_core_backing_store=_driver_core_backing_store
            )
            self.open_files[_remote] = {
                "file" : _file,
                "attr_names" : list(_file.get_node("/")._v_children.keys())
            }

    def _set_driver(self):
        """
        Setting HDF5 driver type

        Returns:
            :obj:`str` : HDF5 driver type string
        """
        if self.config.batch_atomic:
            return 'H5FD_CORE', False
        else:
            if os.name == "posix":
                return 'H5FD_SEC2', True
            elif os.name == "nt":
                return 'H5FD_WINDOWS', True
            else:
                raise ValueError("{} OS not supported!".format(os.name))