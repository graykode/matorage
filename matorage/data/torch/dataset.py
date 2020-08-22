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
import torch
import tables

from matorage.data.data import MTRData


class Dataset(torch.utils.data.Dataset, MTRData):
    """
    Dataset class for Pytorch Dataset

    This class is customized for the dataset of the PyTorch, so it is operated by the following procedure.

    1. The ``_object_file_mapper`` manages the minio object as key and the downloaded local path as value. \
    When minio object is downloaded, it is recorded in ``_object_file_maper``.
    2. We read ``_object_file_mapper`` and download only new objects that are not there.
    3. ``__getitem__`` brings numpy data in local data from data index.

    Args:
        config (:obj:`matorage.DataConfig`, **require**):
            dataset configuration
        num_worker_threads (:obj:`int`, optional, defaults to `4`):
            Number of backend storage worker to upload or download.
        clear (:obj:`boolean`, optional, defaults to `True`):
            Delete all files stored on the local storage after the program finishes.
        cache_folder_path (:obj:`str`, optional, defaults to `~/.matorage`):
            Cached folder path to check which files are downloaded complete.
        index (:obj:`boolean`, optional, defaults to `False`):
            Setting for index mode.

    Examples::

        from matorage import DataConfig
        from matorage.torch import Dataset
        from torch.utils.data import DataLoader

        data_config = DataConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            dataset_name='array_test',
            attributes=[
                ('array', 'uint8', (3, 224, 224)),
            ]
        )

        dataset = Dataset(config=data_config, clear=True)

        # iterative mode
        for array in DataLoader(dataset):
            print(array)

        # index mode
        print(dataset[0])

    """

    def __init__(self, config, **kwargs):
        super(Dataset, self).__init__(config, **kwargs)
        self.open_files = {}

    def __len__(self):
        return self.end_indices[-1]

    def __getitem__(self, idx):
        if not self.index:
            return self._get_item_with_download(idx)
        else:
            return self._get_item_with_inmemory(idx)

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
                        attr_name=_attr_name,
                    )
                    if list(return_tensor[_attr_name].size()) == [1]:
                        return_tensor[_attr_name] = return_tensor[_attr_name].item()
                except:
                    raise IOError("Crash on concurrent read")

            return list(return_tensor.values())
        else:
            raise ValueError(
                "objectname({}) is not exist in {}".format(
                    _objectname, self._object_file_mapper
                )
            )

    def _exit(self):
        """
        Close all opened files and remove.

        """
        super(Dataset, self)._exit()
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

        """
        if self.index:
            raise FileNotFoundError("index mode can't not open files.")

        _driver, _driver_core_backing_store = self._set_driver()
        for _remote, _local in self._object_file_mapper.items():
            if os.path.splitext(_local)[1] != '.h5':
                continue
            _file = tables.open_file(
                _local,
                "r",
                driver=_driver,
                driver_core_backing_store=_driver_core_backing_store,
            )
            self.open_files[_remote] = {
                "file": _file,
                # "attr_names": list(_file.get_node("/")._v_children.keys()),
            }

            # Critial Bug fixed:
            # We must sort by attribute of metadata name
            self.open_files[_remote]["attr_names"] = list(self.attribute.keys())

    def _set_driver(self):
        """
        Setting HDF5 driver type

        Returns:
            :obj:`str` : HDF5 driver type string
        """

        if os.name == "posix":
            return "H5FD_SEC2", True
        elif os.name == "nt":
            return "H5FD_WINDOWS", True
        else:
            raise ValueError("{} OS not supported!".format(os.name))
