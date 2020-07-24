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
import sys
import uuid
import atexit
import psutil
import datetime
import tempfile
import tables as tb
import numpy as np
from time import sleep
from functools import reduce
from minio import Minio

from matorage.utils import is_tf_available, is_torch_available
from matorage.data.config import DataConfig
from matorage.data.uploader import DataUploader

class DataSaver(object):
    r""" Dataset saver classes.
        This class is initialized independently of the process and goes through the process of uploading
        the numby data created through the pre-processing process to the MinIO.
        Update the file, push the upload queue if it exceeds a certain size, close the file, and create a new file.
        After saving, you should disconnect the data saver.

        To make This procedure easier to understand, the following is written in the pseudo-code.
            ```python
            per_one_batch_data_size = array_size // batch size
            per_one_file_batch_size = max(1, self.config.max_object_size // per_one_batch_data_size)
            for batch_idx in range(bzs):
                if get_current_stored_batch_size() < per_one_file_batch_size:
                    file.append(data[batch_idx])
                else:
                    file_closing()
                    new_file is opened
                    new_file.append(data[batch_idx])
            All files are closed.
            ```

        Note:
            - Deep Learning Framework Type : All(pure python is also possible)
            - **All processes should call the constructors of this class independently.**

        Args:
            config (:obj:`matorage.config.MTRConfig`, `require`):

        Example::
            Single Process example
                ```python
                data_saver = DataSaver(config=data_config)
                row = 100
                data = np.random.rand(64, 3, 224, 224)

                start = time.time()

                for _ in tqdm(range(row)):
                    preprocessing_work()
                    data_saver({
                        'array' : data
                    })

                data_saver.disconnect()
                ```
    """

    def __init__(self, config):

        self.config = config
        self.filter = tb.Filters(**config.compressor)

        self._filelist = []
        self._file, self._earray = self._get_newfile()

        self._disconnected = False

        self._client = Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        )
        self._uploader = DataUploader(
            client=self._client,
            bucket=self.config.bucket_name,
            num_worker_threads=self.config.num_worker_threads,
            inmemory=self.config.inmemory
        )

        atexit.register(self._exit)

    def _append_all(self):
        """
        append all array in `name` node.
        datas is `dict` type. key is `str`, value is `numpy.ndarray`
        **`value` is `numpy.ndarray` type with (B, *) shape, B means batch size**
        example:
            {
                'image' : np.random.rand(16, 28, 28),
                'target' : np.random.rand(16)
            }

        Returns:
            :None
        """
        array_size = self._get_array_size()
        bzs = list(self._datas.values())[0].shape[0]

        per_one_batch_data_size = array_size // bzs
        per_one_file_batch_size = max(
            int(self.atomic),
            self.config.max_object_size // per_one_batch_data_size
        )

        for batch_idx in range(bzs):
            if self._get_current_stored_batch_size() < per_one_file_batch_size:
                for name, array in self._datas.items():
                    self._earray[name].append(array[batch_idx, None])
            else:
                self._file_closing()
                self._file, self._earray = self._get_newfile()
                for name, array in self._datas.items():
                    self._earray[name].append(array[batch_idx, None])

    def _check_attr_name(self, name):
        """
        check attribute names is exist

        Returns:
            :None
        """
        if name not in self._earray.keys():
            raise KeyError("attribute name {} is not exist!".format(name))

    def _check_datas(self):
        """
        Check data dictionary

        Returns:
            :None
        """

        if not isinstance(self._datas, dict):
            raise TypeError("datas shoud be dict type.", self.__call__.__doc__)

        bzs = 0
        for name, array in self._datas.items():
            self._check_attr_name(name=name)

            if is_tf_available() and not isinstance(array, np.ndarray):
                array = array.numpy()
            if is_torch_available() and not isinstance(array, np.ndarray):
                array = array.numpy()

            assert isinstance(array, np.ndarray), "array type is not `numpy.ndarray`"

            if bzs:
                if bzs != array.shape[0]:
                    raise ValueError("each datas array batch sizes are not same.")
            else:
                bzs = array.shape[0]

            # This resape is made into a (B, *) shape.
            # Shape is lowered to two contiguous dimensions, enabling IO operations to operate very quickly.
            # https://www.slideshare.net/HDFEOS/caching-and-buffering-in-hdf5#25
            if len(array.shape) == 1:
                # this array is ground truth
                array = array.reshape(-1, 1)

            self._datas[name] = array.reshape(-1, reduce(lambda x, y: x * y, array.shape[1:]))

    def __call__(self, datas):
        """
        datas is `dict` type. key is `str`, value is `numpy.ndarray`
        **`value` is `numpy.ndarray` type with (B, *) shape, B means batch size**
        example:
            {
                'image' : np.random.rand(16, 28, 28),
                'target' : np.random.rand(16)
            }

        Note:
            file size will be closed and uploaded if bigger than `max_object_size`

        Returns:
            :None
        """
        self._disconnected = False

        self._datas = datas

        self._check_datas()

        self._append_all()

    def _file_closing(self):
        _length = len(list(self._earray.values())[0])
        _last_index = self.config.get_indexer_last

        if not self.config.inmemory:
            self._file.close()
            self._uploader.set_queue(self._file.filename, self._filename)
        else:
            self._uploader.set_queue(self._file.get_file_image(), self._filename)
            self._file.close()
        # Set filename indexer
        _current_index = _last_index + _length
        self.config.set_indexer({
            _current_index : {
                "name" : os.path.basename(self._filename),
                "length" : _length
            }
        })

    def _create_name(self, length=16):
        return tempfile.mktemp("{}.h5".format(uuid.uuid4().hex[:length]))

    def _exit(self):
        self._file.close()
        self._disconnected = True

    def _get_array_size(self):
        """
        Get size of all array .

        Returns:
            :obj:`datas size(bytes)`
        """
        size = 0
        for name, array in self._datas.items():
            size += array.nbytes
        return size

    def _get_current_stored_batch_size(self):
        """
        Get current file stored batch size

        Returns:
            :obj:`integer`: current stored batch size in a opened file.
        """
        return len(list(self._earray.values())[0])

    def _get_newfile(self):
        """
        Get new file inode and it's attribute

        Returns:
            :obj:`tuple(tables.File, dict)`
            second item is pytable's attribute
            {
                'name1' : tables.EArray, 'name2' : tables.EArray
            }
        """
        _driver, _driver_core_backing_store = self._set_driver()

        self._filename = self._create_name()
        self._filelist.append(self._filename)
        file = tb.open_file(
            self._filename, 'a',
            driver=_driver,
            driver_core_backing_store=_driver_core_backing_store
        )

        # create expandable array
        earray = {}
        for _earray in self.config.flatten_attributes:
            earray[_earray.name] = file.create_earray(
                file.root, _earray.name,
                _earray.type,
                shape=tuple([0]) + _earray.shape,
                filters=self.filter
            )

        return (file, earray)

    def _get_size(self):
        if self.config.inmemory:
            return sys.getsizeof(self._file.get_file_image())
        else:
            return self._file.get_filesize()

    def _set_driver(self):
        """
        Setting HDF5 driver type

        Returns:
            :obj:`str` : HDF5 driver type string
        """
        if self.config.inmemory:
            return 'H5FD_CORE', False
        else:
            if os.name == "posix":
                return 'H5FD_SEC2', True
            elif os.name == "nt":
                return 'H5FD_WINDOWS', True
            else:
                raise ValueError("{} OS not supported!".format(os.name))

    @property
    def get_filelist(self):
        return self._filelist

    def disconnect(self):
        self._file_closing()
        self._uploader.join_queue()

        # metadata set
        key = uuid.uuid4().hex[:16]
        _metadata_file = tempfile.mktemp(f'{key}.json')
        self.config.metadata.to_json_file(_metadata_file)
        self._client.fput_object(
            self.config.bucket_name,
            f'metadata/{key}.json',
            _metadata_file
        )
        os.remove(_metadata_file)

    @property
    def get_disconnected(self):
        return self._disconnected