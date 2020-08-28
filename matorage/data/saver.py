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
import tempfile
import tables as tb
import numpy as np
from functools import reduce
from minio import Minio

from matorage.nas import NAS
from matorage.utils import is_tf_available, is_torch_available, check_nas
from matorage.uploader import Uploader

_KB = 1024
"""The size of a Kilobyte in bytes"""

_MB = 1024 * _KB
"""The size of a Megabyte in bytes"""


class DataSaver(object):
    """

    This class must be created independently for the process. The independent process uses
    multiple threads to upload to storage and generates unique metadata information when upload is complete.
    Update the file, push the upload queue if it exceeds a certain size, close the file, and create a new file.
    After saving, you should disconnect the data saver.

    To make This procedure easier to understand, the following is written in the pseudo-code.

    .. code-block::

        per_one_batch_data_size = array_size // num_batch
        per_one_file_batch_size = max_object_size // per_one_batch_data_size
        for batch_idx in range(num_batch):
            if get_current_stored_batch_size() < per_one_file_batch_size:
                file.append(data[batch_idx])
            else:
                file_closing()
                new_file is opened
                new_file.append(data[batch_idx])
        All files are closed.

    Note:
        - Deep Learning Framework Type : All(pure python is also possible)
        - **All processes should call the constructors of this class independently.**
        - After data save is over, you must disconnect through the disconnect function.

    Args:
        config (:obj:`matorage.DataConfig`, **require**):
            A DataConfig instance object
        multipart_upload_size (:obj:`integer`, optional, defaults to `5 * 1024 * 1024`):
            size of the incompletely uploaded object.
            You can sync files faster with `multipart upload in MinIO. <https://github.com/minio/minio-py/blob/master/minio/api.py#L1795>`_
            This is because MinIO clients use multi-threading, which improves IO speed more
            efficiently regardless of Python's Global Interpreter Lock(GIL).
        num_worker_threads (:obj:`integer`, optional, defaults to 4):
            number of backend storage worker to upload or download.

        inmemory (:obj:`boolean`, optional, defaults to `False`):
            If you use this value as `True`, then you can use `HDF5_CORE driver <https://support.hdfgroup.org/HDF5/doc/TechNotes/VFL.html#TOC1>`_
            so the temporary file for uploading or downloading to backend storage,
            such as MinIO, is not stored on disk but is in the memory.
            Keep in mind that using memory is fast because it doesn't use disk IO, but it's not always good.
            If default option(False), then `HDF5_SEC2` driver will be used on posix OS(or `HDF5_WINDOWS` in Windows).

        refresh (:obj:`boolean`, optional, defaults to `False`):
            All existing data is erased and overwritten.

    Single Process example

    Examples::

        import numpy as np
        from tqdm import tqdm
        from matorage import DataConfig, DataSaver

        data_config = DataConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            dataset_name='array_test',
            attributes=[
                ('array', 'uint8', (3, 224, 224)),
            ]
        )

        data_saver = DataSaver(config=data_config)
        row = 100
        data = np.random.rand(64, 3, 224, 224)

        for _ in tqdm(range(row)):
            data_saver({
                'array' : data
            })

        data_saver.disconnect()

    """

    def __init__(
        self,
        config,
        multipart_upload_size=5 * _MB,
        num_worker_threads=4,
        inmemory=False,
        refresh=False,
    ):

        self.config = config

        # Storage configuration
        self.multipart_upload_size = multipart_upload_size
        self.num_worker_threads = num_worker_threads

        # HDF5 configuration
        self.inmemory = inmemory

        self.filter = tb.Filters(**config.compressor)

        self._filelist = []
        self._file, self._earray = self._get_newfile()

        self._disconnected = False

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
        self._check_and_create_bucket(refresh=refresh)

        self._uploader = Uploader(
            client=self._client,
            bucket=self.config.bucket_name,
            num_worker_threads=self.num_worker_threads,
            multipart_upload_size=self.multipart_upload_size,
            inmemory=self.inmemory,
        )

        atexit.register(self._exit)

    def _append_file(self):
        """
        upload file to key called `<bucket_name>/key`.
        appended data is `Dict[str, str]`
        **`value` is file path of `str` type**
        example:
        {
            'key' : 'value.txt',
        }

        """
        for key, filepath in self._datas.items():
            self._uploader.set_queue(
                local_file=filepath,
                remote_file=key,
            )

            self.config.set_files(key)

    def _append_numpy(self):
        """
        append numpy array in `name` node.
        appended data is `Dict[str, numpy.ndarray]` type.
        **`value` is `numpy.ndarray` type with (B, *) shape, B means batch size**
        example:
            {
                'image' : np.random.rand(16, 28, 28),
                'target' : np.random.rand(16)
            }

        """
        array_size = self._get_array_size()
        bzs = list(self._datas.values())[0].shape[0]

        per_one_batch_data_size = array_size // bzs
        per_one_file_batch_size = max(
            1, self.config.max_object_size // per_one_batch_data_size
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

    def _check_and_create_bucket(self, refresh):
        if not self._client.bucket_exists(self.config.bucket_name):
            self._client.make_bucket(
                self.config.bucket_name, location=self.config.region
            )
        elif refresh:
            objects = self._client.list_objects(self.config.bucket_name, recursive=True)
            for obj in objects:
                self._client.remove_object(self.config.bucket_name, obj.object_name)

    def _check_attr_name(self, name):
        """
        check attribute names is exist

        """
        if name not in self._earray.keys():
            raise KeyError("attribute name {} is not exist!".format(name))

    def _check_data_filetype(self):
        """
        Check data which is file type

        """
        if not isinstance(self._datas, dict):
            raise TypeError("datas shoud be dict type.", self.__call__.__doc__)

        for key, filepath in self._datas.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError("{} is not found".format(filepath))

    def _check_data_numpytype(self):
        """
        Check data which is numpy array type

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

            self._datas[name] = array.reshape(
                -1, reduce(lambda x, y: x * y, array.shape[1:])
            )

    def __call__(self, datas, filetype=False):
        """

        Args:
            datas (:obj:`Dict[str, numpy.ndarray] or Dict[str, str]`, **require**):
                if filetype is false, `datas` is `Dict[str, numpy.ndarray]` type, **`value` is `numpy.ndarray` type with (B, *) shape, B means batch size**.
                else true, `datas` is `Dict[str, str]` type, **`value` is file path of `str` type**.
            filetype (:obj:`boolean`, optional):
                Indicates whether the type of data to be added to this bucket is a simple file type.

        Examples::

            data_saver = DataSaver(config=data_config)
            data_saver({
                'image' : np.random.rand(16, 28, 28),
                'target' : np.random.rand(16)
            })

        When used as shown below, filetype data is saved with a key called `<bucket_name>/raw_image`.

        Examples::

            data_saver = DataSaver(config=data_config)
            data_saver({
                'raw_image' : 'test.jpg'
            })
            print(data_config.get_filetype_list)

        """
        self._disconnected = False

        self._datas = datas

        if not filetype:
            self._check_data_numpytype()
            self._append_numpy()
        else:
            self._check_data_filetype()
            self._append_file()

    def _file_closing(self):
        _length = len(list(self._earray.values())[0])
        _last_index = self.config.get_length

        if not self.inmemory:
            self._file.close()
            self._uploader.set_queue(
                local_file=self._file.filename,
                remote_file=os.path.basename(self._filename),
            )
        else:
            self._uploader.set_queue(
                local_file=self._file.get_file_image(),
                remote_file=os.path.basename(self._filename),
            )
            self._file.close()
        # Set filename indexer
        _current_index = _last_index + _length
        self.config.set_indexer(
            {
                _current_index: {
                    "name": os.path.basename(self._filename),
                    "length": _length,
                }
            }
        )

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
            self._filename,
            "a",
            driver=_driver,
            driver_core_backing_store=_driver_core_backing_store,
        )

        # create expandable array
        earray = {}
        for _earray in self.config.flatten_attributes:
            earray[_earray.name] = file.create_earray(
                file.root,
                _earray.name,
                _earray.type,
                shape=tuple([0]) + _earray.shape,
                filters=self.filter,
            )

        return (file, earray)

    def _get_size(self):
        if self.inmemory:
            return sys.getsizeof(self._file.get_file_image())
        else:
            return self._file.get_filesize()

    def _set_driver(self):
        """
        Setting HDF5 driver type

        Returns:
            :obj:`str` : HDF5 driver type string
        """
        if self.inmemory:
            return "H5FD_CORE", False
        else:
            if os.name == "posix":
                return "H5FD_SEC2", True
            elif os.name == "nt":
                return "H5FD_WINDOWS", True
            else:
                raise ValueError("{} OS not supported!".format(os.name))

    @property
    def get_downloaded_dataset(self):
        """
        get local paths of downloaded dataset in local storage

        Returns:
            :obj:`list`: local path of downloaded datasets
        """
        return self._filelist

    def disconnect(self):
        """
        disconnecting datasaver. close all opened files and upload to backend storage.
        Must be called after ``datasaver`` function to store data safely.

        Examples::

            data_saver = DataSaver(config=data_config)
            data_saver({
                'image' : np.random.rand(16, 28, 28),
                'target' : np.random.rand(16)
            })
            data_saver.disconnect()

        """
        self._file_closing()
        self._uploader.join_queue()

        # metadata set
        key = uuid.uuid4().hex[:16]
        _metadata_file = tempfile.mktemp(f"{key}.json")
        self.config.metadata.to_json_file(_metadata_file)
        self._client.fput_object(
            self.config.bucket_name, f"metadata/{key}.json", _metadata_file
        )
        os.remove(_metadata_file)

    @property
    def get_disconnected(self):
        return self._disconnected
