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

import tensorflow as tf
import tensorflow_io as tfio

from matorage.data.data import MTRData

class MTRDataset(MTRData):
    r"""MTRDataset class for Tensorflow Dataset

        This class is customized for the dataset of the Tensorflow, so it is operated by the following procedure.
        1. The `_object_file_mapper` manages the minio object as key and the downloaded local path as value.
            {'tmpv7sy5_1fff7845eccd874068.h5': '/tmp/tmpja6wo221tmpv7sy5_1fff7845eccd874068.h5'}
            When minio object is downloaded, it is recorded in _object_file_maper.
        2. We read `_object_file_mapper` and download only new objects that are not there.
        3. if Tensorflow v2, we use `tfio.IODataset.from_hdf5` and parallel `interleave` more fast

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

    def __init__(self, config, batch_size, num_worker_threads=4, clear=True, inmemory=False, cache_folder_path='~/.matorage'):
        super(MTRDataset, self).__init__(config, num_worker_threads, clear, inmemory, cache_folder_path)
        _dataset = tf.data.Dataset.from_tensor_slices(self.objectnames)
        self._batch_size = batch_size
        self._dataloader = _dataset.interleave(self._create_tfiodata, cycle_length=len(self.objectnames))

    def _create_tfiodata(self, filename):
        _tfios = []
        for _attr_name, _attr_value in self.attribute.items():
            _tfios.append(
                tfio.IODataset.from_hdf5(
                    filename,
                    dataset=f"/{_attr_name}",
                    spec=tf.as_dtype(_attr_value["type"])
                ).map(
                    lambda x: tf.reshape(x, _attr_value["shape"])
                )
            )
        return tf.data.Dataset.zip(
            tuple(_tfios)
        ).batch(self._batch_size, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE
        )

    @property
    def objectnames(self):
        """
        Get object name in minio storage

        Returns:
            :obj:`list`: object name of list
        """
        return list(self._object_file_mapper.values())

    @property
    def dataloader(self):
        """
        Get dataloader

        Returns:
            :obj:`InterleaveDataset`:
        """
        return self._dataloader