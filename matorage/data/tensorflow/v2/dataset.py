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
import tensorflow as tf
import tensorflow_io as tfio

from matorage.data.data import MTRData
from matorage.utils import logger


class Dataset(MTRData):
    """
    Dataset class for Tensorflow Dataset

    This class is customized for the dataset of the PyTorch, so it is operated by the following procedure.

    1. The ``_object_file_mapper`` manages the minio object as key and the downloaded local path as value. \
    When minio object is downloaded, it is recorded in ``_object_file_maper``.
    2. We read ``_object_file_mapper`` and download only new objects that are not there.
    3. if Tensorflow v2(2.2.0>=), we use ``tfio.IODataset.from_hdf5`` and parallel ``interleave`` more fast

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

        batch_size (:obj:`integer`, `optional`, defaults to `1`):
            how many samples per batch to load.
        shuffle (:obj:`boolean`, `optional`, defaults to `False`):
            set to True to have the data reshuffled at every epoch.
        seed (:obj:`integer`, `optional`, defaults to `0`):
            random seed used to shuffle the sampler if ``shuffle=True``.

    Examples::

        from matorage import DataConfig
        from matorage.tensorflow import Dataset

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
        for array in dataset.dataloader:
            print(array)

        # index mode
        print(dataset[0])

    """

    def __init__(self, config, batch_size=1, **kwargs):

        # class parameters
        self._batch_size = batch_size
        self._shuffle = kwargs.pop("shuffle", False)
        self._seed = kwargs.pop("seed", 0)
        self.index = kwargs.pop("index", False)

        super(Dataset, self).__init__(config, **kwargs)

        if not self.index:
            _dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
            if self._shuffle:
                _dataset = _dataset.shuffle(len(self.filenames), seed=self._seed)
            self._dataloader = _dataset.interleave(
                self._get_item_with_download, cycle_length=len(self.filenames)
            )

    def __getitem__(self, idx):
        return self._get_item_with_inmemory(idx)

    def _get_item_with_download(self, filename):
        _tfios = []
        for _attr_name, _attr_value in self.attribute.items():
            _tfios.append(
                tfio.IODataset.from_hdf5(
                    filename,
                    dataset=f"/{_attr_name}",
                    spec=tf.as_dtype(_attr_value["type"]),
                ).map(
                    lambda x: tf.reshape(x, _attr_value["shape"])
                    if not _attr_value["shape"] == [1]
                    else x[0]
                )
            )
        _tfiodataset = tf.data.Dataset.zip(tuple(_tfios))
        if self._shuffle:
            _tfiodataset = _tfiodataset.shuffle(1000, seed=self._seed)
        _tfiodataset = _tfiodataset.batch(
            self._batch_size, drop_remainder=True
        ).prefetch(tf.data.experimental.AUTOTUNE)
        return _tfiodataset

    def _reshape_convert_tensor(self, numpy_array, attr_name):
        """
        Reshape numpy tensor and convert from numpy to torch tensor.
        In matorage dataset save in 2D (bz, N) shape to cpu L1 cache manage dataset fast.
        Therefore, this function restores the shape for the user to use.

        Returns:
            :obj:`tf.tensor`
        """
        _shape = self.attribute[attr_name]["shape"]
        numpy_array = numpy_array.reshape(_shape)
        tensor = tf.convert_to_tensor(numpy_array)
        return tensor

    @property
    def filenames(self):
        """
        Get filenames(file absolute path) in local storage

        Returns:
            :obj:`list`: filenames(file absolute path) in local storage
        """
        return list(self._object_file_mapper.values())

    @property
    def dataloader(self):
        """
        Get iterative dataloader

        Returns:
            :obj:`InterleaveDataset`:iterative tf.data.dataset
        """
        return self._dataloader
