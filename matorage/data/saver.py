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
import uuid
import tables as tb
import numpy as np
from functools import reduce

from matorage.utils import auto_attr_check, is_tf_available, is_torch_available
from matorage.data.config import DataConfig

@auto_attr_check
class DataSaver(object):
    r""" Dataset saver classes.
        Although it plays a similar role as `[torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html)`
        , it is a class that is contradictory. If you attempt to write 100 data sequentially from 60000 data,
        it will work like a `consumer-producer pattern` with the logic in the following order.
            1. We know that 60000/100=600 data will be produced because 60000 data is divided into 100.
            2. The data that builds up in production is appended to a file of unique names.
            3. If the file size exceeds 'OBJECT_SIZE', it is uploaded to MinIO with deleted in local and the new file is written.
        Step3 occurs asynchronously. Therefore, while step3 is in progress, step1 and 2 are in progress.

        To make the above procedure easier to understand, the following is written in the pseudo-code.
            ```python
            file is opened, if file already exist, there will be append mode.
            for data(shape : 100 x 784) in multiprocessing(dataset(shape : 60000 x 784))
                if OBJECT_SIZE <= file size
                    file is closed
                    lock other processes until new_file is opened
                    new_file is opened
                    new_file.append(data)
                    asynchronously upload to backend storage
                else
                    file.append(data)
            file is closed
            ```
        In order to prevent multiple processes entering into the 'OBJECT_SIZE E= file size' during multiprocessing,
        other processes must be locked for a while until a new file is created.

        Note:
            - Both Tensorflow and Pytorch are eligible for this class.
            - According to the [MinIO python client document](https://docs.min.io/docs/python-client-api-reference.html),
            MinIO is a thread safety, but during multiprocessing, you must create a new MinIO Class object for each process.

        Args:
            config (:obj:`matorage.config.MTRConfig`, `require`):
                S3 object storage endpoint.

        Example::

    """
    config = DataConfig

    def __init__(self, config):
        self.config = config
        self._driver = self._set_driver(config)

        self._current = tb.open_file(self._get_name(), 'a')

        self.filter = tb.Filters(**config.compressor)

    def __call__(self, array):
        """
        **`array` must be `numpy.ndarray` type with (B, *) shape **

        Returns:
            :None
        """
        if is_tf_available() and not isinstance(array, np.ndarray):
            import tensorflow as tf
            assert isinstance(array, tf.python.framework.ops.EagerTensor), \
                "array type is not `numpy.ndarray` nor `EagerTensor`"
        if is_torch_available() and not isinstance(array, np.ndarray):
            import torch
            assert isinstance(array, torch.Tensor), \
                "array type is not `numpy.ndarray` nor `torch.Tensor`"

        assert isinstance(array, np.ndarray), "array type is not `numpy.ndarray`"

        # This resape is made into a (B, *) shape.
        # Shape is lowered to two contiguous dimensions, enabling IO operations to operate very quickly.
        # https://www.slideshare.net/HDFEOS/caching-and-buffering-in-hdf5#25
        array = array.reshape(-1, reduce(lambda x, y: x * y, array.shape[1:]))

    def _get_name(self, length=16):
        return "{}.h5".format(uuid.uuid4().hex[:length])

    def _set_driver(self, config):
        if config.inmemory:
            return 'H5FD_CORE'
        else:
            if os.name == "posix":
                return 'H5FD_SEC2'
            elif os.name == "nt":
                return 'H5FD_WINDOWS'
            else:
                raise ValueError("{} OS not supported!".format(os.name))