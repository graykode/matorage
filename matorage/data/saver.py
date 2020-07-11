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

from matorage.utils import auto_attr_check, _tf_available, _torch_available
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

    def __call__(self, array):
        """
        **`array` must be `numpy.ndarray` type with (B, *) shape **

        Returns:
            :None
        """
        assert isinstance()