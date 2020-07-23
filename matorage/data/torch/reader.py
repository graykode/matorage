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

from minio import Minio
from torch.utils.data import Dataset

class DataReader(Dataset):
    r"""DataReader class for Pytorch Dataset
        This class will be objective by each process. So we have to create `MinIO` object
        (Specifically, it is NOT safe to share it between multiple processes, for example when using multiprocessing.
        Pool. The solution is simply to create a new Minio object in each process, and not share it between processes.)
        [python-client-api-reference](https://docs.min.io/docs/python-client-api-reference.html)
        To avoid this problem, we have to create minio client object in `__getitem__` function

        If file is downloaded from remote storage every time by item index, reading speed will be very slow.
        So, we use file in-memory caching algorithm which is shareable with all workers to fast.

        The `DataReader` is carried out in the following procedure.
            1. Find number of thread to fetch file in a batch
            2. Each thread fetch object file from backend storage
            3. It does not go through step2 every time, and if there is a file in-memory cache, it does not fetch.
        below is pseudo-code.

        ```python
            num_thread = find(data index quires)
            data = []
            for
        ```
        Note:
            - The worst is that the index values from each batch must be fetched as much as the batch size for all other file objects.

        Args:
            config (:obj:`matorage.config.MTRConfig`, `require`):
                S3 object storage endpoint.
            num_caches (:obj:`int`, `optional`):
                number of file caches.
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.num_caches = kwargs.pop("num_caches", 16)

    def __len__(self):
        return self.config.get_indexer_last

    def __getitem__(self, item):
        _client = self._create_client()
        pass

    def _create_client(self):
        return Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        )