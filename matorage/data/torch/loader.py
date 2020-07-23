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

import json
import random
from minio import Minio
from torch.utils.data import DataLoader

class MTRDataLoader(DataLoader):
    r"""MTRDataLoader class for Pytorch Dataset

        `MTRDataLoader` class is more complicated than `DataSaver`
        Let's suppose we have 10000(images) * 3 * 224 * 224 float64 images.
        And each data are splitted into 100 images, so 100 files(10000/100, 15.05MB) exist.
        When creating random-generated query indexes through data sampler,
        the worst-case scenario can occur for all indexes without overlapping files.
        Then we have to fetch 1.5GB(15.05MB * 100) in every batch.
        To avoid this problem, we introduce the following pseudo-code algorithms:

        ```python
            object_list = get all obejct list in MinIO with metadata

            Local Shuffle Algorithm

            object_list = shuffle(object_list)
            re-mapping index_mapper according to new order of object list
            set data index queries with Local Shuffle Algorithm
            for idx, object in enumerate(range(len(object_list), window_size)):
                if idx == 0:
                    files = fetching(object)
                if pre-fetching option is `True` and idx != len(object_list) - 1:
                    next_files = pre-fetching(next_object)
               data = read(files)

                yield data

                if idx != len(object_list) - 1:
                    data = new_data
        ```

        Note:
            Deep Learning Framework Type : Pytorch

        Args:
            config (:obj:`matorage.config.MTRConfig`, `require`):
                S3 object storage endpoint.
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.shuffle = kwargs.get("shuffle", False)
        self.seed = kwargs.pop("seed", 42)

        reindexer = self._reindexing(
            bucket_name=self.config.bucket_name
        )

        super(MTRDataLoader, self).__init__(**kwargs)

    def _create_client(self):
        return Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        )

    def _reindexing(self, bucket_name):
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

        if self.shuffle:
            random.seed(seed=self.seed)
            random.shuffle(total_index)

        reindexer = {}
        for _index in total_index:
            key = list(reindexer.keys())[-1] + _index["length"] if reindexer else _index["length"]
            reindexer[key] = _index["name"]

        return reindexer