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

from matorage.utils import auto_attr_check

class _DataMapper(object):
    r""" Data indices mapper of dataset configuration classes.
        Handles a few parameters configuration for only dataset.

        Note:
            This class is recommended not to be used in the code of the user.
            **`DataConfig` must be mapped with only one `_DataMapper`.**

            INDEX_MAPPER
                When multiple atomical objects are also divided into one data,
                it is difficult to bring data into absolute index numbers.
                For example, suppose that 60000 datasets were divided into 600 objects, 100 each.
                Also, it is assumed that all data has been stored sequentially.
                If we want access to the 15011th data index, we need access to the 11th data from the 150th object file.
                As the page table in the OS, this class helps to map from the absolute index to the relative index.

    """

    def __init__(self, dataset_name, bucket_name):
        self._dataset_name = dataset_name
        self._bucket_name = bucket_name
        self._indices_mapper = {}

    def __call__(self, mapper):
        self._indices_mapper.update(mapper)

    def __len__(self):
        return len(self._indices_mapper)