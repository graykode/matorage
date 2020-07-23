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

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import (
    _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
)

class MTRDataLoader(DataLoader):
    def __init__(self, *args, prefetch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_size = prefetch_size if prefetch_size is not None else 2 * kwargs.get('num_workers', 0)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)
