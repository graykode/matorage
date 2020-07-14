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

from matorage.serialize import Serialize
from matorage.utils import auto_attr_check

@auto_attr_check
class _DataMetadata(Serialize):
    r""" Metadata of dataset configuration classes.
        Handles a few parameters configuration for only dataset.
        This class is stored together in the MinIO in json format.

        Note:
            This class is recommended not to be used in the code of the user.
            **`DataConfig` must be mapped with only one `_DataMetadata`.**

    """
    datas = dict

    def __init__(self, **kwargs):
        self.datas = kwargs

    def __len__(self):
        return len(self.datas)