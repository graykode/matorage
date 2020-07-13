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

import tables

from matorage.utils import auto_attr_check

@auto_attr_check
class DataAttribute(object):

    name = str
    shape = tuple

    def __init__(self, name, type, shape):
        self.name = name
        self.type = type
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape