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

import copy

from tables.atom import Atom
from matorage.serialize import Serialize

class DataAttribute(Serialize):

    def __init__(self, name, type, shape):

        self.name = name

        if isinstance(type, str):
            self.type = Atom.from_type(type)
        else:
            self.type = type

        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "type"):
            output["type"] = self.type.type
        return output