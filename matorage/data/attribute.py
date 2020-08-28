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
from tables.atom import Atom, StringAtom

from matorage.serialize import Serialize


class DataAttribute(Serialize):
    """
    DataAttribute classes.

    Args:
        name (:obj:`string`, **require**):
            data attribute name .
        type (:obj:`string`, **require**):
            data attribute type. select in `string`, `bool`, `int8`, `int16`, `int32`, `int64`,
            `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`
        shape (:obj:`tuple`, **require**):
            data attribute shape. For example, if you specify a shape with (2, 2), you can store an array of (B, 2, 2) shapes.
        itemsize (:obj:`integer`, optional, defaults to 0):
            itemsize(bytes) for string type attribute. Must be set for string type attribute.

    Examples::

        >>> from matorage import DataAttribute
        >>> attribute = DataAttribute('array', 'uint8', (2, 2))
        >>> attribute.name
        'array'
        >>> attribute.shape
        (2, 2)
        >>> attribute.type
        UInt8Atom(shape=(), dflt=0)

    """

    type_list = [
        "string",
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ]

    def __init__(self, name, type, shape, itemsize=0):

        self.name = name
        self.itemsize = itemsize

        if not isinstance(type, str):
            raise TypeError("`type` is not str type")

        if type not in self.type_list:
            raise ValueError("`type` {} is unavailable type".format(type))
        elif type == "string":
            if itemsize == 0:
                raise ValueError("set `itemsize` for string type")
            self.type = StringAtom(itemsize=itemsize)
        else:
            self.type = Atom.from_type(type)

        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[string, any]`: Dictionary of all the attributes that make up this configuration instance

        Examples::

            >>> from matorage import DataAttribute
            >>> attribute = DataAttribute('array', 'uint8', (2, 2))
            >>> attribute.to_dict()
            {'name': 'array', 'type': 'uint8', 'shape': (2, 2)}

        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "type") or "type" in output:
            output["type"] = self.type.type
        if hasattr(self.__class__, "itemsize") or "itemsize" in output:
            output["itemsize"] = self.itemsize
        return output
