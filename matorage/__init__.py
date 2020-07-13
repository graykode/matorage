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

from matorage.data.config import DataConfig
from matorage.data.attribute import DataAttribute
from matorage.data.saver import DataSaver

import tables

"""
    We are following the same data type method as Pytables.
    http://www.pytables.org/usersguide/libref/declarative_classes.html?highlight=int8atom#atom-sub-classes
"""
StringAtom = tables.StringAtom
BoolAtom = tables.BoolAtom

IntAtom = tables.IntAtom
Int8Atom = tables.Int8Atom
Int16Atom = tables.Int16Atom
Int32Atom = tables.Int32Atom
Int64Atom = tables.Int64Atom

UIntAtom = tables.UIntAtom
UInt8Atom = tables.UInt8Atom
UInt16Atom = tables.UInt16Atom
UInt32Atom = tables.UInt32Atom
UInt64Atom = tables.UInt64Atom

FloatAtom = tables.FloatAtom
Float32Atom = tables.Float32Atom
Float64Atom = tables.Float64Atom

ComplexAtom = tables.ComplexAtom
Time32Atom = tables.Time32Atom
Time64Atom = tables.Time64Atom
EnumAtom = tables.EnumAtom

__all__ = [
    'DataConfig',
    'DataAttribute',
    'DataSaver',

    'StringAtom',
    'BoolAtom',

    'IntAtom',
    'Int8Atom',
    'Int16Atom',
    'Int32Atom',
    'Int64Atom',

    'UIntAtom',
    'UInt8Atom',
    'UInt16Atom',
    'UInt32Atom',
    'UInt64Atom',

    'FloatAtom',
    'Float32Atom',
    'Float64Atom',

    'ComplexAtom',
    'Time32Atom',
    'Time64Atom',
    'EnumAtom'
]