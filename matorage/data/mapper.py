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

class _DatasetMapper(object):
    r""" Dataset mapper classes.
        When multiple atomical objects are also divided into one dataset,
        it is difficult to bring dataset into absolute index numbers.
        For example, suppose that 60000 datasets were divided into 600 objects, 100 each.
        Also, it is assumed that all dataset has been stored sequentially.
        If we want access to the 15011th dataset index, we need access to the 11th dataset from the 150th object file.
        As the page table in the OS, this class helps to map the index relative to the absolute index.

        Note:
            This class is recommended not to be used in the code of the user.
            **`MRTConfig` must be mapped with only one `MRTMapper`.**

        Args:


        Example::
    """

    def __init__(self):
        pass