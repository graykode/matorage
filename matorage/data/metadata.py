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

from matorage.serialize import Serialize


class DataMetadata(Serialize):
    r""" Metadata of dataset configuration classes.
        Handles a few parameters configuration for only dataset.
        This class is stored together in the MinIO in json format.

        Note:
            This class is recommended not to be used in the code of the user.
            **`DataConfig` must be mapped with only one `DataMetadata`.**

        About indexer's work
            we manages index to dict type as total data length as key and file names as value.
            Examples:
                "indexer": {
                    "15960": "tmpjopm9_qr118171adec9d410d.h5",
                    "31920": "tmpnzdga4zfc952613ed2bf4fad.h5",
                    "47880": "tmpy3fwohhx178eb7c1d17d4a78.h5",
                    "60000": "tmpidco3y8u794f8adde1e5479a.h5"
                }
    """

    def __init__(self, **kwargs):
        self.endpoint = kwargs["endpoint"]
        self.dataset_name = kwargs.pop("dataset_name", None)
        self.additional = kwargs.pop("additional", {})

        self.attributes = kwargs.pop("attributes", None)
        self.compressor = kwargs.pop("compressor", {"complevel": 0, "complib": "zlib"})

        self.indexer = {}
        self.filetype = []

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "attributes") or "attributes" in output:
            output["attributes"] = [
                _attribute.to_dict() for _attribute in self.attributes
            ]
        if (
            hasattr(self.__class__, "flatten_attributes")
            or "flatten_attributes" in output
        ):
            output["flatten_attributes"] = [
                _attribute.to_dict() for _attribute in self.flatten_attributes
            ]
        return output
