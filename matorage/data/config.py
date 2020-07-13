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
import json
import tables
from minio import Minio

from matorage.utils import auto_attr_check
from matorage.config import MTRConfig

from matorage.data.mapper import _DataMapper
from matorage.data.metadata import _DataMetadata
from matorage.data.attribute import DataAttribute

@auto_attr_check
class DataConfig(MTRConfig):
    r""" Dataset configuration classes.
        Handles a few parameters configuration for only dataset.

        Note:
            The variables in this class are static variables that are calculated
            only once when DataClass is declared.

        Args:
            dataset_name (:obj:`string`, `require`):
                dataset name.
            additional (:obj:`dict`, `optional`, defaults to `{}`):
                Parameters for additional description of datasets.
                The key and value of the dictionay can be specified very freely.
                example
                    ```python
                        {
                            'data_creator' : 'taehwanjung',
                            'data_version' : 0.1
                            ...
                        }
                    ```
            attributes (:obj:`tuple`, `require`):
                DataAttribute type of list for data attributes
                example
                    `attributes = DataAttribute('image', matorage.UInt8Atom, (28 * 28))`
                    or
                    ```python
                    attributes = (
                        DataAttribute('image', matorage.UInt8Atom, (28 * 28)),
                        DataAttribute('target', matorage.UInt8Atom, (1))
                    )
                    ```
            compressor (:obj:`dict`, `optional`, defaults to `{"level" : 0, "lib" : "zlip"}`):
                Data compressor option, it same with [pytable's Filter](http://www.pytables.org/usersguide/libref/helper_classes.html#tables.Filters)

    """
    # Requirement Arguments
    endpoint = str
    access_key = str
    secret_key = str

    dataset_name = str
    additional = dict
    attributes = tuple
    filter = dict
    bucket_name = str

    _metadata = _DataMetadata
    _mapper = _DataMapper

    def __init__(self, **kwargs):
        super(DataConfig, self).__init__(**kwargs)

        self.dataset_name = kwargs.pop("dataset_name", None)
        self.additional = kwargs.pop("additional", {})
        self.attributes = kwargs.pop("attributes", None)
        self.compressor = kwargs.pop("compressor", {
            "complevel" : 0,
            "complib" : "zlib"
        })
        self.bucket_name = self._hashmap_transfer()

        self._check_all()

        self._metadata = _DataMetadata(**self.__dict__)
        self._mapper = _DataMapper(
            dataset_name=self.dataset_name,
            bucket_name=self.bucket_name
        )

    def _check_all(self):
        """
        Check all class variable is fine.

        Returns:
            :obj: `None`:
        """

        if isinstance(self.attributes, list):
            self.attributes = tuple(self.attributes)
        if isinstance(self.attributes, DataAttribute):
            self.attributes = (self.attributes,)

        attribute_names = set()
        for attribute in self.attributes:
            assert isinstance(attribute.type(), tables.atom.Atom)
            if attribute.name in attribute_names:
                raise KeyError("{} is already exist in {}".format(attribute.name, attribute_names))
            else:
                attribute_names.add(attribute.name)

        if self.compressor['complevel'] < 0  or 9 < self.compressor['complevel']:
            raise ValueError("Compressor level is {} must be 0-9 interger".format(self.compressor['level']))
        if self.compressor['complib'] not in ('zlib', 'lzo', 'bzip2', 'blosc'):
            raise ValueError("compressor mode {} is not valid. select in "
                             "zlib, lzo, bzip2, blosc".format(self.compressor['lib']))

        self._check_bucket()

    def _check_bucket(self):
        """
        Check bucket name is exist. If not exist, create new bucket

        Returns:
            :obj: `None`:
        """
        minioClient = Minio(self.endpoint,
                            access_key=self.access_key,
                            secret_key=self.secret_key,
                            secure=self.secure)
        if not minioClient.bucket_exists(self.bucket_name):
            minioClient.make_bucket(self.bucket_name)
        else:
            raise ValueError("{} {} is already exist!".format(self.dataset_name, self.additional))

    def _hashmap_transfer(self):
        """
        Get unikey bucket name with `dataset_name` and `additional`

        Returns:
            :obj: `str`:
        """
        key = self.dataset_name + json.dumps(self.additional)
        return str(hash(key) % 10**16)

    def _to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        if hasattr(self.__class__, "dataset_name"):
            output["dataset_name"] = self.__class__.dataset_name

        if "metadata" in output:
            output["metadata"] = self._metadata.to_dict()

        return output

    @property
    def get_mapper(self):
        return self._mapper

    @get_mapper.setter
    def set_mapper(self, mapper):
        self._mapper(mapper)