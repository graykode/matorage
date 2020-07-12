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
import tables

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
            attributes (:obj:`list`, `require`):
                DataAttribute type of list for data attributes
                example
                    `attributes = [DataAttribute('image', np.int8, (28 * 28))]`
                    or
                    ```python
                    attributes = [
                        DataAttribute('image', np.int8, (28 * 28)),
                        DataAttribute('target', np.int8, (1))
                    ]
                    ```
            compressor (:obj:`dict`, `optional`, defaults to `{"level" : 0, "lib" : "zlip"}`):
                Data compressor option, it same with [pytable's Filter](http://www.pytables.org/usersguide/libref/helper_classes.html#tables.Filters)

    """
    dataset_name = str
    additional = dict
    attributes = list
    filter = dict
    bucket_name = str

    _metadata = _DataMetadata
    _mapper = _DataMapper

    def __init__(self, **kwargs):
        super(DataConfig, self).__init__(**kwargs)

        self.dataset_name = kwargs.pop("dataset_name", None)
        self.additional = kwargs.pop("additional", {})
        self.attributes = kwargs.pop("attribute", None)
        self.compressor = kwargs.pop("compressor", {
            "level" : 0,
            "lib" : "zlip"
        })
        self.bucket_name = self._hashmap_transfer()

        self._check_all()

        self._metadata = _DataMetadata(**self.__dict__)
        self._mapper = _DataMetadata(
            dataset_name=self.dataset_name,
            bucket_name=self.bucket_name
        )

    def _check_all(self):
        """
        Check all class variable is fine.

        Returns:
            :obj: `None`:
        """
        self._check_bucket(bucket_name=self.bucket_name)

        if isinstance(self.attributes, DataAttribute):
            self.attributes = [self.attributes]

        for attribute in self.attributes:
            if len(attribute.shape) < 2:
                raise ValueError("Shape is 1 dimension. shape should be (Batch, *)")
            assert isinstance(attribute.type, tables.atom.Atom)

        if self.compressor['level'] < 0  or 9 < self.compressor['level']:
            raise ValueError("Compressor level is {} must be 0-9 interger".format(self.compressor['level']))
        if self.compressor['lib'] not in ('zlib', 'lzo', 'bzip2', 'blosc'):
            raise ValueError("compressor mode {} is not valid. select in "
                             "zlib, lzo, bzip2, blosc".format(self.compressor['lib']))

    def _check_bucket(self, bucket_name):
        """
        Check bucket name is exist.

        Returns:
            :obj: `None`:
        """
        pass

    def _hashmap_transfer(self):
        """
        Get unikey bucket name with `dataset_name` and `additional`

        Returns:
            :obj: `str`:
        """
        pass

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