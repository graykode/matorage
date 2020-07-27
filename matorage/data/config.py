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
import hashlib
from minio import Minio
from functools import reduce

from matorage.nas import NAS
from matorage.utils import check_nas
from matorage.config import MTRConfig
from matorage.data.metadata import DataMetadata
from matorage.data.attribute import DataAttribute

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
                    `attributes = DataAttribute('image', matorage.UInt8Atom, (28 * 28))`
                    or
                    ```python
                    attributes = [
                        DataAttribute('image', matorage.UInt8Atom, (28 * 28)),
                        DataAttribute('target', matorage.UInt8Atom, (1))
                    ]
                    ```
            compressor (:obj:`dict`, `optional`, defaults to `{"level" : 0, "lib" : "zlip"}`):
                Data compressor option, it same with [pytable's Filter](http://www.pytables.org/usersguide/libref/helper_classes.html#tables.Filters)

    """

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

        self.metadata = DataMetadata(**self.__dict__)

    def _check_all(self):
        """
        Check all class variable is fine.

        Returns:
            :obj: `None`:
        """
        self._check_bucket()

        if self.attributes is None:
            raise ValueError("attributes is empty")
        if isinstance(self.attributes, tuple):
            self.attributes = list(self.attributes)
        if isinstance(self.attributes, DataAttribute):
            self.attributes = [self.attributes]

        attribute_names = set()
        for attribute in self.attributes:
            assert isinstance(attribute.type, tables.atom.Atom)
            if attribute.name in attribute_names:
                raise KeyError("{} is already exist in {}".format(attribute.name, attribute_names))
            else:
                attribute_names.add(attribute.name)

        # To convert `self.attributes`'s shape to be flatten
        self.flatten_attributes = copy.deepcopy(self.attributes)
        self._convert_type_flatten()

        if self.compressor['complevel'] < 0  or 9 < self.compressor['complevel']:
            raise ValueError("Compressor level is {} must be 0-9 interger".format(self.compressor['level']))
        if self.compressor['complib'] not in ('zlib', 'lzo', 'bzip2', 'blosc'):
            raise ValueError("compressor mode {} is not valid. select in "
                             "zlib, lzo, bzip2, blosc".format(self.compressor['lib']))

    def _check_bucket(self):
        """
        Check bucket name is exist. If not exist, create new bucket
        If bucket and metadata sub folder exist, get metadata(attributes, compressor) from there.

        Returns:
            :obj: `None`:
        """
        _client = Minio(self.endpoint,
                            access_key=self.access_key,
                            secret_key=self.secret_key,
                            secure=self.secure) if not check_nas(self.endpoint) else NAS(self.endpoint)
        if not _client.bucket_exists(self.bucket_name):
            _client.make_bucket(self.bucket_name)
        else:
            objects = _client.list_objects(
                self.bucket_name,
                prefix='metadata/'
            )
            _metadata = None
            for obj in objects:
                _metadata = _client.get_object(self.bucket_name, obj.object_name)
                break
            if not _metadata:
                return

            metadata_dict = json.loads(_metadata.read().decode('utf-8'))
            if self.endpoint != metadata_dict['endpoint']:
                raise ValueError("Already created endpoint({}) doesn't current endpoint str({})"
                                 " It may occurs permission denied error".format(metadata_dict['endpoint'], self.endpoint))

            self.compressor = metadata_dict['compressor']
            self.attributes = [
                DataAttribute(**item) for item in metadata_dict['attributes']
            ]

    def _convert_type_flatten(self):
        for attribute in self.flatten_attributes:
            attribute.shape = (reduce(lambda x, y: x * y, attribute.shape),)

    def _hashmap_transfer(self):
        """
        Get unikey bucket name with `dataset_name` and `additional`

        Returns:
            :obj: `str`:
        """
        if not isinstance(self.dataset_name, str):
            raise ValueError("dataset_name {} is empty or not str type".format(self.dataset_name))
        if not isinstance(self.additional, dict):
            raise TypeError("additional is not dict type")

        key = self.dataset_name + json.dumps(self.additional)
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(
            self.__class__.__base__(**self.__dict__).__dict__
        )
        output["dataset_name"] = self.metadata.dataset_name
        output["additional"] = self.metadata.additional
        return output

    @classmethod
    def from_json_file(cls, json_file):
        """
        Constructs a `Config` from the path to a json file of parameters.

        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.

        """
        config_dict = cls._dict_from_json_file(json_file)
        metadata_dict = cls._load_metadata_from_bucket(config_dict)

        if 'dataset_name' in config_dict:
            del config_dict['dataset_name']
        if 'additional' in config_dict:
            del config_dict['additional']
        if 'attributes' in metadata_dict:
            metadata_dict['attributes'] = [
                DataAttribute(**item) for item in metadata_dict['attributes']
            ]

        return cls(
            **config_dict, **metadata_dict,
            metadata=DataMetadata(**metadata_dict)
        )

    def set_indexer(self, index):
        self.metadata.indexer.update(index)

    @property
    def get_indexer_last(self):
        keys = list(self.metadata.indexer.keys())
        if len(keys) == 0:
            return 0
        else:
            return keys[-1]

    @classmethod
    def _load_metadata_from_bucket(cls, config_dict):
        """
        Load `metadata.json` from bucket name

        Returns:
            :obj: `dict`: metadata json
        """
        _bucket_name = hashlib.md5(
            (config_dict['dataset_name'] + json.dumps(config_dict['additional'])).encode('utf-8')
        ).hexdigest()

        _client = Minio(config_dict['endpoint'],
                            access_key=config_dict['access_key'],
                            secret_key=config_dict['secret_key'],
                            secure=config_dict['secure']) \
            if not check_nas(config_dict['endpoint']) else NAS(config_dict['endpoint'])
        if not _client.bucket_exists(_bucket_name):
            raise AssertionError("{} with {} is not exist on {} or key is mismathced".format(
                config_dict['dataset_name'], config_dict['additional'], config_dict['endpoint']
            ))
        objects = _client.list_objects(
            _bucket_name,
            prefix='metadata/'
        )
        _metadata = None
        for obj in objects:
            _metadata = _client.get_object(_bucket_name, obj.object_name)
            break
        if not _metadata:
            raise AssertionError("metadata folder is not exist in minio storage")

        return json.loads(_metadata.read().decode('utf-8'))