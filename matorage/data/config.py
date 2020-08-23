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

_KB = 1024
"""The size of a Kilobyte in bytes"""

_MB = 1024 * _KB
"""The size of a Megabyte in bytes"""

import copy
import json
import tables
import hashlib
import tempfile
from minio import Minio
from functools import reduce

from matorage.nas import NAS
from matorage.utils import check_nas, logger
from matorage.config import StorageConfig
from matorage.data.metadata import DataMetadata
from matorage.data.attribute import DataAttribute


class DataConfig(StorageConfig):
    """
    Dataset configuration classes. This class overrides ``StorageConfig``.

    Args:
        endpoint (:obj:`string`, **require**):
            S3 object storage endpoint. or If use NAS setting, NAS folder path.
        access_key (:obj:`string`, optional, defaults to `None`):
            Access key for the object storage endpoint. (Optional if you need anonymous access).
        secret_key (:obj:`string`, optional, defaults to `None`):
            Secret key for the object storage endpoint. (Optional if you need anonymous access).
        secure (:obj:`boolean`, optional, defaults to `False`):
            Set this value to True to enable secure (HTTPS) access. (Optional defaults to False unlike the original MinIO).
        max_object_size (:obj:`integer`, optional, defaults to `10MB`):
            One object file is divided into `max_object_size` and stored.

        dataset_name (:obj:`string`, **require**):
            dataset name.
        attributes (:obj:`list`, **require**):
            DataAttribute type of list for data attributes
        additional (:obj:`dict`, optional, defaults to ``{}``):
            Parameters for additional description of datasets. The key and value of the dictionay can be specified very freely.
        compressor (:obj:`dict`, optional, defaults to :code:`{"complevel" : 0, "complib" : "zlib"}`):
            Data compressor option. It consists of a dict type that has complevel and complib as keys.
            For further reference, read `pytable's Filter <http://www.pytables.org/usersguide/libref/helper_classes.html#tables.Filters>`_.

            - complevel (:obj:`integer`, defaults to 0) : compressor level(0~9). The larger the number, the more compressed it is.
            - complib (:obj:`string`, defaults to 'zlib') : compressor library. choose in zlib, lzo, bzip2, blosc
        max_object_size (:obj:`integer`, optional, defaults to `10MB`):
            One object file is divided into `max_object_size` and stored.

    Examples::

        from matorage import DataConfig, DataAttribute
        data_config = DataConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            dataset_name='mnist',
            additional={
                "framework" : "pytorch",
                "mode" : "training"
            },
            compressor={
                "complevel" : 0,
                "complib" : "zlib"
            },
            attributes=[
                ('image', 'float32', (28, 28)),
                ('target', 'int64', (1, ))
            ]
        )

        data_config.to_json_file('data_config.json')
        data_config2 = DataConfig.from_json_file('data_config.json')

    If you have NAS(network access storage) settings, You can save/load faster by using the endpoint as a NAS folder path.

    Examples::

        from matorage import DataConfig

        # NAS example
        data_config = DataConfig(
            endpoint='~/shared',
            dataset_name='mnist',
            additional={
                "framework" : "pytorch",
                "mode" : "training"
            },
            compressor={
                "complevel" : 0,
                "complib" : "zlib"
            },
            attributes=[
                ('image', 'float32', (28, 28)),
                ('target', 'int64', (1, ))
            ]
        )


    """

    def __init__(self, **kwargs):
        super(DataConfig, self).__init__(**kwargs)
        self.type = "dataset"

        self.dataset_name = kwargs.pop("dataset_name", None)
        self.additional = kwargs.pop("additional", {})
        self.attributes = kwargs.pop("attributes", None)
        self.compressor = kwargs.pop("compressor", {"complevel": 0, "complib": "zlib"})
        self.max_object_size = kwargs.pop("max_object_size", 10 * _MB)

        self.bucket_name = self._hashmap_transfer()

        self._check_all()

        self.metadata = DataMetadata(**self.__dict__)

    def _check_all(self):
        """
        Check all class variable is fine.

        """
        self._check_bucket()

        if self.attributes is None:
            raise ValueError("attributes is empty")
        if isinstance(self.attributes, tuple):
            self.attributes = DataAttribute(
                name=self.attributes[0],
                type=self.attributes[1],
                shape=self.attributes[2],
            )
        if isinstance(self.attributes, DataAttribute):
            self.attributes = [self.attributes]

        for i, attr in enumerate(self.attributes):
            if isinstance(attr, tuple):
                self.attributes[i] = DataAttribute(attr[0], attr[1], attr[2])

        attribute_names = set()
        for attribute in self.attributes:
            assert isinstance(attribute.type, tables.atom.Atom)
            if attribute.name in attribute_names:
                raise KeyError(
                    "{} is already exist in {}".format(attribute.name, attribute_names)
                )
            else:
                attribute_names.add(attribute.name)

        # To convert `self.attributes`'s shape to be flatten
        self.flatten_attributes = copy.deepcopy(self.attributes)
        self._convert_type_flatten()

        if self.compressor["complevel"] < 0 or 9 < self.compressor["complevel"]:
            raise ValueError(
                "Compressor level is {} must be 0-9 interger".format(
                    self.compressor["level"]
                )
            )
        if self.compressor["complib"] not in ("zlib", "lzo", "bzip2", "blosc"):
            raise ValueError(
                "compressor mode {} is not valid. select in "
                "zlib, lzo, bzip2, blosc".format(self.compressor["lib"])
            )

    def _check_bucket(self):
        """
        Check bucket name is exist. If not exist, create new bucket
        If bucket and metadata sub folder exist, get metadata(attributes, compressor) from there.

        """

        aws_region_list = [
            'us-east-1',
            'us-east-2',
            'us-west-1',
            'us-west-2',
            'eu-west-1',
            'eu-west-2',
            'ca-central-1',
            'eu-central-1',
            'sa-east-1',
            'cn-north-1',
            'ap-southeast-1',
            'ap-southeast-2',
            'ap-northeast-1',
            'ap-northeast-2',
        ]

        if "amazonaws.com" in self.endpoint:
            if (self.region not in aws_region_list):
                raise AssertionError(
                    'AWS endpoint should has region argument from {}'.format(
                        aws_region_list
                    )
                )
            if f"s3.{self.region}.amazonaws.com" not in self.endpoint:
                raise AssertionError('Endpoint has to be {}'.format(
                        f"s3.{self.region}.amazonaws.com"
                    )
                )

        _client = (
            Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                region=self.region,
            )
            if not check_nas(self.endpoint)
            else NAS(self.endpoint)
        )
        if _client.bucket_exists(self.bucket_name):
            objects = _client.list_objects(self.bucket_name, prefix="metadata/")
            _metadata = None
            for obj in objects:
                _metadata = _client.get_object(self.bucket_name, obj.object_name)
                break
            if not _metadata:
                return

            metadata_dict = json.loads(_metadata.read().decode("utf-8"))
            if self.endpoint != metadata_dict["endpoint"]:
                raise ValueError(
                    "Already created endpoint({}) doesn't current endpoint str({})"
                    " It may occurs permission denied error".format(
                        metadata_dict["endpoint"], self.endpoint
                    )
                )

            self.compressor = metadata_dict["compressor"]
            self.attributes = [
                DataAttribute(**item) for item in metadata_dict["attributes"]
            ]
        else:
            logger.info(
                "{} {} is not exist!".format(self.dataset_name, str(self.additional))
            )

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
            raise ValueError(
                "dataset_name {} is empty or not str type".format(self.dataset_name)
            )
        if not isinstance(self.additional, dict):
            raise TypeError("additional is not dict type")

        key = self.dataset_name + json.dumps(self.additional, indent=4, sort_keys=True)
        return self.type + hashlib.md5(key.encode("utf-8")).hexdigest()

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__class__.__base__(**self.__dict__).__dict__)
        output["dataset_name"] = self.metadata.dataset_name
        output["additional"] = self.metadata.additional
        output["attributes"] = [_attribute.to_dict() for _attribute in self.attributes]
        output["compressor"] = self.metadata.compressor
        return output

    @classmethod
    def from_json_file(cls, json_file):
        """
        Constructs a `Config` from the path to a json file of parameters.

        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`DataConfig`: An instance of a configuration object

        """
        config_dict = cls._dict_from_json_file(json_file)

        config_dict["attributes"] = [
            DataAttribute(**item) for item in config_dict["attributes"]
        ]

        return cls(**config_dict)

    def set_indexer(self, index):
        self.metadata.indexer.update(index)

    def set_files(self, files):
        self.metadata.filetype.append(files)

    @property
    def get_length(self):
        """
        Get length of dataset in ``DataConfig``

        Returns:
            :obj:`integer`: length of dataset

        """
        keys = list(self.metadata.indexer.keys())
        if len(keys) == 0:
            return 0
        else:
            return keys[-1]