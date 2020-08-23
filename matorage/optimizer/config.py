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

import json
import copy
import hashlib
from minio import Minio

from matorage.nas import NAS
from matorage.config import StorageConfig
from matorage.utils import check_nas, logger


class OptimizerConfig(StorageConfig):
    """
    Optimizer configuration classes. This class overrides ``StorageConfig``.

    Args:
        endpoint (:obj:`string`, **require**):
            S3 object storage endpoint. or If use NAS setting, NAS folder path.
        access_key (:obj:`string`, optional, defaults to `None`):
            Access key for the object storage endpoint. (Optional if you need anonymous access).
        secret_key (:obj:`string`, optional, defaults to `None`):
            Secret key for the object storage endpoint. (Optional if you need anonymous access).
        secure (:obj:`boolean`, optional, defaults to `False`):
            Set this value to True to enable secure (HTTPS) access. (Optional defaults to False unlike the original MinIO).

        optimizer_name (:obj:`string`, **require**):
            optimizer name.
        additional (:obj:`dict`, optional, defaults to ``{}``):
            Parameters for additional description of optimizers. The key and value of the dictionay can be specified very freely.
        compressor (:obj:`dict`, optional, defaults to :code:`{"complevel" : 0, "complib" : "zlib"}`):
            Optimizer compressor option. It consists of a dict type that has complevel and complib as keys.
            For further reference, read `pytable's Filter <http://www.pytables.org/usersguide/libref/helper_classes.html#tables.Filters>`_.

            - complevel (:obj:`integer`, defaults to 0) : compressor level(0~9). The larger the number, the more compressed it is.
            - complib (:obj:`string`, defaults to 'zlib') : compressor library. choose in zlib, lzo, bzip2, blosc

    Examples::

        from matorage import OptimizerConfig

        optimizer_config = OptimizerConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey',
            optimizer_name='testoptimizer',
            additional={
                "version" : "1.0.1"
            }
        )

        optimizer_config.to_json_file('testoptimizer.json')
        optimizer_config2 = OptimizerConfig.from_json_file('testoptimizer.json')

    """

    def __init__(self, **kwargs):
        super(OptimizerConfig, self).__init__(**kwargs)
        self.type = "optimizer"

        self.optimizer_name = kwargs.pop("optimizer_name", None)
        self.additional = kwargs.pop("additional", {})
        self.compressor = kwargs.pop("compressor", {"complevel": 0, "complib": "zlib"})

        self.bucket_name = self._hashmap_transfer()

        self.metadata = {
            "endpoint": self.endpoint,
            "optimizer_name": self.optimizer_name,
            "additional": self.additional,
            "compressor": self.compressor,
            "optimizer": {},
            "scheduler": {},
        }

        self._check_all()

    def _check_all(self):
        """
        Check all class variable is fine.

        """
        self._check_bucket()

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
        _client = (
            Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                region=self.region
            )
            if not check_nas(self.endpoint)
            else NAS(self.endpoint)
        )
        if _client.bucket_exists(self.bucket_name):
            try:
                _metadata = _client.get_object(self.bucket_name, "metadata.json")
            except:
                _client.remove_bucket(self.bucket_name)
                raise FileNotFoundError(
                    "metadata.json is not in bucket name {}"
                    ", So this bucket will be removed".format(self.bucket_name)
                )

            metadata_dict = json.loads(_metadata.read().decode("utf-8"))
            if self.endpoint != metadata_dict["endpoint"]:
                raise ValueError(
                    "Already created endpoint({}) doesn't current endpoint str({})"
                    " It may occurs permission denied error".format(
                        metadata_dict["endpoint"], self.endpoint
                    )
                )

            self.compressor = metadata_dict["compressor"]
            self.metadata = metadata_dict
        else:
            logger.info(
                "{} {} is not exist!".format(self.optimizer_name, str(self.additional))
            )

    def _hashmap_transfer(self):
        """
        Get unikey bucket name with `optimizer_name` and `additional`

        Returns:
            :obj: `str`:
        """
        if not isinstance(self.optimizer_name, str):
            raise ValueError(
                "optimizer_name {} is empty or not str type".format(self.optimizer_name)
            )
        if not isinstance(self.additional, dict):
            raise TypeError("additional is not dict type")

        key = (
            self.type
            + self.optimizer_name
            + json.dumps(self.additional, indent=4, sort_keys=True)
        )
        return self.type + hashlib.md5(key.encode("utf-8")).hexdigest()

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__class__.__base__(**self.__dict__).__dict__)
        output["type"] = self.type
        output["optimizer_name"] = self.optimizer_name
        output["additional"] = self.additional
        output["compressor"] = self.metadata["compressor"]
        return output
