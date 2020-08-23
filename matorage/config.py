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
from minio import Minio

from matorage.nas import NAS
from matorage.utils import check_nas, logger
from matorage.serialize import Serialize


class StorageConfig(Serialize):
    """
    Storage connector configuration classes.
    For MinIO, see `this page <https://docs.min.io/docs/python-client-api-reference.html>`_ for more details.

    Args:
        endpoint (:obj:`string`, **require**):
            S3 object storage endpoint. or If use NAS setting, NAS folder path.
        access_key (:obj:`string`, optional, defaults to `None`):
            Access key for the object storage endpoint. (Optional if you need anonymous access).
        secret_key (:obj:`string`, optional, defaults to `None`):
            Secret key for the object storage endpoint. (Optional if you need anonymous access).
        secure (:obj:`boolean`, optional, defaults to `False`):
            Set this value to True to enable secure (HTTPS) access. (Optional defaults to False unlike the original MinIO).

    Examples::

        from matorage import StorageConfig
        storage_config = StorageConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey'
        )


    """

    def __init__(self, **kwargs):

        # MinIO configuration
        self.endpoint = kwargs.pop("endpoint", None)
        self.access_key = kwargs.pop("access_key", None)
        self.secret_key = kwargs.pop("secret_key", None)
        self.secure = kwargs.pop("secure", False)
        self.region = kwargs.pop("region", None)

    def get_datasets(self):
        """
        Get all datasets from endpoint storage

        Returns:
            :obj:`list`:`dict` type of item(`dataset_name`, `additional`, `compressor`, `attributes`)
        """
        res = []

        _client = self._create_client()
        buckets = _client.list_buckets()
        for bucket in buckets:
            if bucket.name.startswith('dataset'):
                _metadata_names = _client.list_objects(bucket.name, prefix="metadata/")
                for _metadata_name in _metadata_names:
                    _metadata = _client.get_object(
                        bucket.name, _metadata_name.object_name
                    )
                    metadata_dict = json.loads(_metadata.read().decode("utf-8"))
                    res.append({
                        'dataset_name': metadata_dict['dataset_name'],
                        'additional': metadata_dict['additional'],
                        'compressor': metadata_dict['compressor'],
                        'attributes': metadata_dict['attributes'],
                    })
                    break
        return res

    def _get_type(self, type):
        res = []

        _client = self._create_client()
        buckets = _client.list_buckets()
        for bucket in buckets:
            if bucket.name.startswith(type):
                _metadata = _client.get_object(bucket.name, "metadata.json")
                metadata_dict = json.loads(_metadata.read().decode("utf-8"))
                res.append({
                    f'{type}_name': metadata_dict[f'{type}_name'],
                    'additional': metadata_dict['additional'],
                    'compressor': metadata_dict['compressor'],
                    type: metadata_dict[type],
                })

        return res

    def get_models(self):
        """
        Get all models from endpoint storage

        Returns:
            :obj:`list`:`dict` type of item(`model_name`, `additional`, `compressor`, `model`)
        """

        return self._get_type(type='model')

    def get_optimizers(self):
        """
        Get all optimizers from endpoint storage

        Returns:
            :obj:`list`:`dict` type of item(`model_name`, `additional`, `compressor`, `optimizer`)
        """

        return self._get_type(type='optimizer')

    def _create_client(self):
        return (
            Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                region=self.region,
            )
            if not check_nas(self.endpoint)
            else NAS(self.endpoint)
        )
