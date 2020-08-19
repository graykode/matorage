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

import os
import unittest
from minio import Minio
from urllib.parse import urlsplit

from matorage.nas import NAS


class DataTest(unittest.TestCase):
    data_config = None
    data_config_file = None
    data_saver = None
    dataset = None
    storage_config = {
        "endpoint": "127.0.0.1:9000",
        "access_key": "minio",
        "secret_key": "miniosecretkey",
        "secure": False,
    }

    def check_nas(self, endpoint):
        _url_or_path = "//" + endpoint
        u = urlsplit(_url_or_path)
        if u.path:
            return True
        if u.netloc:
            return False
        raise ValueError("This endpoint is not suitable.")

    def tearDown(self):
        if self.data_saver is not None:
            # delete bucket
            client = (
                Minio(**self.storage_config)
                if not self.check_nas(self.data_config.endpoint)
                else NAS(self.data_config.endpoint)
            )
            objects = client.list_objects(self.data_config.bucket_name, recursive=True)
            for obj in objects:
                client.remove_object(self.data_config.bucket_name, obj.object_name)
            client.remove_bucket(self.data_config.bucket_name)

            # remove on local file
            for _file in self.data_saver.get_downloaded_dataset:
                if os.path.exists(_file):
                    os.remove(_file)

        if self.dataset is not None:
            if os.path.exists(self.dataset.cache_path):
                os.remove(self.dataset.cache_path)

        if self.data_config_file is not None:
            if os.path.exists(self.data_config_file):
                os.remove(self.data_config_file)


@unittest.skipIf(
    'access_key' not in os.environ or 'secret_key' not in os.environ, 'S3 Skip'
)
class DataS3Test(unittest.TestCase):

    data_config = None
    data_config_file = None
    data_saver = None
    dataset = None
    storage_config = None

    def tearDown(self):
        if self.data_saver is not None:
            # delete bucket
            client = Minio(**self.storage_config)
            objects = client.list_objects(self.data_config.bucket_name, recursive=True)
            for obj in objects:
                client.remove_object(self.data_config.bucket_name, obj.object_name)
            client.remove_bucket(self.data_config.bucket_name)

            # remove on local file
            for _file in self.data_saver.get_downloaded_dataset:
                if os.path.exists(_file):
                    os.remove(_file)
