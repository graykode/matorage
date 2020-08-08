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


class OptimizerTest(unittest.TestCase):
    optimizer_config = None
    optimizer_config_file = None
    optimizer_manager = None
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
        if self.optimizer_manager is not None:
            # delete bucket
            client = (
                Minio(**self.storage_config)
                if not self.check_nas(self.optimizer_config.endpoint)
                else NAS(self.optimizer_config.endpoint)
            )
            objects = client.list_objects(
                self.optimizer_config.bucket_name, recursive=True
            )
            for obj in objects:
                client.remove_object(self.optimizer_config.bucket_name, obj.object_name)
            client.remove_bucket(self.optimizer_config.bucket_name)

        if self.optimizer_config_file is not None:
            if os.path.exists(self.optimizer_config_file):
                os.remove(self.optimizer_config_file)
