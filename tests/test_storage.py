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


class StorageTest(unittest.TestCase):
    minio_config = None
    nas_config = None
    storage_config = {
        "endpoint": "127.0.0.1:9001",
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
        if self.minio_config:
            objects = self.minio_config.list_objects('testminio', recursive=True)
            for obj in objects:
                self.minio_config.remove_object('testminio', obj.object_name)
            self.minio_config.remove_bucket('testminio')

        if self.nas_config:
            objects = self.nas_config.list_objects('testnas', recursive=True)
            for obj in objects:
                self.nas_config.remove_object('testnas', obj.object_name)
            self.nas_config.remove_bucket('testnas')

        if os.path.exists('test'):
            os.remove('test')