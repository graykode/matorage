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

class DataTest(unittest.TestCase):
    data_config = None
    data_saver = None
    storage_config = {
        'endpoint': '127.0.0.1:9000',
        'access_key': 'minio',
        'secret_key': 'miniosecretkey',
        'secure': False
    }

    def tearDown(self):
        if self.data_config is not None:
            # delete bucket
            client = Minio(**self.storage_config)
            objects = client.list_objects(self.data_config.bucket_name)
            for obj in objects:
                client.remove_object(self.data_config.bucket_name, obj.object_name)
            client.remove_bucket(self.data_config.bucket_name)
        if self.data_saver is not None:
            # remove on local file
            for _file in self.data_saver.get_filelist:
                if os.path.exists(_file):
                    os.remove(_file)
