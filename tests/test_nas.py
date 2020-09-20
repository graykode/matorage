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

from tests.test_storage import StorageTest

from matorage.nas import NAS

class DataSaverTest(StorageTest, unittest.TestCase):
    minio_config = Minio(
        endpoint="127.0.0.1:9000",
        access_key="minio",
        secret_key="miniosecretkey",
        secure=False,
    )
    nas_config = NAS(path="/tmp")

    def _create_bucket(self):
        self.minio_config.make_bucket('testminio')
        self.nas_config.make_bucket('testnas')

    def _get_obj_names(self, generator):
        res = sorted([g.object_name for g in generator])
        return res

    def _check_assertEqual(self, prefix=''):
        for r in [True, False]:
            self.assertEqual(
                self._get_obj_names(
                    self.minio_config.list_objects('testminio', prefix=prefix, recursive=r)
                ),
                self._get_obj_names(
                    self.nas_config.list_objects('testnas', prefix=prefix, recursive=r)
                )
            )

    def test_bucket_exists(self):
        self._create_bucket()
        self.assertEqual(
            self.minio_config.bucket_exists('testminio'),
            self.nas_config.bucket_exists('testnas')
        )

    def test_fput_object_with_list_objects(self):
        self._create_bucket()

        with open('test', 'w') as f:
            f.write('this is test')

        for _object_name in ['test', 'metadata/test']:
            self.minio_config.fput_object(
                bucket_name='testminio',
                object_name=_object_name,
                file_path='test'
            )
        for _object_name in ['test', 'metadata/test']:
            self.nas_config.fput_object(
                bucket_name='testnas',
                object_name=_object_name,
                file_path='test'
            )

        self._check_assertEqual(prefix='')
        self._check_assertEqual(prefix='metadata/')
        self._check_assertEqual(prefix='metadata')
        self._check_assertEqual(prefix='meta')

        self._check_assertEqual(prefix='metadata/t')
        self._check_assertEqual(prefix='metadata/test')

        self.assertEqual(
            self._get_obj_names(
                self.minio_config.list_objects('testminio', prefix='metadata/test/')
            ),
            self._get_obj_names(
                self.nas_config.list_objects('testnas', prefix='metadata/test/')
            )
        )

        self.assertNotEqual(
            self._get_obj_names(
                self.minio_config.list_objects('testminio', prefix='metadata/test/', recursive=True)
            ),
            self._get_obj_names(
                self.nas_config.list_objects('testnas', prefix='metadata/test/', recursive=True)
            )
        )

def suite():
    suties = unittest.TestSuite()
    suties.addTests(unittest.makeSuite(DataSaverTest))
    return suties


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
