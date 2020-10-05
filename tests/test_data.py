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
import shutil
import unittest
from minio import Minio
from environs import Env
from urllib.parse import urlsplit
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from matorage.nas import NAS
from matorage.data.orm import *


def delete_database(bucket_name):
    database = create_engine(
        f'postgresql://minio:miniosecretkey@127.0.0.1:5433/matorage'
    )
    Session = sessionmaker(bind=database)
    session = Session()

    session.query(Attributes).filter_by(bucket_id=bucket_name).delete()
    session.query(Indexer).filter_by(bucket_id=bucket_name).delete()
    session.query(Files).filter_by(bucket_id=bucket_name).delete()
    session.query(Bucket).filter_by(id=bucket_name).delete()

    session.commit()

    session.close()
    database.dispose()

class DataTest(unittest.TestCase):
    data_config = None
    data_config_file = None
    data_saver = None
    dataset = None
    storage_config = {
        "endpoint": "127.0.0.1:9001",
        "database" : "127.0.0.1:5433",
        "access_key": "minio",
        "secret_key": "miniosecretkey",
        "secure": False,
    }
    nas_config = {
        "endpoint": "/tmp/unittest",
        "database": "127.0.0.1:5433",
        "access_key": "minio",
        "secret_key": "miniosecretkey",
    }
    cache_folder_path = "/tmp/unittest_cache"

    def check_nas(self, endpoint):
        _url_or_path = "//" + endpoint
        u = urlsplit(_url_or_path)
        if u.path:
            return True
        if u.netloc:
            return False
        raise ValueError("This endpoint is not suitable.")

    # def setUp(self):
    #     if os.path.exists(self.cache_folder_path):
    #         shutil.rmtree(self.cache_folder_path)

    def tearDown(self):
        if self.data_saver is not None:
            # delete bucket
            client = (
                Minio(
                    endpoint=self.storage_config['endpoint'],
                    access_key=self.storage_config['access_key'],
                    secret_key=self.storage_config['secret_key'],
                    secure=self.storage_config['secure']
                )
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

            # remove database
            delete_database(bucket_name=self.data_config.bucket_name)

        if self.dataset is not None and not self.check_nas(self.dataset.config.endpoint):
            if os.path.exists(self.cache_folder_path):
                shutil.rmtree(self.cache_folder_path)

        if self.data_config_file is not None:
            if os.path.exists(self.data_config_file):
                os.remove(self.data_config_file)


@unittest.skipIf(
    'access_key' not in os.environ or 'secret_key' not in os.environ, 'S3 Skip'
)
class DataS3Test(unittest.TestCase):
    cache_folder_path = "/tmp/unittest_cache"

    data_config = None
    data_config_file = None
    data_saver = None
    dataset = None
    storage_config = None

    def tearDown(self):
        if self.data_saver is not None:
            # delete bucket
            client = Minio(
                endpoint=self.storage_config['endpoint'],
                access_key=self.storage_config['access_key'],
                secret_key=self.storage_config['secret_key'],
                secure=self.storage_config['secure']
            )
            objects = client.list_objects(self.data_config.bucket_name, recursive=True)
            for obj in objects:
                client.remove_object(self.data_config.bucket_name, obj.object_name)
            client.remove_bucket(self.data_config.bucket_name)

            # remove on local file
            for _file in self.data_saver.get_downloaded_dataset:
                if os.path.exists(_file):
                    os.remove(_file)

            # remove database
            delete_database(bucket_name=self.data_config.bucket_name)