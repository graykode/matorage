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

import unittest
from environs import Env
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from matorage.data.orm import *

database = '127.0.0.1:5432'

class DataSaverTest(unittest.TestCase):
    session = None

    def setUp(self):
        self.env = Env()
        self.env.read_env('../docker/.env')
        assert self.env("MATORAGE_ACCESS_KEY")
        assert self.env("MATORAGE_SECRET_KEY")

        _access_key = self.env("MATORAGE_ACCESS_KEY")
        _secret_key = self.env("MATORAGE_SECRET_KEY")

        self.database = create_engine(
            f'postgresql://{_access_key}:{_secret_key}@{database}/matorage'
        )

    def tearDown(self):
        if self.session:
            self.session.query(Attributes).filter_by(bucket_id='test_bucket').delete()
            self.session.query(Indexer).filter_by(bucket_id='test_bucket').delete()
            self.session.query(Bucket).filter_by(id='test_bucket').delete()

            self.session.commit()
            self.session.close()

        if self.database:
            self.database.dispose()

    def test_create(self):
        bucket = Bucket(
            id='test_bucket',
            additional=str({
                "test" : "test"
            }),
            dataset_name='test_dataset',
            endpoint='127.0.0.1:9001',
            compressor=str({
                "complevel": 4,
                "complib": "zlib"
            }),
            filetype=str(['test1.h5', 'test2.h5']),
            sagemaker=True
        )

        attribute1 = Attributes(
            name='test_attributes',
            type='float64',
            shape=str((1, 2,)),
            itemsize=32,
            bucket_id='test_bucket'
        )

        attribute2 = Attributes(
            name='test_attributes',
            type='float64',
            shape=str((1, 2,)),
            itemsize=32,
            bucket_id='test_bucket'
        )

        indexer1 = Indexer(
            indexer_end=100,
            length=100,
            name='test1.h5',
            bucket_id='test_bucket'
        )

        indexer2 = Indexer(
            indexer_end=200,
            length=200,
            name='test2.h5',
            bucket_id='test_bucket'
        )

        Session = sessionmaker(bind=self.database)
        self.session = Session()
        self.session.add(bucket)
        self.session.add(attribute1)
        self.session.add(attribute2)
        self.session.add(indexer1)
        self.session.add(indexer2)

        self.session.commit()