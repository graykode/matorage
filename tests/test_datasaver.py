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
import numpy as np

from tests.test_data import DataTest, DataS3Test

from matorage.data.config import DataConfig
from matorage.data.saver import DataSaver
from matorage.data.attribute import DataAttribute


class DataSaverTest(DataTest, unittest.TestCase):
    def test_dataconfig_one_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_one_attribute",
            attributes=DataAttribute("x", "uint8", (1)),
        )

    def test_dataconfig_one_attribute_with_tuple_attributes(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_one_attribute_with_tuple_attributes",
            attributes=("x", "uint8", (1)),
        )

    def test_reload_dataconfig(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_reload_dataconfig",
            attributes=DataAttribute("x", "uint8", (1)),
        )
        self.data_config_file = "data_config_file.json"
        self.data_config.to_json_file(self.data_config_file)

        self.data_config = None

        self.data_config = DataConfig.from_json_file(self.data_config_file)

    def test_dataconfig_two_attributes(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_two_attributes",
            attributes=[
                DataAttribute("x", "uint8", (1)),
                DataAttribute("y", "uint8", (1)),
            ],
        )

    def test_dataconfig_two_attribute_with_tuple_attributes(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_two_attribute_with_tuple_attributes",
            attributes=[("x", "uint8", (1)), ("y", "uint8", (1))],
        )

    def test_dataconfig_attributes_already_exist(self):
        with self.assertRaisesRegex(KeyError, "is already exist in"):
            self.data_config = DataConfig(
                **self.storage_config,
                dataset_name="test_dataconfig_attributes_already_exist",
                attributes=[
                    DataAttribute("x", "uint8", (1)),
                    DataAttribute("x", "uint8", (1)),
                ],
            )

    def test_dataconfig_string_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_string_attribute",
            attributes=[DataAttribute("x", "string", (1), itemsize=32)],
        )

    def test_dataconfig_bool_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_bool_attribute",
            attributes=[DataAttribute("x", "bool", (1))],
        )

    def test_dataconfig_int8_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_int8_attribute",
            attributes=[DataAttribute("x", "int8", (1))],
        )

    def test_dataconfig_int16_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_int16_attribute",
            attributes=[DataAttribute("x", "int16", (1))],
        )

    def test_dataconfig_int32_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_int32_attribute",
            attributes=[DataAttribute("x", "int32", (1))],
        )

    def test_dataconfig_uint8_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_uint8_attribute",
            attributes=[DataAttribute("x", "uint8", (1))],
        )

    def test_dataconfig_uint16_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_uint16_attribute",
            attributes=[DataAttribute("x", "uint16", (1))],
        )

    def test_dataconfig_uint32_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_uint32_attribute",
            attributes=[DataAttribute("x", "uint32", (1))],
        )

    def test_dataconfig_uint64_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_uint64_attribute",
            attributes=[DataAttribute("x", "uint64", (1))],
        )

    def test_dataconfig_float32_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_float32_attribute",
            attributes=[DataAttribute("x", "float32", (1))],
        )

    def test_dataconfig_float64_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_dataconfig_float64_attribute",
            attributes=[DataAttribute("x", "float64", (1))],
        )

    def test_datasaver_string_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_string_attribute",
            attributes=[DataAttribute("x", "string", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([["a", "b"], ["c", "d"], ["e", "f"]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_bool_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_bool_attribute",
            attributes=[DataAttribute("x", "bool", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[True, False], [False, True], [True, True]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_int8_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_int8_attribute",
            attributes=[DataAttribute("x", "int8", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_int16_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_int16_attribute",
            attributes=[DataAttribute("x", "int16", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_int32_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_int32_attribute",
            attributes=[DataAttribute("x", "int32", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_int64_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_int64_attribute",
            attributes=[DataAttribute("x", "int64", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_uint8_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_uint8_attribute",
            attributes=[DataAttribute("x", "uint8", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_uint16_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_uint16_attribute",
            attributes=[DataAttribute("x", "uint16", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_uint32_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_uint32_attribute",
            attributes=[DataAttribute("x", "uint32", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_uint64_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_uint64_attribute",
            attributes=[DataAttribute("x", "uint64", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_float32_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_float32_attribute",
            attributes=[DataAttribute("x", "float32", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_float64_attribute(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_float64_attribute",
            attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_inmemory(self):
        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_inmemory",
            attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config, inmemory=True)
        x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_zlib(self):
        for level in range(10):
            self.data_config = DataConfig(
                **self.storage_config,
                dataset_name="test_datasaver_zlib",
                attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
                compressor={"complevel": level, "complib": "zlib"},
            )
            self.data_saver = DataSaver(config=self.data_config)
            x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            self.assertEqual(x.shape, (3, 2))
            self.data_saver({"x": x})
            self.data_saver.disconnect()

    def test_datasaver_lzo(self):
        for level in range(10):
            self.data_config = DataConfig(
                **self.storage_config,
                dataset_name="test_datasaver_lzo",
                attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
                compressor={"complevel": level, "complib": "lzo"},
            )
            self.data_saver = DataSaver(config=self.data_config)
            x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            self.assertEqual(x.shape, (3, 2))
            self.data_saver({"x": x})
            self.data_saver.disconnect()

    def test_datasaver_bzip2(self):
        for level in range(10):
            self.data_config = DataConfig(
                **self.storage_config,
                dataset_name="test_datasaver_bzip2",
                attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
                compressor={"complevel": level, "complib": "bzip2"},
            )
            self.data_saver = DataSaver(config=self.data_config)
            x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            self.assertEqual(x.shape, (3, 2))
            self.data_saver({"x": x})
            self.data_saver.disconnect()

    def test_datasaver_blosc(self):
        for level in range(10):
            self.data_config = DataConfig(
                **self.storage_config,
                dataset_name="test_datasaver_blosc",
                attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
                compressor={"complevel": level, "complib": "blosc"},
            )
            self.data_saver = DataSaver(config=self.data_config)
            x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            self.assertEqual(x.shape, (3, 2))
            self.data_saver({"x": x})
            self.data_saver.disconnect()

    def test_datasaver_nas(self):

        self.data_config = DataConfig(
            endpoint="/tmp",
            dataset_name="test_datasaver_nas",
            attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_refresh(self):

        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_refresh",
            attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
        )
        for refresh in [False, True]:
            self.data_saver = DataSaver(config=self.data_config, refresh=refresh)
            x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            self.assertEqual(x.shape, (3, 2))
            self.data_saver({"x": x})
            self.data_saver.disconnect()


    def test_datasaver_filetype(self):
        from matorage.torch import Dataset

        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_filetype",
            attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})

        _file = open("test.txt", "w")
        _file.write('this is test')
        self.data_saver({"file": "test.txt"}, filetype=True)
        _file.close()

        self.data_saver.disconnect()

        self.dataset = Dataset(config=self.data_config)
        self.assertEqual(
            self.dataset.get_filetype_list, ["file"]
        )
        _local_filepath = self.dataset.get_filetype_from_key("file")
        with open(_local_filepath, 'r') as f:
            self.assertEqual(f.read(), 'this is test')


@unittest.skipIf(
    'access_key' not in os.environ or 'secret_key' not in os.environ, 'S3 Skip'
)
class DataS3SaverTest(DataS3Test, unittest.TestCase):
    def test_datasaver_s3(self):
        self.storage_config = {
            'endpoint': 's3.us-east-1.amazonaws.com',
            'access_key': os.environ['access_key'],
            'secret_key': os.environ['secret_key'],
            'region': 'us-east-1',
            'secure': False,
        }

        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_s3",
            attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})
        self.data_saver.disconnect()

    def test_datasaver_s3_filetype(self):
        from matorage.torch import Dataset

        self.storage_config = {
            'endpoint': 's3.us-east-1.amazonaws.com',
            'access_key': os.environ['access_key'],
            'secret_key': os.environ['secret_key'],
            'region': 'us-east-1',
            'secure': False,
        }

        self.data_config = DataConfig(
            **self.storage_config,
            dataset_name="test_datasaver_s3_filetype",
            attributes=[DataAttribute("x", "float64", (2), itemsize=32)],
        )
        self.data_saver = DataSaver(config=self.data_config)
        x = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(x.shape, (3, 2))
        self.data_saver({"x": x})

        _file = open("test.txt", "w")
        _file.write('this is test')
        self.data_saver({"file": "test.txt"}, filetype=True)
        _file.close()

        self.data_saver.disconnect()

        self.dataset = Dataset(config=self.data_config)
        self.assertEqual(
            self.dataset.get_filetype_list, ["file"]
        )
        _local_filepath = self.dataset.get_filetype_from_key("file")
        with open(_local_filepath, 'r') as f:
            self.assertEqual(f.read(), 'this is test')


def suite():
    suties = unittest.TestSuite()
    suties.addTests(unittest.makeSuite(DataSaverTest))
    suties.addTests(unittest.makeSuite(DataS3SaverTest))
    return suties


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
