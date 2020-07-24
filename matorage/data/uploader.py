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

import io
import os
import uuid
from minio import ResponseError

from matorage.connector import MRTConnector

class DataUploader(MRTConnector):
    r""" File Storage uploader class with multi thread.
        MinIO is thread-safety, according to document.
        Although Python Global Interpreter Lock(GIL), multi thread can benefit greatly from file IO.
    """

    def do_job(self, fileitem, filename):

        if isinstance(filename, str):
            minio_key = os.path.basename(filename)
        try:
            if not self._inmemory:
                self._client.fput_object(
                    self._bucket, minio_key, fileitem,
                    part_size=self._multipart_upload_size
                )
                os.remove(fileitem)
            else:
                fileimage = io.BytesIO(fileitem)
                self._client.put_object(
                    self._bucket, minio_key, fileimage, len(fileitem),
                    part_size=self._multipart_upload_size
                )
        except ResponseError as err:
            print(err)