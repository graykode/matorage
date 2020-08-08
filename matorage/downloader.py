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
from minio import ResponseError

from matorage.connector import MTRConnector


class Downloader(MTRConnector):
    r""" File Storage downloader class with multi thread.
        MinIO is thread-safety, according to document.
        Although Python Global Interpreter Lock(GIL), multi thread can benefit greatly from file IO.
    """

    def __init__(self, client, bucket, num_worker_threads):
        super(Downloader, self).__init__(client, bucket, num_worker_threads)

    def do_job(self, local_file, remote_file):
        if isinstance(remote_file, str):
            minio_key = os.path.basename(remote_file)
        try:
            self._client.fget_object(self._bucket, minio_key, local_file)
        except ResponseError as err:
            print(err)
