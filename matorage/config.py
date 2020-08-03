
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

_KB = 1024
"""The size of a Kilobyte in bytes"""

_MB = 1024 * _KB
"""The size of a Megabyte in bytes"""

from matorage.serialize import Serialize

class MTRConfig(Serialize):
    """
    Storage connector configuration classes.
    For MinIO, see `this page <https://docs.min.io/docs/python-client-api-reference.html>`_ for more details.

    .. code-block:: python

        from matorage import MTRConfig

        storage_config = MTRConfig(
            endpoint='127.0.0.1:9000',
            access_key='minio',
            secret_key='miniosecretkey'
        )

    Args:
        endpoint (:obj:`string`, **require**):
            S3 object storage endpoint. or If use NAS setting, NAS folder path.
        access_key (:obj:`string`, optional, defaults to `None`):
            Access key for the object storage endpoint. (Optional if you need anonymous access).
        secret_key (:obj:`string`, optional, defaults to `None`):
            Secret key for the object storage endpoint. (Optional if you need anonymous access).
        secure (:obj:`boolean`, optional, defaults to `False`):
            Set this value to True to enable secure (HTTPS) access. (Optional defaults to False unlike the original MinIO).
        max_object_size (:obj:`integer`, optional, defaults to `10MB`):
            One object file is divided into `max_object_size` and stored.

    """

    def __init__(self, **kwargs):

        # MinIO configuration
        self.endpoint = kwargs.pop("endpoint", None)
        self.access_key = kwargs.pop("access_key", None)
        self.secret_key = kwargs.pop("secret_key", None)
        self.secure = kwargs.pop("secure", False)
        self.max_object_size = kwargs.pop("max_object_size", 10 * _MB)
