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
from glob import iglob


class Obj(object):
    def __init__(self, object_name):
        self.object_name = object_name


class NAS(object):
    def __init__(self, path):
        self.path = path

    def bucket_exists(self, bucket_name):
        return os.path.exists(os.path.join(self.path, bucket_name))

    def fget_object(self, bucket_name, object_name, file_path):
        raise NotImplementedError

    def fput_object(self, bucket_name, object_name, file_path, part_size=None):
        _filename = os.path.join(self.path, bucket_name, object_name)
        os.makedirs(os.path.dirname(_filename), exist_ok=True)
        shutil.copyfile(src=file_path, dst=_filename)

    def get_object(self, bucket_name, object_name):
        _filename = os.path.join(self.path, bucket_name, object_name)
        return open(_filename, "rb")

    def put_object(self, bucket_name, object_name, data, length, part_size=None):
        _filename = os.path.join(self.path, bucket_name, object_name)
        os.makedirs(os.path.dirname(_filename), exist_ok=True)
        data.seek(0)
        with open(_filename, "wb") as f:
            shutil.copyfileobj(data, f, length=length)

    def list_objects(self, bucket_name, prefix="", recursive=False):
        _base = os.path.join(self.path, bucket_name)
        _query = os.path.join(self.path, bucket_name, prefix)
        objects = []

        if '/' in _query:
            _query = _query.rsplit('/', 1)[0] + '/'
        else:
            _query = _base

        for f in iglob(f"{_query}/**", recursive=recursive):
            if os.path.isdir(f):
                f = f.replace(f"{_base}/", '')
                if f and not recursive:
                    objects.append(f"{f}/")
            else:
                f = f.replace(f"{_base}/", '')
                if f:
                    objects.append(f)

        return [Obj(o) for o in objects if o.startswith(prefix)]

    def make_bucket(self, bucket_name, location='us-east-1'):
        os.makedirs(os.path.join(self.path, bucket_name))

    def remove_bucket(self, bucket_name):
        shutil.rmtree(os.path.join(self.path, bucket_name))

    def remove_object(self, bucket_name, object_name):
        _path = os.path.join(self.path, bucket_name, object_name)
        if os.path.isdir(_path):
            shutil.rmtree(_path)
        else:
            os.remove(_path)