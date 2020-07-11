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

from setuptools import setup
from sutils import find_name, get_setuptools

project_name = "matorage"
version = "0.1.0"

if __name__ == "__main__":
    project_name = find_name()
    setup(
        # Project Name, Version
        name=project_name,
        version=version,
        # Author
        license="Apache License, Version 2.0",
        author="TaeHwan-Jung",
        author_email="nlkey2022@gmail.com",
        description="",
        url="",
        # Platform, Requires
        python_requires=">=3.5",
        platforms=["any"],
        **get_setuptools()
    )