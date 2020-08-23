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
from setuptools import setup
from sutils import find_name, get_setuptools, check_torch_tf_version

project_name = "matorage"
version = os.environ.get('MATORAGE_VERSION', '0.0.0')

if __name__ == "__main__":
    check_torch_tf_version()

    project_name = find_name()

    with open('README.md', 'r') as t:
        README = t.read()

    setup(
        # Project Name, Version
        name=project_name,
        version=version,
        long_description=README,
        long_description_content_type='text/markdown',
        # Author
        license="Apache License, Version 2.0",
        author="TaeHwan-Jung",
        author_email="nlkey2022@gmail.com",
        description="matorage is Matrix or Tensor(multidimensional matrix) "
                    "Object Storage with high availability "
                    "distributed systems for Deep Learning framework.",
        url="https://github.com/graykode/matorage",
        # Platform, Requires
        python_requires=">=3.5",
        platforms=["any"],
        project_urls={
            "Documentation": "https://matorage.readthedocs.io/en/stable/",
            "Source Code": "https://github.com/graykode/matorage",
        },
        **get_setuptools()
    )