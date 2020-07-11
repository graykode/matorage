
# Original Code
# https://github.com/PyTables/PyTables/blob/master/setup.py
# See https://github.com/graykode/matorage/blob/0.1.0/NOTICE
# modified by TaeHwan Jung(@graykode)

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

import sys
from setuptools import find_packages

def find_name(name="matorage"):
    """If "--name-with-python-version" is on the command line then
    append "-pyX.Y" to the base name"""
    if "--name-with-python-version" in sys.argv:
        name += "-py%i.%i" % (sys.version_info[0], sys.version_info[1])
        sys.argv.remove("--name-with-python-version")
    return name

def get_requirements(file='../requirements.txt'):
    # Fetch the requisites
    with open(file) as f:
        requirements = f.read().splitlines()
    return requirements

def get_setuptools():
    setuptools_kwargs = {}

    setuptools_kwargs['zip_safe'] = False
    setuptools_kwargs['install_requires'] = get_requirements()

    # Detect packages automatically.
    setuptools_kwargs['packages'] = find_packages(exclude=['*.bench'])

    return setuptools_kwargs