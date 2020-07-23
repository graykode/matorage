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

import torch
from torch.utils.data import Sampler

class MTRSampler(Sampler):
    r"""MTRSampler class for Pytorch Dataset Sampler
        It is difficult to get indexes from other distributed files due to network I/O.
        For examples,
            files = [100.h5, 200.h5 , 300.h5, 400.h5, 500.h5, 600.h5] (0~100 indices : 100.h5 ... 500~600 indices : 600.h5)
            If batch index quires [101,201,301,401,501], We download all files every steps.
        So we use Local Shuffle Algorithm:
            files = shuffle(files) >> [200.h5, 300.h5, 100.h5, 400.h5, 600.h5, 500.h5]
            set window size = 2
            files = [[200.h5, 300.h5], [100.h5, 400.h5], [600.h5, 500.h5]]
            files = [
                        local_shuffle([200.h5, 300.h5]),
                        local_shuffle([100.h5, 400.h5]),
                        local_shuffle([600.h5, 500.h5]])
                    ]

        Note:
            Deep Learning Framework Type : Pytorch

        Args:
            ends (:obj:`list`, `require`):
                list of integer value for end indexes.
                In the above example, [100, 200, 300, 400, 500, 600].
    """
    def __init__(self, ends):
        self.ends = ends

    def __iter__(self):
        generators = torch.randperm(self.ends[0])
        for idx in range(1, len(self.ends)):
            generators = torch.cat([
                generators,
                torch.randperm(self.ends[idx] - self.ends[idx - 1]) + self.ends[idx - 1]
            ], dim=0)
        return iter(generators)

    def __len__(self):
        return self.ends[-1]