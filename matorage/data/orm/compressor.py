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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class Compressor(Base):
    __tablename__ = 'compressor'
    id = Column(Integer, primary_key=True, nullable=False)
    complevel = Column(Integer, nullable=False)
    complib = Column(String(255), nullable=False)

    def __repr__(self):
        return "<Compressor(" \
                   "id='%d', " \
                   "complevel='%d', " \
                   "complib='%s'" \
                ")>" % (
            self.id, self.additional, self.dataset_name
        )