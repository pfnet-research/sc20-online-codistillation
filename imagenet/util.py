# Copyright 2020 Preferred Networks, Inc.
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

class Stats(object):
    def __init__(self):
        self.sum = 0
        self.n = 0

    def add(self, x):
        self.sum += x
        self.n += 1

    def average(self):
        if self.n == 0:
            return -1.0
        else:
            return self.sum / self.n

    def clear(self):
        self.sum = 0
        self.n = 0
