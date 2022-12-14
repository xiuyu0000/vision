# Copyright 2021 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Test image io """

from matplotlib.image import imread

iamge_path = "tests/ut/common/images/avatar.origin.png"


def test_imread():
    out = imread(iamge_path)
    assert out.shape == (575, 575, 4)
