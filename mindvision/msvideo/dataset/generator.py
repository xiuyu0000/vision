# Copyright 2022
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
""" Video Dataset Generator. """

import random
import imageio.v3 as iio
import numpy as np
import decord


class DatasetGenerator:
    """
    Dataset generator for getting video path and its corresponding label.
    Args:
        path(list): Video file path list.
        label(list): The label of each video,
        seq(int): The number of frames of the intercepted video.
        mode(str): Frame fetching method, options:["part", "discrete", "average", "interval"].
        suffix(str): Format of video file. options:["picture", "video"].
        align(boolean): The video contains multiple actions.
        frame_interval(int): Interval between sampling frames.
        num_clips(int): The number of samples of a video.
    """

    def __init__(self, path, label, seq=16, mode="part", suffix="video", align=False, frame_interval=1, num_clips=1):
        """Init Video Generator."""
        self.path = path
        self.label = label
        self.seq = seq
        self.mode = mode
        self.suffix = suffix
        self.align = align
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def __getitem__(self, item):
        """Get the video and label for each item."""
        if self.suffix == "picture":
            frame_path_list = self.path[item]
            num_frame = len(frame_path_list)
            video = []
            for frame_path in frame_path_list:
                with open(frame_path, "rb")as f:
                    content = f.read()
                frame = iio.imread(content, index=None, format_hint=".jpg")
                video.append(frame)
            video = np.concatenate(video, axis=0)
        if self.suffix == "video":
            video = decord.VideoReader(self.path[item], num_threads=8)
            num_frame = video.__len__()
        label = self.label[item]
        action_start = 0
        if self.align:
            pos_start = self.label[item][1]
            pos_end = self.label[item][2]
            frame_len = num_frame
            action_start = int(pos_start * frame_len)
            num_frame = int(pos_end * frame_len - pos_start * frame_len)
            label = self.label[item][0]
        sample_list = self.__sampler__(num_frame=num_frame, action_start=action_start)
        if self.suffix == "picture":
            video = [video[sample_list[i]] for i in range(self.num_clips)]
        if self.suffix == "video":
            video = [video.get_batch(sample_list[i]).asnumpy() for i in range(self.num_clips)]
        video = np.concatenate(video, axis=0)
        return video, label

    def __len__(self):
        """Get the the size of dataset."""
        return len(self.path)

    def __sampler__(self, num_frame, action_start):
        """Get video sampling index."""
        region_len = self.seq * self.frame_interval + self.num_clips
        sample_list = []
        if self.mode == "part":
            start = random.sample(range(0, abs(num_frame - region_len) + 1), self.num_clips)
            sample_list = [[start[j] + i for i in range(self.seq)] for j in range(self.num_clips)]
        if self.mode == "discrete":
            cnt = self.num_clips
            while cnt > 0:
                cnt -= 1
                sample = random.sample(list(range(num_frame)), self.seq)
                sample.sort()
                sample_list.append(sample)
        if self.mode == "average":
            interval = num_frame // self.seq
            offset = random.sample(range(interval + 1), self.num_clips)
            sample_list = [[offset[j] + i * interval for i in range(self.seq)] for j in range(self.num_clips)]
        if self.mode == "interval":
            offset = random.sample(range(abs(num_frame - region_len) + 1), self.num_clips)
            sample_list = [[offset[j] + i * self.frame_interval for i in range(self.seq)] for j in
                           range(self.num_clips)]
        sample_map = map(lambda x: list(map(lambda y: action_start + y % num_frame, x)), sample_list)
        sample_list = list(sample_map)
        return sample_list
