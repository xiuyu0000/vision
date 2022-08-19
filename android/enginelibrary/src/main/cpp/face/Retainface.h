/*
 * Copyright 2022 Huawei Technologies Co., Ltd
 * Sort of Code from https://github.com/OpenFirework/Retinaface_CPP.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MS_VISION_RETAINFACE_H
#define MS_VISION_RETAINFACE_H

#include "opencv2/opencv.hpp"

using namespace std;

struct FaceObject {
    cv::Rect_<float> rect;
    float prob;
};

struct Anchor {
    float cx;
    float cy;
    float s_kx;
    float s_ky;
};

class RetinaFace {
public:
    RetinaFace();

    ~RetinaFace();

    cv::Mat PreProcess(const cv::Mat img);

    bool DetectFace(cv::Mat img, float *cls_data, float *loc_data, vector<FaceObject> &faces);

private:
    vector<Anchor> Anchors_;
    int width_;
    int height_;
    int long_size_;
    float mean_vals_[3];
    float conf_;
    int top_k_;
};

#endif //MS_VISION_RETAINFACE_H
