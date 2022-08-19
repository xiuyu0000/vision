/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MS_VISION_HEADPOSEESTIMATION_H
#define MS_VISION_HEADPOSEESTIMATION_H

#include <opencv2/opencv.hpp>

bool HeadPoseEstimation(cv::Mat img, const float *face, cv::Mat &euler_angle);

#endif //MS_VISION_HEADPOSEESTIMATION_H
