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

#ifndef MS_VISION_UTILS_H
#define MS_VISION_UTILS_H

#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"

using mindspore::dataset::LiteMat;
using mindspore::dataset::LPixelType;
using mindspore::dataset::LDataType;

bool BitmapToMSLiteMat(JNIEnv *env, const jobject &src_bitmap, LiteMat *lite_mat);

bool BitmapToMatrix(JNIEnv *env, jobject obj_bitmap, cv::Mat &matrix);

void BitmapToCVMat(JNIEnv *env, jobject &bitmap, cv::Mat &mat);

void ConvertNHWC2NCHW(float *src, float *dst, ImgDims inputDims);

bool DisplayLiteMatFirstPixel(LiteMat lite_mat);

#endif //MS_VISION_UTILS_H
