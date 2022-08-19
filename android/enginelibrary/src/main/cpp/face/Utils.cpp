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
#include <jni.h>
#include <android/bitmap.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "MSNetWork.h"
#include "Utils.h"


#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)


bool BitmapToMSLiteMat(JNIEnv *env, const jobject &src_bitmap, LiteMat *lite_mat) {
    bool ret;
    AndroidBitmapInfo info;
    void *pixels = nullptr;
    LiteMat &lite_mat_bgr = *lite_mat;
    AndroidBitmap_getInfo(env, src_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        MS_PRINT("Image Err, Request RGBA.");
        return false;
    }

    AndroidBitmap_lockPixels(env, src_bitmap, &pixels);
    if (info.stride == info.width * 4) {
        ret = InitFromPixel(reinterpret_cast<const unsigned char *>(pixels),
                            LPixelType::RGBA2RGB, LDataType::UINT8,
                            info.width, info.height, lite_mat_bgr);
        if (!ret) {
            MS_PRINT("Init From RGBA error.");
        }
    } else {
        unsigned char *pixels_ptr = new unsigned char[info.width * info.height * 4];
        unsigned char *ptr = pixels_ptr;
        unsigned char *data = reinterpret_cast<unsigned char *>(pixels);
        for (int i = 0; i < info.height; i++) {
            memcpy(ptr, data, info.width * 4);
            ptr += info.width * 4;
            data += info.stride;
        }
        ret = InitFromPixel(reinterpret_cast<const unsigned char *>(pixels_ptr),
                            LPixelType::RGBA2RGB, LDataType::UINT8,
                            info.width, info.height, lite_mat_bgr);
        if (!ret) {
            MS_PRINT("Init From RGBA error.");
        }
        delete[] (pixels_ptr);
    }
    AndroidBitmap_unlockPixels(env, src_bitmap);
    return ret;
}


bool MatrixMSLiteMat(const cv::Mat &matrix, LiteMat *lite_mat) {

}


bool BitmapToMatrix(JNIEnv *env, jobject obj_bitmap, cv::Mat &matrix) {
    void *bitmapPixels;            // Save picture pixel data
    AndroidBitmapInfo bitmapInfo;   // Save picture parameters

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        // Establish temporary mat
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);
        tmp.copyTo(matrix);  // Copy to target matrix
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, matrix, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
    cv::cvtColor(matrix, matrix, cv::COLOR_RGB2BGR);
    AndroidBitmap_unlockPixels(env, obj_bitmap);
    return true;
}


bool DisplayLiteMatFirstRow(LiteMat lite_mat) {
    float *images = reinterpret_cast<float *>(lite_mat.data_ptr_);
    for (int i = 0; i < lite_mat.width_; ++i) {
        float pixel_r = images[i];
        float pixel_g = images[i + 1];
        float pixel_b = images[i + 2];
        MS_PRINT("%f, %f, %f", pixel_r, pixel_g, pixel_b);
        break;
    }
    return true;
}

void BitmapToMat2(JNIEnv *env, jobject &bitmap, cv::Mat &mat, jboolean needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = nullptr;
    cv::Mat &dst = mat;

    MS_PRINT("BitmapToMat");
    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);
    MS_PRINT("dst.create()");
    dst.create(info.height, info.width, CV_8UC4);
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        MS_PRINT("Mat tmp1()");
        cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
        if (needUnPremultiplyAlpha) {
            MS_PRINT("cvtColor1");
            cv::cvtColor(tmp, dst, cv::COLOR_RGBA2BGR);
            MS_PRINT("cvtColor1 end!");
        } else {
            MS_PRINT("tmp.copyTo()");
            tmp.copyTo(dst);
        }
    } else {
        MS_PRINT("Mat tmp2()");
        cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
        MS_PRINT("cvtColor2");
        cv::cvtColor(tmp, dst, cv::COLOR_BGR5652RGBA);
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    return;
}


void BitmapToCVMat(JNIEnv *env, jobject &bitmap, cv::Mat &mat) {
    BitmapToMat2(env, bitmap, mat, true);
}


void ConvertNHWC2NCHW(float *src, float *dst, ImgDims inputDims) {
    int size = inputDims.width * inputDims.height;

    for (int tmpSize = size; tmpSize > 0; tmpSize--) {
        for (int i = 0; i < inputDims.channel; i++) {
            dst[size * i] = src[i];
        }
        src += inputDims.channel;
        dst++;
    }
}