/*
 * Copyright 2022
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
#include <sstream>
#include <iterator>
#include <cstring>
#include <vector>
#include <string>
#include <set>
#include <utility>
#include <vision_lite.h>
#include "opencv2/opencv.hpp"
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"

#include "MSNetWork.h"
#include "FaceDetectionNetnative.h"
#include "HeadPoseEstimation.h"
#include "Retainface.h"
#include "Utils.h"


using mindspore::dataset::LiteMat;

#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)


bool
ImagePreProcess(const LiteMat &lite_mat_src, LiteMat *lite_mat_ptr, int resize_w, int resize_h) {
    bool ret;

    // resize image
    LiteMat lite_mat_resize;
    ret = ResizeBilinear(lite_mat_src, lite_mat_resize, resize_w, resize_h);
    if (!ret) {
        MS_PRINT("ResizeBilinear error.");
        return false;
    }
    // normalization image
    LiteMat lite_mat_norm;
    ret = ConvertTo(lite_mat_resize, lite_mat_norm, 1.0 / 255.0);
    if (!ret) {
        MS_PRINT("Normalization error.");
        return false;
    }
    // Mean Normalize image(not necessary)
    LiteMat &lite_mat = *lite_mat_ptr;
    std::vector<float> means = {0.485, 0.456, 0.406};
    std::vector<float> stds = {0.229, 0.224, 0.225};
    ret = SubStractMeanNormalize(lite_mat_norm, lite_mat, means, stds);
    if (!ret) {
        MS_PRINT("Mean Normalize error.");
        return false;
    }

    return true;
}

bool
RetainFacePreProcess(const LiteMat &lite_mat_src, LiteMat *lite_mat_ptr, int src_width,
                     int src_height) {
    bool ret;
    float long_size = max(src_width, src_height);
    float scale = 640. / long_size;
    int pad_width = int(src_width * scale);
    int pad_height = int(src_height * scale);

    // resize image
    LiteMat lite_mat_resize;
    ret = ResizeBilinear(lite_mat_src, lite_mat_resize, pad_width, pad_height);
    if (!ret) {
        MS_PRINT("ResizeBilinear error.");
        return false;
    }
    // normalization image
    LiteMat lite_mat_norm1;
    ret = ConvertTo(lite_mat_resize, lite_mat_norm1, 1.0);
    if (!ret) {
        MS_PRINT("Normalization error.");
        return false;
    }
    LiteMat lite_mat_norm2;
    std::vector<float> means = {104, 117, 123};
    std::vector<float> stds = {1, 1, 1};
    ret = SubStractMeanNormalize(lite_mat_norm1, lite_mat_norm2, means, stds);
    if (!ret) {
        MS_PRINT("Mean Normalize error.");
        return false;
    }
    // pad image.
    LiteMat &lite_mat = *lite_mat_ptr;
    if (pad_width == 640) {
        ret = Pad(lite_mat_norm2, lite_mat, 0, 640 - pad_height, 0, 0,
                  mindspore::dataset::PaddBorderType::PADD_BORDER_CONSTANT);
    } else {
        ret = Pad(lite_mat_norm2, lite_mat, 0, 0, 0, 640 - pad_width,
                  mindspore::dataset::PaddBorderType::PADD_BORDER_CONSTANT);
    }
    if (!ret) {
        MS_PRINT("Pad Image error.");
        return false;
    }
    return true;
}


char *Face2DetectionCreateLocalModelBuffer(JNIEnv *env, jobject model_buffer) {
    jbyte *model_addr = static_cast<jbyte *>(env->GetDirectBufferAddress(model_buffer));
    int model_len = static_cast<int>(env->GetDirectBufferCapacity(model_buffer));
    char *buffer(new char[model_len]);
    memcpy(buffer, model_addr, model_len);
    return buffer;
}


/**
 * @param srcImageWidth The width of the original input image.
 * @param srcImageHeight The height of the original input image.
 * @return
 */
std::string
DetectionPostProcess(cv::Mat img, float *cls_data, float *loc_data) {
    MS_PRINT("init DetectionPostProcess.");
    RetinaFace retrain_face;

    vector<FaceObject> faces;
    bool ret = retrain_face.DetectFace(img, cls_data, loc_data, faces);
    if (!ret) {
        MS_PRINT("DetectFace error.");
        return NULL;
    }
    // get the first bounding box face. (x, y, w, h)
    std::string ret_boxes = "";
    ret_boxes += std::to_string(int(faces.at(0).rect.x));
    ret_boxes += ", ";
    ret_boxes += std::to_string(int(faces.at(0).rect.y));
    ret_boxes += ", ";
    ret_boxes += std::to_string(int(faces.at(0).rect.width));
    ret_boxes += ", ";
    ret_boxes += std::to_string(int(faces.at(0).rect.height));
    return ret_boxes;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_mindspore_enginelibrary_train_FaceTrain_loadModel(JNIEnv *env, jobject thiz,
                                                            jobject buffer,
                                                            jint numThread) {
    if (nullptr == buffer) {
        MS_PRINT("error, buffer is nullptr!");
        return (jlong) nullptr;
    }
    jlong bufferLen = env->GetDirectBufferCapacity(buffer);
    MS_PRINT("MindSpore get bufferLen:%d", static_cast<int>(bufferLen));
    if (0 == bufferLen) {
        MS_PRINT("error, bufferLen is 0!");
        return (jlong) nullptr;
    }

    char *model_buffer = Face2DetectionCreateLocalModelBuffer(env, buffer);
    if (model_buffer == nullptr) {
        MS_PRINT("model_buffer create failed!");
        return (jlong) nullptr;
    }

    MS_PRINT("MindSpore loading Model.");
    void **labelEnv = new void *;
    MSNetWork *ms_net = new MSNetWork;
    *labelEnv = ms_net;

    mindspore::lite::Context *context = new mindspore::lite::Context;
    context->thread_num_ = numThread;
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = mindspore::lite::NO_BIND;
    context->device_list_[0].device_info_.cpu_device_info_.enable_float16_ = false;
    context->device_list_[0].device_type_ = mindspore::lite::DT_CPU;

    ms_net->CreateSessionMS(model_buffer, bufferLen, context);
    delete context;
    if (ms_net->session() == nullptr) {
        delete ms_net;
        delete labelEnv;
        MS_PRINT("MindSpore create session failed!.");
        return (jlong) nullptr;
    }
    MS_PRINT("MindSpore create session successfully.");
    env->DeleteLocalRef(buffer);
    MS_PRINT("Ptr released successfully.");
    return (jlong) labelEnv;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_mindspore_enginelibrary_train_FaceTrain_runBoundingBoxNet(JNIEnv *env, jobject thiz,
                                                                    jlong netEnv,
                                                                    jobject src_bitmap) {
    LiteMat lite_mat_src, lite_mat;
    if (!BitmapToMSLiteMat(env, src_bitmap, &lite_mat_src)) {
        MS_PRINT("Bitmap to MindSpore LiteMat error.");
        return NULL;
    }
    // android bitmap to jni will resize the image with scale 0.91.
    float scale = 1; // 0.91;
    int src_width = int(lite_mat_src.width_ * scale);
    int src_height = int(lite_mat_src.height_ * scale);
    cv::Mat src_img(cv::Size(src_width, src_height), CV_8UC3);
    MS_PRINT("src_width %d, src_height %d", src_width, src_height);


    if (!RetainFacePreProcess(lite_mat_src, &lite_mat, src_width, src_height)) {
        MS_PRINT("Pre process input image error.");
        return NULL;
    }
    // Get the mindspore inference environment which created in loadModel().
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    if (labelEnv == nullptr) {
        MS_PRINT("MindSpore Lite error, labelEnv is a nullptr.");
        return NULL;
    }
    // Get session.
    MSNetWork *ms_net = static_cast<MSNetWork *>(*labelEnv);
    auto ms_session = ms_net->session();
    if (ms_session == nullptr) {
        MS_PRINT("MindSpore Lite error, Session is a nullptr.");
        return NULL;
    }
    MS_PRINT("MindSpore Lite get session.");
    auto ms_inputs = ms_session->GetInputs();
    auto ms_intensor = ms_inputs.front();
    float *dataHWC = reinterpret_cast<float *>(lite_mat.data_ptr_);

    ImgDims input_dims;
    input_dims.channel = lite_mat.channel_;
    input_dims.width = lite_mat.width_;
    input_dims.height = lite_mat.height_;
    // copy input Tensor
    memcpy(ms_intensor->MutableData(), dataHWC,
           input_dims.channel * input_dims.width * input_dims.height * sizeof(float));
    MS_PRINT("MindSpore Lite get inputs.");

    auto status = ms_session->RunGraph();
    if (status != mindspore::lite::RET_OK) {
        MS_PRINT("MindSpore RunGraph error.");
        return NULL;
    }

    // begin get the output tensor.
    float *cls_data;
    float *boxes_data;
    std::string cls_name = "Default/Softmax-op506";
    std::string boxes_name = "Default/Concat-op487";
    auto names = ms_session->GetOutputTensorNames();
    for (const auto &name: names) {
        auto temp_dat = ms_session->GetOutputByTensorName(name);
        if (name == cls_name) {
            cls_data = reinterpret_cast<float *>(temp_dat->MutableData());
        }
        if (name == boxes_name) {
            boxes_data = reinterpret_cast<float *>(temp_dat->MutableData());
        }
    }
    if (!cls_data) {
        MS_PRINT("Cannot find output tensor by cls.");
        return NULL;
    }
    if (!boxes_data) {
        MS_PRINT("Cannot find output tensor by boxes.");
        return NULL;
    }

    std::string ret_srt = DetectionPostProcess(src_img, cls_data, boxes_data);
    const char *result_boxes = ret_srt.c_str();
    return (env)->NewStringUTF(result_boxes);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_mindspore_enginelibrary_train_FaceTrain_runPFLDNet(JNIEnv *env, jobject thiz,
                                                             jlong netEnv, jobject src_bitmap) {
    LiteMat lite_mat_src, lite_mat;

    if (!BitmapToMSLiteMat(env, src_bitmap, &lite_mat_src)) {
        MS_PRINT("Bitmap to MindSpore LiteMat error.");
        return NULL;
    }
    if (!ImagePreProcess(lite_mat_src, &lite_mat, 112, 112)) {
        MS_PRINT("Pre process input image error.");
        return NULL;
    }
    // Get the mindspore inference environment which created in loadModel().
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    if (labelEnv == nullptr) {
        MS_PRINT("MindSpore Lite error, labelEnv is a nullptr.");
        return NULL;
    }
    // Get session.
    MSNetWork *ms_net = static_cast<MSNetWork *>(*labelEnv);
    auto ms_session = ms_net->session();
    if (ms_session == nullptr) {
        MS_PRINT("MindSpore Lite error, Session is a nullptr.");
        return NULL;
    }

    MS_PRINT("MindSpore Lite get session.");
    auto ms_inputs = ms_session->GetInputs();
    auto ms_intensor = ms_inputs.front();
    float *dataHWC = reinterpret_cast<float *>(lite_mat.data_ptr_);

    // copy input Tensor
    ImgDims input_dims;
    input_dims.channel = lite_mat.channel_;
    input_dims.width = lite_mat.width_;
    input_dims.height = lite_mat.height_;
    memcpy(ms_intensor->MutableData(), dataHWC,
           input_dims.channel * input_dims.width * input_dims.height * sizeof(float));
    MS_PRINT("MindSpore Lite get ms_inputs.");

    // run lite model
    auto status = ms_session->RunGraph();
    if (status != mindspore::lite::RET_OK) {
        MS_PRINT("MindSpore RunGraph error.");
        return NULL;
    }

    // begin get the output tensor.
    float *landmarks;
    std::string result_name = "Default/head-LandmarkHead/fc-Dense/BiasAdd-op387";
    auto names = ms_session->GetOutputTensorNames();
    for (const auto &name: names) {
        if (name == result_name) {
            auto temp_dat = ms_session->GetOutputByTensorName(name);
            landmarks = reinterpret_cast<float *>(temp_dat->MutableData());
        }
    }
    if (!landmarks) {
        MS_PRINT("Cannot find output tensor by name.");
        return NULL;
    }

    // calculate the euler angle for Head Pose Estimation
    int src_image_width = lite_mat_src.width_;
    int src_image_height = lite_mat_src.height_;
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);
    cv::Mat image(cv::Size(src_image_height, src_image_width), CV_8UC3);
    if (!HeadPoseEstimation(image, landmarks, euler_angle)) {
        MS_PRINT("Calculate euler angle error.");
        return NULL;
    }

    // transfer landmarks to JNI.
    std::string landmarks_str = "";
    for (int i = 0; i < 3; ++i) {
        MS_PRINT("euler_angle[%d]: %f", i, euler_angle.at<double>(i));
    }
    for (int i = 0; i < 136; ++i) {
        int pos;
        if (i % 2 == 0) {
            pos = int(round(landmarks[i] * src_image_width));
        } else {
            pos = int(round(landmarks[i] * src_image_height));
        }
        std::string pos_str = std::to_string(pos);
        landmarks_str += pos_str;
        landmarks_str += ", ";
    }
    MS_PRINT("MindSpore Lite PFLD landmarks str:%s", landmarks_str.c_str());
    landmarks_str = landmarks_str.substr(0, landmarks_str.length() - 2);
    const char *result_landmarks = landmarks_str.c_str();
    return (env)->NewStringUTF(result_landmarks);
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_mindspore_enginelibrary_train_FaceTrain_unloadModel(JNIEnv *env, jobject thiz,
                                                              jlong netEnv) {
    void **labelEnv = reinterpret_cast<void **>(netEnv);
    MSNetWork *ms_net = static_cast<MSNetWork *>(*labelEnv);
    ms_net->ReleaseNets();
    return (jboolean) true;
}



//extern "C"
//JNIEXPORT jstring JNICALL
//Java_com_mindspore_enginelibrary_train_FaceTrain_runTestModel(JNIEnv *env, jobject thiz,
//                                                               jlong net_retina_env, jlong pfld_env,
//                                                               jbyteArray data, jint size) {
//    jbyte *content_array = (env)->GetByteArrayElements(data, NULL);
//    //*env->GetByteArrayRegion(array,0,array_length,content_array); //tried this as well, same results
//    std::string str = "好的";
//    for (int i = 0; i < 10; i++) {
//        MS_PRINT("%d", content_array[i]);
//        std::string pos_str = std::to_string(content_array[i]);
//        str += pos_str;
//        str += ", ";
//    }
//    const char *result = str.c_str();
//    return (env)->NewStringUTF(result);
//}

