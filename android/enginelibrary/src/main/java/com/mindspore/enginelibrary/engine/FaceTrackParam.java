/*
 * Copyright (c) 2022.
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

package com.mindspore.enginelibrary.engine;

import com.megvii.facepp.sdk.Facepp;
import com.mindspore.enginelibrary.listener.FaceTrackerCallback;

/**
 * 人脸检测参数
 */
public final class FaceTrackParam {

    // 是否允许检测
    boolean canFaceTrack = false;
    // 旋转角度
    public int rotateAngle;
    // 是否相机预览检测，true为预览检测，false为静态图片检测
    public boolean previewTrack;
    // 是否允许3D姿态角
    public boolean enable3DPose;
    // 是否允许区域检测
    public boolean enableROIDetect;
    // 检测区域缩放比例
    public float roiRatio;
    // 是否允许106个关键点
    public boolean enable106Points;
    // 是否后置摄像头
    public boolean isBackCamera;
    // 是否允许人脸年龄检测
    public boolean enableFaceProperty;
    // 是否允许多人脸检测
    public boolean enableMultiFace;
    // 最小人脸大小
    public int minFaceSize;
    // 检测间隔
    public int detectInterval;
    // 检测模式
    public int trackMode;
    // 检测回调
    public FaceTrackerCallback trackerCallback;

    private static class FaceParamHolder {
        public static FaceTrackParam instance = new FaceTrackParam();
    }

    private FaceTrackParam() {
        reset();
    }

    public static FaceTrackParam getInstance() {
        return FaceParamHolder.instance;
    }

    /**
     * 重置为初始状态
     */
    public void reset() {
        previewTrack = true;
        enable3DPose = false;
        enableROIDetect = false;
        roiRatio = 0.8f;
        enable106Points = true;
        isBackCamera = false;
        enableFaceProperty = false;
        enableMultiFace = true;
        minFaceSize = 200;
        detectInterval = 25;
        trackMode = Facepp.FaceppConfig.DETECTION_MODE_TRACKING;
        trackerCallback = null;
    }

}
