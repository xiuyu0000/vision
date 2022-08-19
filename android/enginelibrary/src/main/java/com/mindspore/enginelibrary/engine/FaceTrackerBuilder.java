/*
 * Copyright (c) 2022. Huawei Technologies Co., Ltd
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


import com.mindspore.enginelibrary.listener.FaceTrackerCallback;

/**
 * 人脸检测构建器
 */
public final class FaceTrackerBuilder {

    private FaceTracker mFaceTracker;
    private FaceTrackParam mFaceTrackParam;

    public FaceTrackerBuilder(FaceTracker tracker, FaceTrackerCallback callback) {
        mFaceTracker = tracker;
        mFaceTrackParam = FaceTrackParam.getInstance();
        mFaceTrackParam.trackerCallback = callback;
    }

    /**
     * 准备检测器
     */
    public void initTracker() {
        mFaceTracker.initTracker();
    }

    /**
     * 是否预览检测
     *
     * @param previewTrack
     * @return
     */
    public FaceTrackerBuilder previewTrack(boolean previewTrack) {
        mFaceTrackParam.previewTrack = previewTrack;
        return this;
    }

}
