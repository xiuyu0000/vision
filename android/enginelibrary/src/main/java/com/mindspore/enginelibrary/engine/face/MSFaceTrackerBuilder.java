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
 *
 *
 */

package com.mindspore.enginelibrary.engine.face;

import com.mindspore.enginelibrary.listener.FaceTrackerCallback;

public class MSFaceTrackerBuilder {

    private MSFaceTracker msFaceTracker;
    private MSFaceTrackParam msFaceTrackParam;

    public MSFaceTrackerBuilder(MSFaceTracker tracker, FaceTrackerCallback callback) {
        msFaceTracker = tracker;
        msFaceTrackParam = MSFaceTrackParam.getInstance();
        msFaceTrackParam.trackerCallback = callback;
    }


    /**
     * 准备检测器
     */
    public void initTracker() {
        msFaceTracker.initTracker();
    }

    /**
     * 是否预览检测
     *
     * @param previewTrack
     * @return
     */
    public MSFaceTrackerBuilder previewTrack(boolean previewTrack) {
        msFaceTrackParam.previewTrack = previewTrack;
        return this;
    }
}
