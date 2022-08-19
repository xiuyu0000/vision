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

package com.mindspore.facelibrary.opengl.face;

public class Face {

    public float[] landmarks;

    public int width;        // 保存人脸的框 的宽度
    public int height;       // 保存人脸的框 的高度
    public int imgWidth;    // 送去检测的所有宽 屏幕
    public int imgHeight;   // 送去检测的所有高 屏幕

    public Face(float[] landmarks, int width, int height, int imgWidth, int imgHeight) {
        this.landmarks = landmarks;
        this.width = width;
        this.height = height;
        this.imgWidth = imgWidth;
        this.imgHeight = imgHeight;
    }
}
