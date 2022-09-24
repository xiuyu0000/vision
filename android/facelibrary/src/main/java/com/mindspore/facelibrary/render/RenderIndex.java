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

package com.mindspore.facelibrary.render;

/**
 * 渲染类型索引
 */
public final class RenderIndex {
    public static final int CameraIndex = 0;        // 相机输入索引
    public static final int BeautyIndex = 1;        // 美颜索引
    public static final int MakeupIndex = 2;        // 彩妆索引
    public static final int FaceAdjustIndex = 3;    // 美型索引
    public static final int FilterIndex = 4;        // 滤镜索引
    public static final int ResourceIndex = 5;      // 资源索引
    public static final int DepthBlurIndex = 6;     // 景深索引
    public static final int VignetteIndex = 7;      // 暗角索引
    public static final int DisplayIndex = 8;       // 显示索引
    public static final int FacePointIndex = 9;     // 人脸关键点索引
    public static final int NumberIndex = 10;       // 索引个数
}
