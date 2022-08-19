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

package com.mindspore.facelibrary.presenter;

import android.graphics.SurfaceTexture;
import android.opengl.EGLContext;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.facelibrary.color.bean.DynamicColor;
import com.mindspore.facelibrary.resource.bean.ResourceData;

public abstract class PreviewPresenter<T extends AppCompatActivity> extends IPresenter<T> {
    public PreviewPresenter(T target) {
        super(target);
    }

    public abstract void onBindSharedContext(EGLContext context);


    /**
     * SurfaceTexture 创建
     * @param surfaceTexture
     */
    public abstract void onSurfaceCreated(SurfaceTexture surfaceTexture);

    /**
     * SurfaceTexture 发生变化
     * @param width
     * @param height
     */
    public abstract void onSurfaceChanged(int width, int height);

    /**
     * SurfaceTexture 销毁
     */
    public abstract void onSurfaceDestroyed();

    /**
     * 切换道具资源
     * @param resourceData
     */
    public abstract void changeResource(@NonNull ResourceData resourceData);

    /**
     * 切换滤镜
     * @param color
     */
    public abstract void changeDynamicFilter(DynamicColor color);
//
//    /**
//     * 切换彩妆
//     * @param makeup
//     */
//    public abstract void changeDynamicMakeup(DynamicMakeup makeup);

    /**
     * 切换滤镜
     * @param filterIndex
     */
    public abstract void changeDynamicFilter(int filterIndex);

    /**
     * 前一个滤镜
     */
    public abstract int previewFilter();

    /**
     * 下一个滤镜
     */
    public abstract int nextFilter();

    /**
     * 获取当前的滤镜索引
     * @return
     */
    public abstract int getFilterIndex();

    /**
     * 是否允许比较
     * @param enable
     */
    public abstract void showCompare(boolean enable);

    /**
     * 拍照
     */
    public abstract void takePicture();

    /**
     * 切换相机
     */
    public abstract void switchCamera();

    /**
     * 开始录制
     */
    public abstract void startRecord();

    /**
     * 停止录制
     */
    public abstract void stopRecord();

    /**
     * 取消录制
     */
    public abstract void cancelRecord();

    /**
     * 是否正处于录制过程
     * @return true：正在录制，false：非录制状态
     */
    public abstract boolean isRecording();

    /**
     * 是否打开闪光灯
     * @param on    打开闪光灯
     */
    public abstract void changeFlashLight(boolean on);

    /**
     * 是否允许边框模糊
     * @param enable true:允许边框模糊
     */
    public abstract void enableEdgeBlurFilter(boolean enable);

    /**
     * 选择音乐
     * @param path
     */
    public abstract void setMusicPath(String path);

    /**
     * 打开相机设置
     */
    public abstract void onOpenCameraSettingPage();
}
