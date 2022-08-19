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

package com.mindspore.facelibrary.stickers.bean;

/**
 * 某个部位的动态贴纸数据
 */
public class DynamicStickerData {
    public int width;               // 贴纸宽度
    public int height;              // 贴纸高度
    public int frames;              // 贴纸帧数
    public int action;              // 动作，0表示默认显示，这里用来处理贴纸音乐、动作等
    public String stickerName;      // 贴纸名称，用于标记贴纸所在文件夹以及png文件的
    public int duration;            // 贴纸帧显示间隔
    public boolean stickerLooping;  // 贴纸是否循环渲染
    public String audioPath;        // 音乐路径，不存在时，路径为空字符串
    public boolean audioLooping;    // 音乐是否循环播放
    public int maxCount;            // 最大贴纸渲染次数

    @Override
    public String toString() {
        return "DynamicStickerData{" +
                "width=" + width +
                ", height=" + height +
                ", frames=" + frames +
                ", action=" + action +
                ", stickerName='" + stickerName + '\'' +
                ", duration=" + duration +
                ", stickerLooping=" + stickerLooping +
                ", audioPath='" + audioPath + '\'' +
                ", audioLooping=" + audioLooping +
                ", maxCount=" + maxCount +
                '}';
    }
}
