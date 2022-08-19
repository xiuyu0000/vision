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

package com.mindspore.facelibrary.color.bean;

import java.util.ArrayList;
import java.util.List;

public class DynamicColorBaseData {

    public String name;                         // 滤镜名称
    public String vertexShader;                 // vertex shader名称
    public String fragmentShader;               // fragment shader名称
    public List<String> uniformList;            // 统一变量字段列表
    public List<UniformData> uniformDataList;   // 统一变量数据列表，目前主要用于存放滤镜的png文件
    public float strength;                      // 默认强度
    public boolean texelOffset;                 // 是否存在宽高偏移量的统一变量
    public String audioPath;                    // 滤镜音乐滤镜
    public boolean audioLooping;                // 音乐是否循环播放

    public DynamicColorBaseData() {
        uniformList = new ArrayList<>();
        uniformDataList = new ArrayList<>();
        texelOffset = false;
    }

    @Override
    public String toString() {
        return "DynamicColorData{" +
                "name='" + name + '\'' +
                ", vertexShader='" + vertexShader + '\'' +
                ", fragmentShader='" + fragmentShader + '\'' +
                ", uniformList=" + uniformList +
                ", uniformDataList=" + uniformDataList +
                ", strength=" + strength +
                ", texelOffset=" + texelOffset +
                ", audioPath='" + audioPath + '\'' +
                ", audioLooping=" + audioLooping +
                '}';
    }

    /**
     * 统一变量数据对象
     */
    public static class UniformData {

        public String uniform;  // 统一变量字段
        public String value;    // 与统一变量绑定的纹理图片

        public UniformData(String uniform, String value) {
            this.uniform = uniform;
            this.value = value;
        }
    }

}
