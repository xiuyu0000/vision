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

package com.mindspore.facelibrary.resource.bean;

/**
 * 资源数据
 */
public class ResourceData {

    public String name;         // 名称
    public String zipPath;      // 压缩包路径，绝对路径，"assets://" 或 "file://"开头
    public ResourceType type;   // 资源类型
    public String unzipFolder;  // 解压文件夹名称
    public String thumbPath;    // 缩略图路径

    // 处理文件绝对路径的zip包资源
    public ResourceData(String name, String zipPath, ResourceType type, String unzipFolder, String thumbPath) {
        this.name = name;
        this.zipPath = zipPath;
        this.type = type;
        this.unzipFolder = unzipFolder;
        this.thumbPath = thumbPath;
    }
}
