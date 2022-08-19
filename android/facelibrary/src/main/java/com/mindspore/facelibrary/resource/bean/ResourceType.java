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
 * 资源枚举类型
 */
public enum ResourceType {

    NONE("none", -1),       // 没有资源
    STICKER("sticker", 0),  // 贴纸资源类型
    FILTER("filter", 1),    // 滤镜资源类型
    EFFECT("effect", 2),    // 特效资源类型
    MAKEUP("makeup", 3),    // 彩妆资源类型
    MULTI("multi", 4);      // 多种类型混合起来

    private String name;
    private int index;

    ResourceType(String name, int index) {
        this.name = name;
        this.index = index;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }
}
