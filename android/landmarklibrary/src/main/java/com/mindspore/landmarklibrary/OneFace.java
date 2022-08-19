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

package com.mindspore.landmarklibrary;

import android.util.SparseArray;

/**
 * 一个人的关键点数据对象
 */
public class OneFace {
    // 性别标识
    public static final int GENDER_MAN = 0;
    public static final int GENDER_WOMAN = 1;
    // 置信度
    public float confidence;
    // 俯仰角(绕x轴旋转)
    public float pitch;
    // 偏航角(绕y轴旋转)
    public float yaw;
    // 翻滚角(绕z轴旋转)
    public float roll;
    // 年龄
    public float age;
    // 性别
    public int gender;
    // 顶点坐标
    public float[] vertexPoints;

    @Override
    protected OneFace clone() {
        OneFace copy = new OneFace();
        copy.confidence = this.confidence;
        copy.pitch = this.pitch;
        copy.yaw = this.yaw;
        copy.roll = this.roll;
        copy.age = this.age;
        copy.gender = this.gender;
        copy.vertexPoints = this.vertexPoints.clone();
        return copy;
    }

    /**
     * 复制数据
     * @param origin
     * @return
     */
    public static OneFace[] arrayCopy(OneFace[] origin) {
        if (origin == null) {
            return null;
        }
        OneFace[] copy = new OneFace[origin.length];
        for (int i = 0; i < origin.length; i++) {
            copy[i] = origin[i].clone();
        }
        return copy;
    }

    /**
     * 复制数据
     * @param origin
     * @return
     */
    public static OneFace[] arrayCopy(SparseArray<OneFace> origin) {
        if (origin == null) {
            return null;
        }
        OneFace[] copy = new OneFace[origin.size()];
        for (int i = 0; i < origin.size(); i++) {
            copy[i] = origin.get(i).clone();
        }
        return copy;
    }
}
