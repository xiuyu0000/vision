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

package com.mindspore.facelibrary.color;

import android.content.Context;

import com.mindspore.facelibrary.color.bean.DynamicColorData;


/**
 * 颜色滤镜
 */
public class DynamicColorFilter extends DynamicColorBaseFilter {

    public DynamicColorFilter(Context context, DynamicColorData dynamicColorData, String unzipPath) {
        super(context, dynamicColorData, unzipPath);
    }
}
