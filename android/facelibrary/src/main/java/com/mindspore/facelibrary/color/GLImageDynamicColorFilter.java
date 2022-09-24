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

package com.mindspore.facelibrary.color;

import android.content.Context;
import android.text.TextUtils;

import com.mindspore.facelibrary.base.GLImageGroupFilter;
import com.mindspore.facelibrary.color.bean.DynamicColor;


/**
 * 颜色滤镜
 */
public class GLImageDynamicColorFilter extends GLImageGroupFilter {

    public GLImageDynamicColorFilter(Context context, DynamicColor dynamicColor) {
        super(context);
        // 判断数据是否存在
        if (dynamicColor == null || dynamicColor.filterList == null
                || TextUtils.isEmpty(dynamicColor.unzipPath)) {
            return;
        }
        // 添加滤镜
        for (int i = 0; i < dynamicColor.filterList.size(); i++) {
            mFilters.add(new DynamicColorFilter(context, dynamicColor.filterList.get(i), dynamicColor.unzipPath));
        }
    }

    /**
     * 设置滤镜强度
     * @param strength
     */
    public void setStrength(float strength) {
        for (int i = 0; i < mFilters.size(); i++) {
            if (mFilters.get(i) != null && mFilters.get(i) instanceof DynamicColorBaseFilter) {
                ((DynamicColorBaseFilter) mFilters.get(i)).setStrength(strength);
            }
        }
    }
}
