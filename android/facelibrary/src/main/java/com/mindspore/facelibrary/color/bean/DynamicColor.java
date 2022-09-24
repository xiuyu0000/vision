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

package com.mindspore.facelibrary.color.bean;

import java.util.ArrayList;
import java.util.List;

/**
 * 滤镜数据
 */
public class DynamicColor {

    // 滤镜解压的文件夹路径
    public String unzipPath;

    // 滤镜列表
    public List<DynamicColorData> filterList;

    public DynamicColor() {
        filterList = new ArrayList<>();
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("unzipPath: ");
        builder.append(unzipPath);
        builder.append("\n");

        builder.append("data: [");
        for (int i = 0; i < filterList.size(); i++) {
            builder.append(filterList.get(i).toString());
            if (i < filterList.size() - 1) {
                builder.append(",");
            }
        }
        builder.append("]");

        return builder.toString();
    }

}
