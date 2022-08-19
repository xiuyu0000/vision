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

package com.mindspore.facelibrary.ui.bean;

public class FaceBeautyItemBean {
    private int id;
    private int iconUncheck;
    private int iconChecked;
    private String name;

    public FaceBeautyItemBean(int id, int iconUncheck, int iconChecked, String name) {
        this.id = id;
        this.iconUncheck = iconUncheck;
        this.iconChecked = iconChecked;
        this.name = name;
    }

    public int getIconChecked() {
        return iconChecked;
    }

    public FaceBeautyItemBean setIconChecked(int iconChecked) {
        this.iconChecked = iconChecked;
        return this;
    }

    public int getId() {
        return id;
    }

    public FaceBeautyItemBean setId(int id) {
        this.id = id;
        return this;
    }

    public int getIconUncheck() {
        return iconUncheck;
    }

    public FaceBeautyItemBean setIconUncheck(int iconUncheck) {
        this.iconUncheck = iconUncheck;
        return this;
    }

    public String getName() {
        return name;
    }

    public FaceBeautyItemBean setName(String name) {
        this.name = name;
        return this;
    }


}
