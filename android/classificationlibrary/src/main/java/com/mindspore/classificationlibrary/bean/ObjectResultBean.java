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

package com.mindspore.classificationlibrary.bean;

import android.content.Context;
import android.text.TextUtils;

import com.mindspore.classificationlibrary.R;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ObjectResultBean {
    private final String rectID;
    private final String imgID;
    private final String objectName;
    private final float score;
    private final float left;
    private final float top;
    private final float right;
    private final float bottom;

    private ObjectResultBean(Builder builder) {
        this.rectID = builder.rectID;
        this.imgID = builder.imgID;
        this.objectName = builder.objectName;
        this.score = builder.score;
        this.left = builder.left;
        this.top = builder.top;
        this.right = builder.right;
        this.bottom = builder.bottom;
    }

    public static class Builder {
        private String rectID;
        private String imgID;
        private String objectName;
        private float score;
        private float left;
        private float top;
        private float right;
        private float bottom;

        public ObjectResultBean build() {
            return new ObjectResultBean(this);
        }

        public Builder setRectID(String rectID) {
            this.rectID = rectID;
            return this;
        }

        public Builder setImgID(String imgID) {
            this.imgID = imgID;
            return this;
        }

        public Builder setObjectName(String objectName) {
            this.objectName = objectName;
            return this;
        }

        public Builder setScore(float score) {
            this.score = score;
            return this;
        }

        public Builder setLeft(float left) {
            this.left = left;
            return this;
        }

        public Builder setTop(float top) {
            this.top = top;
            return this;
        }

        public Builder setRight(float right) {
            this.right = right;
            return this;
        }

        public Builder setBottom(float bottom) {
            this.bottom = bottom;
            return this;
        }
    }


    public String getImgID() {
        return imgID;
    }

    public String getRectID() {
        return rectID;
    }

    public String getObjectName() {
        return objectName;
    }

    public float getScore() {
        return score;
    }

    public float getLeft() {
        return left;
    }

    public float getTop() {
        return top;
    }

    public float getRight() {
        return right;
    }

    public float getBottom() {
        return bottom;
    }


    private static final int NUM_ZERO = 0;
    private static final int NUM_ONE = 1;
    private static final int NUM_TWO = 2;
    private static final int NUM_THREE = 3;
    private static final int NUM_FOUR = 4;
    private static final int NUM_FIVE = 5;
    private static final int NUM_SIX = 6;

    public static List<ObjectResultBean> getRecognitionList(Context context, String result) {
        if (!TextUtils.isEmpty(result)) {
            String[] resultArray = result.split(";");
            if (resultArray.length <= NUM_ZERO) {
                return Collections.emptyList();
            } else {
                List<ObjectResultBean> list = new ArrayList<>();
                String[] CONTENT_ARRAY = context.getResources().getStringArray(R.array.object_detection);

                for (int i = NUM_ZERO; i < resultArray.length; i++) {
                    String singleRecognitionResult = resultArray[i];
                    String[] singleResult = singleRecognitionResult.split("_");
                    if (singleResult.length > NUM_SIX) {
                        ObjectResultBean bean = new Builder()
                                .setRectID(String.valueOf(i + NUM_ONE))
                                .setImgID(null != getData(NUM_ZERO, singleResult) ?
                                        getData(NUM_ZERO, singleResult) : "")
                                .setObjectName(null != getData(NUM_ONE, singleResult) ?
                                        CONTENT_ARRAY[Math.round(Float.parseFloat(getData(NUM_ONE, singleResult)))] : "")
                                .setScore(getFloatData(NUM_TWO, singleResult))
                                .setLeft(getFloatData(NUM_THREE, singleResult))
                                .setTop(getFloatData(NUM_FOUR, singleResult))
                                .setRight(getFloatData(NUM_FIVE, singleResult))
                                .setBottom(getFloatData(NUM_SIX, singleResult))
                                .build();
                        list.add(bean);
                    }
                }
                return list;
            }
        } else {
            return Collections.emptyList();
        }
    }


    private static float getFloatData(int index, String[] singleResult) {
        if (getData(index, singleResult) == null) {
            return NUM_ZERO;
        } else {
            return Float.parseFloat(getData(index, singleResult));
        }
    }

    private static String getData(int index, String[] singleResult) {
        if (index > singleResult.length) {
            return null;
        } else {
            if (!TextUtils.isEmpty(singleResult[index])) {
                return singleResult[index];
            }
        }
        return null;
    }

}