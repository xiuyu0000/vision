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

package com.mindspore.classificationlibrary.bean;

public class CommonResultBean {
    private final String title;
    private final String content;
    private final float score;
    private final String position;

    public CommonResultBean(Builder builder) {
        this.title = builder.title;
        this.content = builder.content;
        this.score = builder.score;
        this.position = builder.position;
    }

    public static class Builder {
        private String title;
        private String content;
        private float score;
        private String position;

        public CommonResultBean build() {
            return new CommonResultBean(this);
        }

        public Builder setTitle(String title) {
            this.title = title;
            return this;
        }

        public Builder setContent(String content) {
            this.content = content;
            return this;
        }

        public Builder setScore(float score) {
            this.score = score;
            return this;
        }

        public Builder setPosition(String position) {
            this.position = position;
            return this;
        }
    }


    public String getTitle() {
        return title;
    }

    public String getContent() {
        return content;
    }

    public float getScore() {
        return score;
    }

    public String getPosition() {
        return position;
    }
}
