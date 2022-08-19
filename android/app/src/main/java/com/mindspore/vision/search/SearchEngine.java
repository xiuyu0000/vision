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

package com.mindspore.vision.search;


import android.os.Environment;

import java.io.File;
import java.util.List;


public class SearchEngine {
    private final File path = Environment.getExternalStorageDirectory();

    private final static String KEYWORD = "custom.json";
    private boolean isSearching;
    private volatile boolean stop;
    private SearchEngineCallback callback;
    private CallbackExecutor callbackExecutor;

    public SearchEngine() {
    }

    public void start(final SearchEngineCallback callback) {
        isSearching = true;
        stop = false;
        callbackExecutor = new CallbackExecutor(callback, 200);
        new Thread(() -> {
            findFileRecursively(path);
            callbackExecutor.onFinish();
            isSearching = false;
        }).start();
    }

    public void start() {
        if (callback != null) {
            start(callback);
        }
    }

    public void stop() {
        stop = isSearching;
    }

    public boolean isSearching() {
        return isSearching;
    }

    public void setCallback(SearchEngineCallback callback) {
        this.callback = callback;
    }

    private void findFileRecursively(final File file) {
        if (stop || file.getName().startsWith(".")) {
            return;
        }

        if (file.isDirectory()) {
            File[] files = file.listFiles();
            if (files != null) {
                callbackExecutor.onSearchDirectory(file);
                for (File f : files) {
                    findFileRecursively(f);
                }
            }
        } else {
            if (keywordFilter(file)) {
                FileItem item = new FileItem(file);
                callbackExecutor.onFind(item);
            }
        }
    }

    private boolean keywordFilter(File file) {
        String fileName = file.getName();
        return fileName.toUpperCase().contains(KEYWORD.toUpperCase());
    }

    public interface SearchEngineCallback {
        void onFind(List<FileItem> fileItems);

        void onSearchDirectory(File file);

        void onFinish();
    }

}
