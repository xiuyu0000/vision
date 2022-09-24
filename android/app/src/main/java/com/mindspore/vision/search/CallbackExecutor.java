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

package com.mindspore.vision.search;

import android.os.Environment;
import android.os.Handler;
import android.os.Looper;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

class CallbackExecutor {
    private final SearchEngine.SearchEngineCallback callback;
    private final long interval;

    private volatile Handler handler;

    private final List<FileItem> cachedItems = new ArrayList<>();
    private File currentDirectory = Environment.getExternalStorageDirectory();
    private volatile boolean isFinished;

    CallbackExecutor(SearchEngine.SearchEngineCallback callback, long interval) {
        this.callback = callback;
        this.interval = interval;
    }

    void onFind(FileItem item) {
        if (handler == null) {
            handler = new Handler(Looper.getMainLooper());
            handler.post(new Timer());
        }
        synchronized (callback) {
            cachedItems.add(item);
        }
    }


    void onSearchDirectory(File file) {
        if (handler == null) {
            handler = new Handler(Looper.getMainLooper());
            handler.post(new Timer());
        }
        currentDirectory = file;
    }

    void onFinish() {
        isFinished = true;
    }

    class Timer implements Runnable {
        @Override
        public void run() {
            synchronized (callback) {
                callback.onFind(cachedItems);
                cachedItems.clear();
                callback.onSearchDirectory(currentDirectory);
                if (isFinished) {
                    callback.onFinish();
                    return;
                }
                handler.postDelayed(this, interval);
            }
        }
    }
}
