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

package com.mindspore.enginelibrary.train;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class DetectionTrain {

    private final static String TAG = "DetectionTrain";

    public static final int TAG_COMMON = 0;
    public static final int TAG_CUSTOM = 1;

    public static final int THREAD_NUM = 2;

    private long netEnv = 0;
    private final Context context;

    static {
        try {
            System.loadLibrary("mlkit-label-MS");
            Log.i(TAG, "load libiMindSpore.so successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "UnsatisfiedLinkError " + e.getMessage());
        }
    }

    public DetectionTrain(Context context) {
        this.context = context;
    }

    public native long loadModel(ByteBuffer buffer, int numThread);

    public native String runNet(long netEnv, Bitmap img);

    public native boolean unloadModel(long netEnv);

    public boolean loadModelFromBuf(String modelPath, int model) {
        ByteBuffer buffer;//numThread's default setting is 2.
        if (TAG_COMMON == model) {
            buffer = loadModelFileFromAssets(modelPath);
        } else {
            buffer = loadModelFileFromFile(modelPath);
        }
        netEnv = loadModel(buffer, THREAD_NUM);  //numThread's default setting is 2.
        return netEnv != 0;
    }

    public String MindSpore_runnet(Bitmap img) {
        return runNet(netEnv, img);
    }

    /**
     * Unbound model
     *
     * @return true
     */
    public void unloadModel() {
        unloadModel(netEnv);
    }

    /**
     * Load model file stream.
     *
     * @param modelPath Model file path.
     * @return Model ByteBuffer.
     */
    public ByteBuffer loadModelFileFromFile(String modelPath) {
        try {
            InputStream is = new FileInputStream(modelPath);
            byte[] bytes = new byte[is.available()];
            is.read(bytes);
            return ByteBuffer.allocateDirect(bytes.length).put(bytes);
        } catch (Exception e) {
            Log.e(TAG, Log.getStackTraceString(e));
        }
        return null;
    }

    public ByteBuffer loadModelFileFromAssets(String modelPath) {
        try {
            InputStream is = context.getAssets().open(modelPath);
            byte[] bytes = new byte[is.available()];
            is.read(bytes);
            return ByteBuffer.allocateDirect(bytes.length).put(bytes);
        } catch (Exception e) {
            Log.e(TAG, Log.getStackTraceString(e));
        }
        return null;
    }

}