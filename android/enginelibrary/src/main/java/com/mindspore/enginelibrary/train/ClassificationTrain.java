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
package com.mindspore.enginelibrary.train;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * Call the MindSpore interface API in the Java layer.
 */
@SuppressLint("LongLogTag")
public class ClassificationTrain {
    private final static String TAG = "ClassificationTrain";

    public static final int TAG_COMMON = 0;
    public static final int TAG_CUSTOM = 1;

    public static final int THREAD_NUM = 2;

    // The address of the running inference environment.
    private long netEnv = 0;
    private final Context context;
    private int model;

    static {
        try {
            System.loadLibrary("mlkit-label-MS");
            Log.i(TAG, "load libiMindSpore.so successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "UnsatisfiedLinkError " + e.getMessage());
            throw new UnsatisfiedLinkError();
        }
    }

    public ClassificationTrain(Context context) {
        this.context = context;
    }

    public native long loadCustomModel(ByteBuffer modelBuffer, int numThread);

    public native String runCustomNet(long netEnv, Bitmap img, int ret_detailed_sum);

    public native boolean unloadCustomModel(long netEnv);

    public native long loadCommonModel(ByteBuffer modelBuffer, int numThread);

    public native String runCommonNet(long netEnv, Bitmap img);

    public native boolean unloadCommonModel(long netEnv);


    public boolean loadModelFromBuf(String modelPath, int model) {
        this.model = model;
        ByteBuffer buffer;
        if (TAG_COMMON == model) {
            buffer = loadModelFileFromAssets(modelPath);
            netEnv = loadCommonModel(buffer, THREAD_NUM);  //numThread's default setting is 2.
            // Loading model failed.
        } else {
            buffer = loadModelFileFromFile(modelPath);
            netEnv = loadCustomModel(buffer, THREAD_NUM);  //numThread's default setting is 2.
            // Loading model failed.
        }
        return netEnv != 0;
    }

    /**
     * Run MindSpore inference.
     */
    public String MindSpore_runnet(Bitmap img, int categoryNum) {
        String ret_str = null;
        if (TAG_COMMON == model) {
            ret_str = runCommonNet(netEnv, img);
        } else if (TAG_CUSTOM == model) {
            ret_str = runCustomNet(netEnv, img, categoryNum);
        }
        return ret_str;
    }

    /**
     * Unload model.
     */
    public void unloadCustomModel() {
        if (TAG_COMMON == model) {
            unloadCommonModel(netEnv);
        } else if (TAG_CUSTOM == model) {
            unloadCustomModel(netEnv);
        }
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
