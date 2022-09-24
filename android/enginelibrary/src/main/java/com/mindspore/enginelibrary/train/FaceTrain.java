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

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;

import com.mindspore.enginelibrary.R;

import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * Call the MindSpore interface API in the Java layer.
 */
@SuppressLint("LongLogTag")
public class FaceTrain {
    private final static String TAG = "FaceTrain";

    // The address of the running inference environment.
    private long boundingBoxNetEnv = 0;
    private long pfldNetEnv = 0;
    private static Context mContext;


    private RenderScript rs;
    private ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic;
    private Type.Builder yuvType, rgbaType;
    private Allocation in, out;

    static {
        try {
            System.loadLibrary("mlkit-label-MS");
            Log.i(TAG, "load libiMindSpore.so successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "UnsatisfiedLinkError " + e.getMessage());
            throw new UnsatisfiedLinkError();
        }
    }

    private static class FaceTrainHolder {
        private static FaceTrain mInstance = new FaceTrain();
    }

    public static FaceTrain getInstance() {
        return FaceTrainHolder.mInstance;
    }

    public void setContext(Context mContext) {
        this.mContext = mContext;

//        rs = RenderScript.create(mContext);
//        yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
    }

    /**
     * 调用C++层的加载模型的方法
     *
     * @param modelBuffer 模型文件
     * @param numThread   线程数 默认2
     * @return
     */
    public native long loadModel(ByteBuffer modelBuffer, int numThread);

    /**
     * 调用C++层获取BoundingBox运行模型方法
     *
     * @param netEnv C++层加载模型返回的long值
     * @param img    图片的bitmap
     * @return
     */
    public native String runBoundingBoxNet(long netEnv, Bitmap img);

    /**
     * 调用C++层获取PFLD运行模型方法
     *
     * @param netEnv C++层加载模型返回的long值
     * @param img    图片的bitmap
     * @return
     */
    public native String runPFLDNet(long netEnv, Bitmap img);

    /**
     * 释放模型
     *
     * @param netEnv C++层加载模型返回的long值
     * @return
     */
    public native boolean unloadModel(long netEnv);


    /**
     * JAVA层加载BoundingBox模型文件
     *
     * @param modelPath
     * @return
     */
    private boolean loadRetinafaceModelFromBuf(String modelPath) {
        ByteBuffer buffer = loadModelFileFromAssets(modelPath);
        boundingBoxNetEnv = loadModel(buffer, 2);  //numThread's default setting is 2.
        return boundingBoxNetEnv != 0;
    }

    /**
     * JAVA层加载PFLD模型文件
     *
     * @param modelPath
     * @return
     */
    private boolean loadPFLDModelFromBuf(String modelPath) {
        ByteBuffer buffer = loadModelFileFromAssets(modelPath);
        pfldNetEnv = loadModel(buffer, 2);  //numThread's default setting is 2.
        return pfldNetEnv != 0;
    }

    /**
     * JAVA层加载整体模型
     *
     * @return 加载成功或失败
     */
    public boolean loadModel() {
        boolean loadRetinaface = loadRetinafaceModelFromBuf("retinaface_mobilenet.ms");
        boolean loadPfld = loadPFLDModelFromBuf("pfld_mobilenet0.25.ms");
        return loadRetinaface && loadPfld;
    }

    /**
     * JAVA层运行BoundingBox
     *
     * @param img bitmap
     * @return
     */
    public String runBoundingBoxMindSporeNet(Bitmap img) {
        return runBoundingBoxNet(boundingBoxNetEnv, img);
    }

    /**
     * JAVA层运行PFLD
     *
     * @param img bitmap
     * @return
     */
    public String runPFLDMindSporeNet(Bitmap img) {
        return runPFLDNet(pfldNetEnv, img);
    }

    /**
     * 获取坐标信息数据
     *
     * @param imgByte
     * @return
     */
    public PointF[] runMindSpore(byte[] imgByte, int width, int height) {
        long start = System.currentTimeMillis();
        if (yuvType == null) {
            yuvType = new Type.Builder(rs, Element.U8(rs)).setX(imgByte.length);
            in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

            rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
            out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
        }

        in.copyFrom(imgByte);

        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);


        Bitmap originBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        out.copyTo(originBitmap);

//        Matrix matrix = new Matrix();
//        matrix.postRotate(-90);
        // 创建新的图片
//        originBitmap = Bitmap.createBitmap(originBitmap, 0, 0, originBitmap.getWidth(), originBitmap.getHeight(), matrix, true);
//
////        Bitmap originBitmap = BitmapUtils.getBitmapFromByte(imgByte);
//        if (originBitmap == null) {
//            return null;
//        }
//
//
//        long end = System.currentTimeMillis();
//
//        Log.e("AAAAAAAAAAAAAAAAAAAA","bitmap花费》》》"+(end-start));
////        originBitmap.copy(Bitmap.Config.ARGB_8888, true);
//
//        String result = runBoundingBoxMindSporeNet(originBitmap);
//        if (TextUtils.isEmpty(result)) {
//            return null;
//        }
//        String[] resultArray = result.split(", ");
//        int[] boundingbox = new int[resultArray.length];
//        for (int i = 0; i < resultArray.length; i++) {
//            boundingbox[i] = Integer.parseInt(resultArray[i]);
//        }
//
//        //裁剪原始图片
//        Bitmap corpBitmap = corpBitmap(boundingbox, originBitmap);
        Bitmap corpBitmap = BitmapFactory.decodeResource(mContext.getResources(), R.drawable.test);


        // PFLD模型获取到后的人脸坐标点
        String landmarksStr = runPFLDMindSporeNet(corpBitmap);
        String[] landmarkStrs = landmarksStr.split(",");
        float[] landmarks = new float[landmarkStrs.length];
        for (int i = 0; i < landmarkStrs.length; i++) {
            landmarks[i] = Float.valueOf(landmarkStrs[i]);
        }
        for (int i = 0; i < landmarks.length; i++) {
            if (i % 2 == 0) {
                landmarks[i] = landmarks[i];
//                landmarks[i] = landmarks[i] + boundingbox[0];
            } else {
                landmarks[i] = landmarks[i];
//                landmarks[i] = landmarks[i] + boundingbox[1];
            }
        }


//        float[] landmarks = new float[]{7, 189, 24, 250, 23, 333, 32, 408, 45, 470, 84, 523, 128, 551,
//                173, 614, 225, 617, 262, 595, 310, 550, 333, 530, 355, 478, 368, 407, 387, 340,
//                381, 265, 389, 214, 78, 138, 83, 114, 130, 99, 171, 101, 215, 125, 262, 117, 301,
//                107, 334, 111, 361, 105, 380, 147, 240, 186, 236, 235, 241, 288, 245, 333, 192,
//                352, 219, 365, 230, 371, 259, 369, 278, 363, 100, 196, 130, 176, 162, 180, 193,
//                207, 161, 208, 133, 208, 285, 207, 293, 183, 322, 189, 339, 192, 317, 209, 292,
//                211, 142, 425, 181, 414, 213, 408, 248, 429, 253, 414, 280, 426, 301, 451, 280,
//                462, 241, 482, 229, 494, 209, 489, 177, 474, 153, 448, 216, 450, 227, 438, 248,
//                438, 287, 445, 259, 453, 234, 462, 214, 453};


        PointF[] point = new PointF[landmarks.length / 2];
        for (int i = 0, j = 0; i < landmarks.length; i = i + 2) {
            point[j++] = new PointF(landmarks[i], landmarks[i + 1]);
        }

        if (corpBitmap != null) {
            corpBitmap.recycle();
            corpBitmap = null;
        }
        return point;
    }

    /**
     * JAVA层释放模型
     */
    public void unloadModel() {
        unloadModel(boundingBoxNetEnv);
        unloadModel(pfldNetEnv);
    }


    private ByteBuffer loadModelFileFromAssets(String modelPath) {
        try {
            InputStream is = mContext.getAssets().open(modelPath);
            byte[] bytes = new byte[is.available()];
            is.read(bytes);
            return ByteBuffer.allocateDirect(bytes.length).put(bytes);
        } catch (Exception e) {
            Log.e(TAG, Log.getStackTraceString(e));
        }
        return null;
    }

    /**
     * 裁剪图片 bounding box 数据进行裁剪
     * (x, y, w, h)
     *
     * @return
     */
    private Bitmap corpBitmap(int[] boudingbox, Bitmap bitmap) {
        return Bitmap.createBitmap(bitmap, boudingbox[0], boudingbox[1], boudingbox[2], boudingbox[3]);
    }

}
