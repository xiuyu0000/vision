/*
 * Copyright (c) 2022.  Huawei Technologies Co., Ltd
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

package com.mindspore.enginelibrary.engine.face;

import android.content.Context;
import android.graphics.PointF;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.mindspore.enginelibrary.listener.FaceTrackerCallback;
import com.mindspore.enginelibrary.train.FaceTrain;
import com.mindspore.enginelibrary.utils.ConUtil;
import com.mindspore.enginelibrary.utils.SensorEventUtil;
import com.mindspore.landmarklibrary.LandmarkEngine;
import com.mindspore.landmarklibrary.OneFace;

/**
 * MindSpore人脸检测器
 */
public class MSFaceTracker {

    private static final String TAG = "MSFaceTracker";

    private final Object mSyncFence = new Object();

    // 人脸检测参数
    private MSFaceTrackParam mFaceTrackParam;

    // 检测线程
    private FaceTrackerThread mTrackerThread;


    private static class FaceTrackerHolder {
        private static MSFaceTracker mInstance = new MSFaceTracker();
    }

    private MSFaceTracker() {
        mFaceTrackParam = MSFaceTrackParam.getInstance();
    }

    public static MSFaceTracker getInstance() {
        return FaceTrackerHolder.mInstance;
    }

    public MSFaceTrackerBuilder setFaceCallback(FaceTrackerCallback callback) {
        return new MSFaceTrackerBuilder(this, callback);
    }

    /**
     * 准备检测器
     */
    void initTracker() {
        synchronized (mSyncFence) {
            mTrackerThread = new FaceTrackerThread("init");
            mTrackerThread.start();
            mTrackerThread.waitUntilReady();
        }
    }

    /**
     * 初始化人脸检测
     *
     * @param context     上下文
     * @param orientation 图像角度
     * @param width       图像宽度
     * @param height      图像高度
     */
    public void prepareFaceTracker(Context context, int orientation, int width, int height) {
        synchronized (mSyncFence) {
            if (mTrackerThread != null) {
                mTrackerThread.prepareFaceTracker(context, orientation, width, height);
            }
        }
    }

    /**
     * 检测人脸
     *
     * @param data
     * @param width
     * @param height
     */
    public void trackFace(byte[] data, int width, int height) {
        synchronized (mSyncFence) {
            if (mTrackerThread != null) {
                mTrackerThread.trackFace(data, width, height);
            }
        }
    }

    /**
     * 销毁检测器
     */
    public void destroyTracker() {
        synchronized (mSyncFence) {
            mTrackerThread.quitSafely();
        }
    }

    /**
     * 是否后置摄像头
     *
     * @param backCamera
     * @return
     */
    public MSFaceTracker setBackCamera(boolean backCamera) {
        mFaceTrackParam.isBackCamera = backCamera;
        return this;
    }

    /**
     * 是否允许3D姿态角
     *
     * @param enable
     * @return
     */
    public MSFaceTracker enable3DPose(boolean enable) {
        mFaceTrackParam.enable3DPose = enable;
        return this;
    }

    /**
     * 是否允许区域检测
     *
     * @param enable
     * @return
     */
    public MSFaceTracker enableROIDetect(boolean enable) {
        mFaceTrackParam.enableROIDetect = enable;
        return this;
    }

    /**
     * 是否允许106个关键点
     *
     * @param enable
     * @return
     */
    public MSFaceTracker enable68Points(boolean enable) {
        mFaceTrackParam.enable68Points = enable;
        return this;
    }

    /**
     * 是否允许多人脸检测
     *
     * @param enable
     * @return
     */
    public MSFaceTracker enableMultiFace(boolean enable) {
        mFaceTrackParam.enableMultiFace = enable;
        return this;
    }

    /**
     * 是否允许人脸年龄检测
     *
     * @param enable
     * @return
     */
    public MSFaceTracker enableFaceProperty(boolean enable) {
        mFaceTrackParam.enableFaceProperty = enable;
        return this;
    }

    /**
     * 最小检测人脸大小
     *
     * @param size
     * @return
     */
    public MSFaceTracker minFaceSize(int size) {
        mFaceTrackParam.minFaceSize = size;
        return this;
    }

    /**
     * 检测时间间隔
     *
     * @param interval
     * @return
     */
    public MSFaceTracker detectInterval(int interval) {
        mFaceTrackParam.detectInterval = interval;
        return this;
    }

    /**
     * 检测模式
     *
     * @param mode
     * @return
     */
    public MSFaceTracker trackMode(int mode) {
        mFaceTrackParam.trackMode = mode;
        return this;
    }

    /**
     * 检测线程
     */
    private static class FaceTrackerThread extends Thread {

        private final Object mStartLock = new Object();
        private boolean mReady = false;

        private Looper mLooper;
        private Handler mHandler;

        private SensorEventUtil mSensorUtil;


        public FaceTrackerThread(String name) {
            super(name);
        }

        @Override
        public void run() {
            Looper.prepare();
            synchronized (this) {
                mLooper = Looper.myLooper();
                notifyAll();
                mHandler = new Handler(mLooper);
            }
            synchronized (mStartLock) {
                mReady = true;
                mStartLock.notify();
            }
            Looper.loop();
            synchronized (this) {
                release();
                mHandler.removeCallbacksAndMessages(null);
                mHandler = null;
            }
            synchronized (mStartLock) {
                mReady = false;
            }
        }

        /**
         * 等待线程准备完成
         */
        public void waitUntilReady() {
            synchronized (mStartLock) {
                while (!mReady) {
                    try {
                        mStartLock.wait();
                    } catch (InterruptedException e) {

                    }
                }
            }
        }

        /**
         * 安全退出
         *
         * @return
         */
        public boolean quitSafely() {
            Looper looper = getLooper();
            if (looper != null) {
                looper.quitSafely();
                return true;
            }
            return false;
        }

        /**
         * 获取Looper
         *
         * @return
         */
        public Looper getLooper() {
            if (!isAlive()) {
                return null;
            }
            synchronized (this) {
                while (isAlive() && mLooper == null) {
                    try {
                        wait();
                    } catch (InterruptedException e) {
                    }
                }
            }
            return mLooper;
        }

        /**
         * 初始化人脸检测
         *
         * @param context     上下文
         * @param orientation 图像角度
         * @param width       图像宽度
         * @param height      图像高度
         */
        public void prepareFaceTracker(final Context context, final int orientation,
                                       final int width, final int height) {
            waitUntilReady();
            mHandler.post(new Runnable() {
                @Override
                public void run() {
                    internalPrepareFaceTracker(context);
                }
            });
        }

        /**
         * 检测人脸
         *
         * @param data   图像数据， NV21 或者 RGBA格式
         * @param width  图像宽度
         * @param height 图像高度
         * @return 是否检测成功
         */
        public void trackFace(final byte[] data, final int width, final int height) {
            waitUntilReady();
            mHandler.post(new Runnable() {
                @Override
                public void run() {
                    internalTrackFace(data, width, height);
                }
            });
        }


        /**
         * 释放资源
         */
        private void release() {
            ConUtil.releaseWakeLock();
            FaceTrain.getInstance().unloadModel();
        }


        /**
         * 初始化人脸检测
         *
         * @param context 上下文
         */
        private synchronized void internalPrepareFaceTracker(Context context) {
            if (mSensorUtil == null) {
                mSensorUtil = new SensorEventUtil(context);
            }
            FaceTrain.getInstance().setContext(context);
            boolean load = FaceTrain.getInstance().loadModel();
            MSFaceTrackParam.getInstance().canFaceTrack = load;
            if (!load){
                return;
            }
        }

        /**
         * 检测人脸
         *
         * @param data   图像数据，预览时为NV21，静态图片则为RGBA格式
         * @param width  图像宽度
         * @param height 图像高度
         * @return 是否检测成功
         */
        private synchronized void internalTrackFace(byte[] data, int width, int height) {
            MSFaceTrackParam faceTrackParam = MSFaceTrackParam.getInstance();
            LandmarkEngine.getInstance().setFaceSize(0);
            if (!faceTrackParam.canFaceTrack ) {
                LandmarkEngine.getInstance().setFaceSize(0);
                if (faceTrackParam.trackerCallback != null) {
                    faceTrackParam.trackerCallback.onTrackingFinish();
                }
                return;
            }


            int orientation = faceTrackParam.previewTrack ? mSensorUtil.orientation : 0;
            int rotation = 0;
            if (orientation == 0) {         // 0
                rotation = faceTrackParam.rotateAngle;
            } else if (orientation == 1) {  // 90
                rotation = 0;
            } else if (orientation == 2) {  // 270
                rotation = 180;
            } else if (orientation == 3) {  // 180
                rotation = 360 - faceTrackParam.rotateAngle;
            }


            PointF[] pointFS = FaceTrain.getInstance().runMindSpore(data, width, height);
            if (pointFS == null) {
                return;
            }


            OneFace oneFace = LandmarkEngine.getInstance().getOneFace(0);
            oneFace.gender = -1;
            oneFace.age = -1;

//            oneFace.pitch = face.pitch;
//            if (faceTrackParam.isBackCamera) {
//                oneFace.yaw = -face.yaw;
//            } else {
//                oneFace.yaw = face.yaw;
//            }
//            oneFace.roll = face.roll;
//            if (faceTrackParam.previewTrack) {
//                if (faceTrackParam.isBackCamera) {
//                    oneFace.roll = (float) (Math.PI / 2.0f + oneFace.roll);
//                } else {
//                    oneFace.roll = (float) (Math.PI / 2.0f - face.roll);
//                }
//            }

            if (oneFace.vertexPoints == null || oneFace.vertexPoints.length != pointFS.length * 2) {
                oneFace.vertexPoints = new float[pointFS.length * 2];
            }
            for (int i = 0; i < pointFS.length; i++) {
                // orientation = 0、3 表示竖屏，1、2 表示横屏
                float x = (pointFS[i].x / height) * 2 - 1;
                float y = (pointFS[i].y / width) * 2 - 1;
                float[] point = new float[]{x, -y};
                if (orientation == 1) {
                    if (faceTrackParam.previewTrack && faceTrackParam.isBackCamera) {
                        point[0] = -y;
                        point[1] = -x;
                    } else {
                        point[0] = y;
                        point[1] = x;
                    }
                } else if (orientation == 2) {
                    if (faceTrackParam.previewTrack && faceTrackParam.isBackCamera) {
                        point[0] = y;
                        point[1] = x;
                    } else {
                        point[0] = -y;
                        point[1] = -x;
                    }
                } else if (orientation == 3) {
                    point[0] = -x;
                    point[1] = y;
                }
                // 顶点坐标
                if (faceTrackParam.previewTrack) {
                    if (faceTrackParam.isBackCamera) {
                        oneFace.vertexPoints[2 * i] = point[0];
                    } else {
                        oneFace.vertexPoints[2 * i] = -point[0];
                    }
                } else { // 非预览状态下，左右不需要翻转
                    oneFace.vertexPoints[2 * i] = point[0];
                }
                oneFace.vertexPoints[2 * i + 1] = point[1];
            }
            LandmarkEngine.getInstance().putOneFace(0, oneFace);
            LandmarkEngine.getInstance().setFaceSize(1);
            // 检测完成回调
            if (faceTrackParam.trackerCallback != null) {
                faceTrackParam.trackerCallback.onTrackingFinish();
            }
        }
    }

}

