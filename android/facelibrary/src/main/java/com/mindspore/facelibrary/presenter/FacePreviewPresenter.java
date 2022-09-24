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

package com.mindspore.facelibrary.presenter;

import android.animation.AnimatorInflater;
import android.animation.AnimatorSet;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.opengl.EGLContext;
import android.util.Log;
import android.view.View;
import android.view.animation.TranslateAnimation;

import androidx.annotation.NonNull;

import com.mindspore.enginelibrary.engine.face.MSFaceTracker;
import com.mindspore.enginelibrary.listener.FaceTrackerCallback;
import com.mindspore.facelibrary.R;
import com.mindspore.facelibrary.camera.CameraParam;
import com.mindspore.facelibrary.camera.CameraXController;
import com.mindspore.facelibrary.camera.ICameraController;
import com.mindspore.facelibrary.camera.OnFrameAvailableListener;
import com.mindspore.facelibrary.camera.OnSurfaceTextureListener;
import com.mindspore.facelibrary.camera.PreviewCallback;
import com.mindspore.facelibrary.color.bean.DynamicColor;
import com.mindspore.facelibrary.listener.OnCaptureListener;
import com.mindspore.facelibrary.listener.OnFpsListener;
import com.mindspore.facelibrary.render.CameraRenderer;
import com.mindspore.facelibrary.resource.ResourceHelper;
import com.mindspore.facelibrary.resource.ResourceJsonCodec;
import com.mindspore.facelibrary.resource.bean.ResourceData;
import com.mindspore.facelibrary.resource.bean.ResourceType;
import com.mindspore.facelibrary.stickers.bean.DynamicSticker;
import com.mindspore.facelibrary.ui.face.FaceMainActivity;
import com.mindspore.landmarklibrary.LandmarkEngine;

import org.jetbrains.annotations.NotNull;

import java.io.File;

public class FacePreviewPresenter extends PreviewPresenter<FaceMainActivity> implements FaceTrackerCallback, PreviewCallback, OnFrameAvailableListener, OnSurfaceTextureListener, OnCaptureListener, OnFpsListener {
    private FaceMainActivity mainActivity;

    private static final String TAG = "FacePreviewPresenter";


    private CameraParam mCameraParam;
    // 渲染器
    private final CameraRenderer mCameraRenderer;

    // 相机接口
    private ICameraController mCameraController;

    public FacePreviewPresenter(FaceMainActivity target) {
        super(target);
        this.mainActivity = target;

        mCameraParam = CameraParam.getInstance();

        mCameraRenderer = new CameraRenderer(this);
        mCameraRenderer.initRenderer();

//        mCameraController = new CameraController(mainActivity);
        mCameraController = new CameraXController(mainActivity);
        mCameraController.setPreviewCallback(this);
        mCameraController.setOnFrameAvailableListener(this);
        mCameraController.setOnSurfaceTextureListener(this);

        // 初始化检测器
        MSFaceTracker.getInstance()
                .setFaceCallback(this)
                .previewTrack(true)
                .initTracker();
    }

    @Override
    public void onResume() {
        super.onResume();
        openCamera();
        mCameraParam.captureCallback = this;
        mCameraParam.fpsCallback = this;
    }

    @Override
    public void onPause() {
        super.onPause();
        mCameraRenderer.onPause();
        closeCamera();
        mCameraParam.captureCallback = null;
        mCameraParam.fpsCallback = null;
    }

    @Override
    public void onStop() {
        super.onStop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // 销毁人脸检测器
        MSFaceTracker.getInstance().destroyTracker();
        // 清理关键点
        LandmarkEngine.getInstance().clearAll();

        mainActivity = null;
        mCameraRenderer.destroyRenderer();
    }

    @Override
    public void onBindSharedContext(EGLContext context) {

    }

    @Override
    public void onSurfaceCreated(SurfaceTexture surfaceTexture) {
        mCameraRenderer.onSurfaceCreated(surfaceTexture);
    }

    @Override
    public void onSurfaceChanged(int width, int height) {
        mCameraRenderer.onSurfaceChanged(width, height);
    }

    public void onSurfaceDestroyed() {
        mCameraRenderer.onSurfaceDestroyed();
    }

    @Override
    public void changeResource(@NonNull @NotNull ResourceData resourceData) {
        ResourceType type = resourceData.type;
        String unzipFolder = resourceData.unzipFolder;
        if (type == null) {
            return;
        }
        try {
            switch (type) {
                // 单纯的滤镜
                case FILTER: {
                    String folderPath = ResourceHelper.getResourceDirectory(mainActivity) + File.separator + unzipFolder;
                    DynamicColor color = ResourceJsonCodec.decodeFilterData(folderPath);
                    mCameraRenderer.changeResource(color);
                    break;
                }

                // 贴纸
                case STICKER: {
                    String folderPath = ResourceHelper.getResourceDirectory(mainActivity) + File.separator + unzipFolder;
                    DynamicSticker sticker = ResourceJsonCodec.decodeStickerData(folderPath);
                    mCameraRenderer.changeResource(sticker);
                    break;
                }

                // TODO 多种结果混合
                case MULTI: {
                    break;
                }

                // 所有数据均为空
                case NONE: {
                    mCameraRenderer.changeResource((DynamicSticker) null);
                    break;
                }

                default:
                    break;
            }
        } catch (Exception e) {
            Log.e(TAG, "parseResource: ", e);
        }
    }

    @Override
    public void changeDynamicFilter(DynamicColor color) {

    }

    @Override
    public void changeDynamicFilter(int filterIndex) {

    }

    @Override
    public int previewFilter() {
        return 0;
    }

    @Override
    public int nextFilter() {
        return 0;
    }

    @Override
    public int getFilterIndex() {
        return 0;
    }

    @Override
    public void showCompare(boolean enable) {

    }

    @Override
    public void takePicture() {
        mCameraRenderer.takePicture();
    }

    @Override
    public void switchCamera() {
        mCameraController.switchCamera();
    }

    @Override
    public void startRecord() {

    }

    @Override
    public void stopRecord() {

    }

    @Override
    public void cancelRecord() {

    }

    @Override
    public boolean isRecording() {
        return false;
    }

    @Override
    public void changeFlashLight(boolean on) {
        if (mCameraController != null) {
            mCameraController.setFlashLight(on);
        }
    }

    @Override
    public void enableEdgeBlurFilter(boolean enable) {

    }

    @Override
    public void setMusicPath(String path) {

    }

    @Override
    public void onOpenCameraSettingPage() {

    }

    @NonNull
    @NotNull
    @Override
    public Context getContext() {
        return mainActivity;
    }

    public void cameraBtnDown(View btnCamera) {
        AnimatorSet animator = (AnimatorSet) AnimatorInflater
                .loadAnimator(mainActivity, R.animator.camera_btn_fade_out);
        animator.setTarget(btnCamera);
        animator.start();
    }

    public void cameraBtnUp(View btnCamera) {
        AnimatorSet animator = (AnimatorSet) AnimatorInflater
                .loadAnimator(mainActivity, R.animator.camera_btn_fade_in);
        animator.setTarget(btnCamera);
        animator.start();
    }

    public void showFaceView(View frameLayout) {
        final TranslateAnimation ctrlAnimation = new TranslateAnimation(
                TranslateAnimation.RELATIVE_TO_SELF, 0, TranslateAnimation.RELATIVE_TO_SELF, 0,
                TranslateAnimation.RELATIVE_TO_SELF, 1, TranslateAnimation.RELATIVE_TO_SELF, 0);
        ctrlAnimation.setDuration(200L);
        frameLayout.setVisibility(View.VISIBLE);
        frameLayout.startAnimation(ctrlAnimation);

    }

    public void closeFaceView(View frameLayout) {
        final TranslateAnimation ctrlAnimation = new TranslateAnimation(
                TranslateAnimation.RELATIVE_TO_SELF, 0, TranslateAnimation.RELATIVE_TO_SELF, 0,
                TranslateAnimation.RELATIVE_TO_SELF, 0, TranslateAnimation.RELATIVE_TO_SELF, 1);
        ctrlAnimation.setDuration(200L);
        frameLayout.setVisibility(View.INVISIBLE);
        frameLayout.startAnimation(ctrlAnimation);
    }

    /**
     * 打开相机
     */
    private void openCamera() {
        mCameraController.openCamera();
        calculateImageSize();
    }

    /**
     * 计算imageView 的宽高
     */

    private void calculateImageSize() {
        int width;
        int height;
        if (mCameraController.getOrientation() == 90 || mCameraController.getOrientation() == 270) {
            width = mCameraController.getPreviewHeight();
            height = mCameraController.getPreviewWidth();
        } else {
            width = mCameraController.getPreviewWidth();
            height = mCameraController.getPreviewHeight();
        }
//        mVideoParams.setVideoSize(width, height);
        mCameraRenderer.setTextureSize(width, height);
    }

    /**
     * 关闭相机
     */
    private void closeCamera() {
        mCameraController.closeCamera();
    }

    @Override
    public void onTrackingFinish() {
        mCameraRenderer.requestRender();
    }

    @Override
    public void onPreviewFrame(byte[] data) {
        Log.d(TAG, "onPreviewFrame: width - " + mCameraController.getPreviewWidth()
                + ", height - " + mCameraController.getPreviewHeight());
        MSFaceTracker.getInstance()
                .trackFace(data, mCameraController.getPreviewWidth(),
                        mCameraController.getPreviewHeight());
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
//        mCameraRenderer.requestRender();
    }

    /**
     * 相机打开回调
     */
    public void onCameraOpened() {
        MSFaceTracker.getInstance()
                .setBackCamera(!mCameraController.isFront())
                .prepareFaceTracker(mainActivity,
                        mCameraController.getOrientation(),
                        mCameraController.getPreviewWidth(),
                        mCameraController.getPreviewHeight());
    }

    @Override
    public void onSurfaceTexturePrepared(@NonNull @NotNull SurfaceTexture surfaceTexture) {
        onCameraOpened();
        mCameraRenderer.bindInputSurfaceTexture(surfaceTexture);
    }

    @Override
    public void onCapture(Bitmap bitmap) {

    }

    @Override
    public void onFpsCallback(float fps) {

    }
}
