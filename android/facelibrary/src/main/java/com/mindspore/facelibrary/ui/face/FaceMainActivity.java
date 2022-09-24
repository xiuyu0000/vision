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

package com.mindspore.facelibrary.ui.face;

import android.graphics.SurfaceTexture;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;

import com.mindspore.enginelibrary.engine.FaceTracker;
import com.mindspore.facelibrary.R;
import com.mindspore.facelibrary.camera.CameraParam;
import com.mindspore.facelibrary.event.FaceCloseEvent;
import com.mindspore.facelibrary.presenter.FacePreviewPresenter;
import com.mindspore.facelibrary.resource.ResourceHelper;
import com.mindspore.facelibrary.resource.bean.ResourceData;
import com.mindspore.facelibrary.ui.widget.CameraTextureView;
import com.mindspore.utilslibrary.ui.BaseActivity;
import com.mindspore.utilslibrary.ui.view.UpImageDownTextView;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;

public class FaceMainActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "FaceMainActivity";

    private FacePreviewPresenter facePreviewPresenter;
    private CameraTextureView cameraTextureView;
    private UpImageDownTextView beautyBtn, stickerBtn, filterBtn, aiBtn;
    private ImageButton btnCamera;
    private FrameLayout frameLayout;

    private FaceBeautyView faceBeautyView;
    private FaceStickerView faceStickerView;

    @Override
    protected void onStart() {
        super.onStart();
        EventBus.getDefault().register(this);
    }

    @Subscribe(threadMode = ThreadMode.MAIN)
    public void onMessageEvent(FaceCloseEvent event) {
        if (frameLayout.getVisibility() == View.VISIBLE) {
            facePreviewPresenter.closeFaceView(frameLayout);
            facePreviewPresenter.cameraBtnUp(btnCamera);
        }
    }

    @Override
    protected void init() {
        facePreviewPresenter = new FacePreviewPresenter(this);
        initView();
//        faceTrackerRequestNetwork();
        initResources();
    }

    private void initView() {
        ImageView closeBtn = findViewById(R.id.closeBtn);
        ImageView moreBtn = findViewById(R.id.moreBtn);
        btnCamera = findViewById(R.id.btnCameraCopy);
        cameraTextureView = findViewById(R.id.viewFinder);
        ImageView switchBtn = findViewById(R.id.switchBtn);
        ImageView galleryBtn = findViewById(R.id.galleryBtn);
        beautyBtn = findViewById(R.id.beautyBtn);
        stickerBtn = findViewById(R.id.stickerBtn);
        filterBtn = findViewById(R.id.filterBtn);
        aiBtn = findViewById(R.id.aiBtn);
        frameLayout = findViewById(R.id.frameLayout);
        closeBtn.setOnClickListener(this);
        moreBtn.setOnClickListener(this);
        btnCamera.setOnClickListener(this);
        switchBtn.setOnClickListener(this);
        galleryBtn.setOnClickListener(this);
        cameraTextureView.setOnClickListener(this);
        beautyBtn.setOnClickListener(this);
        stickerBtn.setOnClickListener(this);
        aiBtn.setOnClickListener(this);
        aiBtn.setOnClickListener(this);

        cameraTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
    }

    @Override
    public int getLayout() {
        return R.layout.activity_face_main;
    }

    @Override
    protected void onResume() {
        super.onResume();
        facePreviewPresenter.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
        facePreviewPresenter.onPause();
    }

    @Override
    public void onStop() {
        super.onStop();
        facePreviewPresenter.onStop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        EventBus.getDefault().unregister(this);
        facePreviewPresenter.onDestroy();
        facePreviewPresenter = null;
    }

    /**
     * 人脸检测SDK验证，可以替换成自己的SDK
     */
    private void faceTrackerRequestNetwork() {
        new Thread(() -> FaceTracker.requestFaceNetwork(this)).start();
    }

    private void initResources() {
        new Thread(() -> {
            ResourceHelper.initAssetsResource(this);
//            FilterHelper.initAssetsFilter(MainActivity.this);
//            MakeupHelper.initAssetsMakeup(MainActivity.this);
        }).start();
    }

    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.beautyBtn) {
            if (faceBeautyView == null) {
                faceBeautyView = new FaceBeautyView(this);
            }
            frameLayout.removeAllViews();
            frameLayout.addView(faceBeautyView);

            facePreviewPresenter.showFaceView(frameLayout);
            facePreviewPresenter.cameraBtnDown(btnCamera);
        } else if (v.getId() == R.id.stickerBtn) {
            if (faceStickerView == null) {
                faceStickerView = new FaceStickerView(this);
                faceStickerView.addOnChangeResourceListener(new FaceStickerView.OnResourceChangeListener() {
                    @Override
                    public void onResourceChange(ResourceData data) {
                        facePreviewPresenter.changeResource(data);
                    }
                });
            }
            frameLayout.removeAllViews();
            frameLayout.addView(faceStickerView);

            facePreviewPresenter.showFaceView(frameLayout);
            facePreviewPresenter.cameraBtnDown(btnCamera);

        } else if (v.getId() == R.id.viewFinder) {
            if (frameLayout.getVisibility() == View.VISIBLE) {
                facePreviewPresenter.closeFaceView(frameLayout);
                facePreviewPresenter.cameraBtnUp(btnCamera);
            }
        } else if (v.getId() == R.id.closeBtn) {
            finish();
        } else if (v.getId() == R.id.aiBtn) {
            CameraParam.getInstance().drawFacePoints = !CameraParam.getInstance().drawFacePoints;
            aiBtn.setCheck(CameraParam.getInstance().drawFacePoints);
        } else if (v.getId() == R.id.switchBtn) {
            facePreviewPresenter.switchCamera();
        }
    }

    // ---------------------------- TextureView SurfaceTexture监听 ---------------------------------
    private TextureView.SurfaceTextureListener mSurfaceTextureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            facePreviewPresenter.onSurfaceCreated(surface);
            facePreviewPresenter.onSurfaceChanged(width, height);
            Log.d(TAG, "onSurfaceTextureAvailable: ");
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            facePreviewPresenter.onSurfaceChanged(width, height);
            Log.d(TAG, "onSurfaceTextureSizeChanged: ");
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
//            facePreviewPresenter.onSurfaceDestroyed();
            Log.d(TAG, "onSurfaceTextureDestroyed: ");
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {

        }
    };


}