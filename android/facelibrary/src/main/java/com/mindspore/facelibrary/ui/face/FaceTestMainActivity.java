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
 *
 *
 */

package com.mindspore.facelibrary.ui.face;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.mindspore.facelibrary.R;
import com.mindspore.facelibrary.ui.filter.FilterResultActivity;
import com.mindspore.utilslibrary.ui.BaseActivity;
import com.mindspore.utilslibrary.ui.common.Constants;
import com.mindspore.utilslibrary.ui.utils.StorageUtils;
import com.mindspore.utilslibrary.ui.utils.Utils;
import com.mindspore.utilslibrary.ui.view.UpImageDownTextView;

import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FaceTestMainActivity extends BaseActivity implements View.OnClickListener {

    private static final String TAG = "FaceMainActivity";

    private static final int QUALITY = 100;
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int REQUEST_FILE_PATH = 2;

    protected int lensType = CameraSelector.LENS_FACING_BACK;
    private ExecutorService cameraExecutor;
    private PreviewView cameraPreview;
    private ProcessCameraProvider cameraProvider;
    private ImageCapture imageCapture;

    private UpImageDownTextView beautyBtn, stickerBtn, filterBtn, aiBtn;
    private ImageButton btnCamera;
    private FrameLayout frameLayout;

    private boolean isAiChecked = true;
    private boolean isFifterChecked = false;


    @Override
    protected void init() {
        initView();
    }

    private void initView() {
        ImageView closeBtn = findViewById(R.id.closeBtn);
        ImageView moreBtn = findViewById(R.id.moreBtn);
        btnCamera = findViewById(R.id.btnCameraCopy);
        cameraPreview = findViewById(R.id.viewFinder);
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
        cameraPreview.setOnClickListener(this);
        beautyBtn.setOnClickListener(this);
        filterBtn.setOnClickListener(this);
        stickerBtn.setOnClickListener(this);
        aiBtn.setOnClickListener(this);

        aiBtn.setCheck(isAiChecked);
        filterBtn.setCheck(isFifterChecked);
        cameraExecutor = Executors.newSingleThreadExecutor();
    }

    @Override
    public int getLayout() {
        return R.layout.activity_face_test_main;
    }

    @Override
    protected void onResume() {
        super.onResume();
        startCamera();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                imageCapture = new ImageCapture.Builder().build();

                cameraProvider.unbindAll();
                CameraSelector cameraSelector =
                        lensType == CameraSelector.LENS_FACING_BACK ?
                                CameraSelector.DEFAULT_BACK_CAMERA :
                                CameraSelector.DEFAULT_FRONT_CAMERA;

                cameraProvider.bindToLifecycle(this, cameraSelector,
                        preview, imageCapture);
                preview.setSurfaceProvider(cameraPreview.getSurfaceProvider());

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    public void takePhoto() {
        if (imageCapture == null) {
            return;
        }
        if (isAiChecked || isFifterChecked) {
            File root = Build.VERSION.SDK_INT >= Build.VERSION_CODES.R ? getExternalFilesDirs(null)[0] :
                    new File(StorageUtils.ABSOLUTE_FILE, Utils.getApp().getPackageName());
            if (!root.exists()) {
                root.mkdirs();
            }
            File mFile = new File(root, "temp.jpg");
            if (mFile.exists()) {
                mFile.delete();
            }
            cameraPhoto(mFile);
        } else {
            toDoNext();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraProvider.unbindAll();
        cameraExecutor.shutdown();
    }

    private void cameraPhoto(File mFile) {
        ImageCapture.Metadata metadata = new ImageCapture.Metadata();
        metadata.setReversedHorizontal(lensType == CameraSelector.LENS_FACING_FRONT);
        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(mFile)
                .setMetadata(metadata)
                .build();

        imageCapture.takePicture(outputOptions, ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull @NotNull ImageCapture.OutputFileResults outputFileResults) {
                        if (lensType == CameraSelector.LENS_FACING_FRONT) {
                            Bitmap oldMap = BitmapFactory.decodeFile(mFile.getAbsolutePath());
                            Matrix matrix = new Matrix();
                            matrix.setScale(-1.0f, 1.0f);
                            oldMap = Bitmap.createBitmap(oldMap, 0, 0,
                                    oldMap.getWidth(), oldMap.getHeight(), matrix, true);
                            try {
                                FileOutputStream out = new FileOutputStream(mFile);
                                oldMap.compress(Bitmap.CompressFormat.JPEG, QUALITY, out);
                                out.flush();
                                out.close();
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        }
                        showImageToResult(mFile.getPath());
                    }

                    @Override
                    public void onError(@NonNull @NotNull ImageCaptureException exception) {
                        String errorMsg = getResources().getString(R.string.camera_failed) + exception.getMessage();
                        Toast.makeText(FaceTestMainActivity.this, errorMsg, Toast.LENGTH_SHORT).show();
                    }
                });
    }


    private void openGallery() {
        if (isAiChecked ||isFifterChecked) {
            Intent intentToPickPic = new Intent(Intent.ACTION_PICK, null);
            intentToPickPic.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
            startActivityForResult(intentToPickPic, RC_CHOOSE_PHOTO);
        } else {
            toDoNext();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (RC_CHOOSE_PHOTO == requestCode) {
            if (resultCode == Activity.RESULT_OK && null != data.getData()) {
                Uri uri = data.getData();
                String[] proj = {MediaStore.Images.Media.DATA};
                Cursor actualImageCursor = managedQuery(uri, proj, null, null, null);
                int actual_image_column_index = actualImageCursor.getColumnIndex(MediaStore.Images.Media.DATA);
                actualImageCursor.moveToFirst();
                String img_path = actualImageCursor.getString(actual_image_column_index);
                showImageToResult(img_path);
            } else {
                Toast.makeText(this, R.string.photo_failed, Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    private void showImageToResult(String filePath) {
        Intent intent = new Intent(FaceTestMainActivity.this, FilterResultActivity.class);
        intent.putExtra(Constants.TRANS_ORIGIN, filePath);
        startActivity(intent);
    }


    @Override
    public void onClick(View v) {
        int id = v.getId();
        if (id == R.id.beautyBtn || id == R.id.stickerBtn ) {
            toDoNext();
            isAiChecked = false;
            aiBtn.setCheck(isAiChecked);

            isFifterChecked = false;
            filterBtn.setCheck(isFifterChecked);
        }else if(id == R.id.filterBtn){
            isAiChecked = false;
            aiBtn.setCheck(isAiChecked);
            isFifterChecked = true;
            filterBtn.setCheck(isFifterChecked);

        } else if (id == R.id.aiBtn) {
            isAiChecked = true;
            aiBtn.setCheck(isAiChecked);
            isFifterChecked = false;
            filterBtn.setCheck(isFifterChecked);
        } else if (id == R.id.viewFinder) {

        } else if (id == R.id.closeBtn) {
            finish();
        } else if (id == R.id.switchBtn) {
            lensType = lensType == CameraSelector.LENS_FACING_BACK ?
                    CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
            startCamera();
        } else if (id == R.id.galleryBtn) {
            if (isAiChecked || isFifterChecked) {
                openGallery();
            } else {
                toDoNext();
            }
        } else if (id == R.id.btnCameraCopy) {
            if (isAiChecked || isFifterChecked) {
                takePhoto();
            } else {
                toDoNext();
            }
        }
    }

    private void toDoNext() {
        Toast.makeText(this, R.string.app_next, Toast.LENGTH_SHORT).show();
        return;
    }

}