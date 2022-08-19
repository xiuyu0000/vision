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
package com.mindspore.vision.ui;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.provider.Settings;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.mindspore.classificationlibrary.bean.ModelLabelBean;
import com.mindspore.classificationlibrary.ui.ResultActivity;
import com.mindspore.facelibrary.ui.face.FaceMainActivity;
import com.mindspore.facelibrary.ui.face.FaceTestMainActivity;
import com.mindspore.utilslibrary.ui.BaseActivity;
import com.mindspore.utilslibrary.ui.view.UpImageDownTextView;
import com.mindspore.vision.R;
import com.mindspore.utilslibrary.ui.common.Constants;
import com.mindspore.classificationlibrary.help.ResultJsonHelper;
import com.mindspore.utilslibrary.ui.utils.PathUtils;
import com.mindspore.utilslibrary.ui.utils.StorageUtils;
import com.mindspore.utilslibrary.ui.utils.Utils;
import com.tbruyelle.rxpermissions2.RxPermissions;

import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends BaseActivity implements View.OnClickListener, View.OnLongClickListener {

    private static final String[] PERMISSIONS = {Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA};

    private final static String MS = "ms";
    private final static String JSON = "json";

    protected int lensType = CameraSelector.LENS_FACING_BACK;
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int REQUEST_FILE_PATH = 2;
    private static final int REQUEST_R_PERMISSIONS = 3;

    private static final int QUALITY = 100;
    private static final long WAIT_TIME = 2000;
    private long touchTime = 0;

    private ExecutorService cameraExecutor;
    private PreviewView cameraPreview;
    private ImageCapture imageCapture;
    private UpImageDownTextView classBtn, detectionBtn, faceBtn, filterBtn;
    private int model = Constants.TAG_CLASSIFICATION;
    private SharedPreferences prefs;
    private ProcessCameraProvider cameraProvider;

    @Override
    public int getLayout() {
        return R.layout.activity_main;
    }

    @Override
    protected void init() {
        ImageView closeBtn = findViewById(R.id.closeBtn);
        ImageView moreBtn = findViewById(R.id.moreBtn);
        ImageButton btnCamera = findViewById(R.id.btnCamera);
        cameraPreview = findViewById(R.id.viewFinder);
        ImageView switchBtn = findViewById(R.id.switchBtn);
        ImageView galleryBtn = findViewById(R.id.galleryBtn);
        classBtn = findViewById(R.id.btnOne);
        detectionBtn = findViewById(R.id.btnTwo);
        faceBtn = findViewById(R.id.btnThree);
        filterBtn = findViewById(R.id.btnFour);
        closeBtn.setOnClickListener(this);
        moreBtn.setOnClickListener(this);
        btnCamera.setOnClickListener(this);
        switchBtn.setOnClickListener(this);
        galleryBtn.setOnClickListener(this);
        classBtn.setOnClickListener(this);
        detectionBtn.setOnClickListener(this);
        faceBtn.setOnClickListener(this);
        filterBtn.setOnClickListener(this);
        classBtn.setOnLongClickListener(this);
        buttonGroupStatus(Constants.TAG_CLASSIFICATION);

        prefs = PreferenceManager.getDefaultSharedPreferences(this);

        cameraExecutor = Executors.newSingleThreadExecutor();
    }

    @Override
    protected void onResume() {
        super.onResume();
        requestPermissions();
    }

    @SuppressLint("CheckResult")
    private void requestPermissions() {
        new RxPermissions(this)
                .request(PERMISSIONS)
                .subscribe(granted -> {
                    if (granted) {
                        startCamera();
                    } else {
                        openAppDetails();
                    }
                });
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

        if (model == Constants.TAG_CLASSIFICATION || model == Constants.TAG_DETECTION || model == Constants.TAG_FACE) {
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
            Toast.makeText(this, getResources().getString(R.string.app_next), (int) WAIT_TIME).show();
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
                        Toast.makeText(MainActivity.this, errorMsg, Toast.LENGTH_SHORT).show();
                    }
                });
    }

    private void openAppDetails() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage(getString(R.string.app_choose_authority));
        builder.setPositiveButton(getString(R.string.app_choose_authority_manual), (dialog, which) -> {
            Intent intent = new Intent();
            intent.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
            intent.addCategory(Intent.CATEGORY_DEFAULT);
            intent.setData(Uri.parse("package:" + getPackageName()));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            intent.addFlags(Intent.FLAG_ACTIVITY_NO_HISTORY);
            intent.addFlags(Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS);
            startActivity(intent);
        });
        builder.setNegativeButton(getString(R.string.app_choose_cancle), null);
        builder.show();
    }

    @Override
    public void onClick(View v) {
        int id = v.getId();
        if (id == R.id.closeBtn) {
            exitApp();
        } else if (id == R.id.switchBtn) {
            lensType = lensType == CameraSelector.LENS_FACING_BACK ?
                    CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
            startCamera();
        } else if (id == R.id.galleryBtn) {
            openGallery();
        } else if (id == R.id.btnCamera) {
            takePhoto();
        } else if (id == R.id.btnOne) {
            model = Constants.TAG_CLASSIFICATION;
            buttonGroupStatus(Constants.TAG_CLASSIFICATION);
        } else if (id == R.id.btnTwo) {
            model = Constants.TAG_DETECTION;
            buttonGroupStatus(Constants.TAG_DETECTION);
        } else if (id == R.id.btnThree) {
            model = Constants.TAG_FACE;
            buttonGroupStatus(Constants.TAG_FACE);
            Intent intent = new Intent(MainActivity.this, FaceTestMainActivity.class);
            startActivity(intent);
        } else if (id == R.id.btnFour) {
            model = Constants.TAG_FILTER;
            buttonGroupStatus(Constants.TAG_FILTER);
        }
    }

    @Override
    public boolean onLongClick(View v) {
        if (v.getId() == R.id.btnOne) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (Environment.isExternalStorageManager()) {
                    openDocument();
                } else {
                    Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                    intent.setData(Uri.parse("package:" + getPackageName()));
                    startActivityForResult(intent, REQUEST_R_PERMISSIONS);
                }
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                openDocument();
            } else {
                openDocument();
            }
        }
        return false;
    }

    private void openDocument() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, REQUEST_FILE_PATH);
    }

    private void openGallery() {
        if (model == Constants.TAG_CLASSIFICATION || model == Constants.TAG_DETECTION || model == Constants.TAG_FACE) {
            Intent intentToPickPic = new Intent(Intent.ACTION_PICK, null);
            intentToPickPic.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
            startActivityForResult(intentToPickPic, RC_CHOOSE_PHOTO);
        } else {
            Toast.makeText(this, getResources().getString(R.string.app_next), (int) WAIT_TIME).show();
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
        } else if (REQUEST_FILE_PATH == requestCode) {
            if (resultCode == Activity.RESULT_OK && data.getData() != null) {

                Uri uri = data.getData();
                String path = PathUtils.getFilePathByUri(MainActivity.this, uri);
                if (!TextUtils.isEmpty(path)) {
                    addCustomJsonFile(path);
                } else {
                    prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
                    Toast.makeText(this, R.string.custom_file_error, Toast.LENGTH_SHORT).show();
                }
            }
        } else if (REQUEST_R_PERMISSIONS == requestCode) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (Environment.isExternalStorageManager()) {
                    openDocument();
                } else {
                    Toast.makeText(this, R.string.app_save_permissions_failed, Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    public void addCustomJsonFile(String path) {
        File JSON_FILE = new File(path);
        if (JSON_FILE.exists() && path.toLowerCase().endsWith(JSON)) {
            ModelLabelBean bean = ResultJsonHelper.gonsAnalyzeJSON(JSON_FILE.getPath());
            if (!TextUtils.isEmpty(bean.getTitle()) && !TextUtils.isEmpty(bean.getFile())
                    && bean.getLabel() != null && bean.getLabel().size() > 0) {
                addCustomMSFile(bean, JSON_FILE);
            } else {
                prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
                Toast.makeText(this, R.string.custom_json_data_error, Toast.LENGTH_SHORT).show();
            }
        } else {
            prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
            Toast.makeText(this, R.string.custom_json_file_error, Toast.LENGTH_SHORT).show();
        }
    }


    public void addCustomMSFile(ModelLabelBean bean, File JSON_FILE) {
        File MS_FILE = new File(JSON_FILE.getParent(), bean.getFile());
        if (MS_FILE.exists() && MS_FILE.getPath().toLowerCase().endsWith(MS)) {
            prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, true).apply();
            prefs.edit().putString(Constants.KEY_CLASSIFICATION_CUSTOM_PATH, JSON_FILE.getPath()).apply();
            Toast.makeText(this, R.string.custom_add_success, Toast.LENGTH_SHORT).show();
        } else {
            prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
            Toast.makeText(this, R.string.custom_ms_file_error, Toast.LENGTH_SHORT).show();
        }
    }

    private void showImageToResult(String filePath) {
        Intent intent = new Intent(MainActivity.this, ResultActivity.class);
        intent.putExtra(Constants.TRANS_MODEL, model);
        intent.putExtra(Constants.TRANS_ORIGIN, filePath);
        startActivity(intent);
    }

    public void buttonGroupStatus(int index) {
        switch (index) {
            case Constants.TAG_DETECTION:
                classBtn.setUnChecked();
                detectionBtn.setChecked();
                faceBtn.setUnChecked();
                filterBtn.setUnChecked();
                break;
            case Constants.TAG_FACE:
                classBtn.setUnChecked();
                detectionBtn.setUnChecked();
                faceBtn.setChecked();
                filterBtn.setUnChecked();
                break;
            case Constants.TAG_FILTER:
                classBtn.setUnChecked();
                detectionBtn.setUnChecked();
                faceBtn.setUnChecked();
                filterBtn.setChecked();
                break;
            case Constants.TAG_SEGMENTATION:

                break;
            default:
                classBtn.setChecked();
                detectionBtn.setUnChecked();
                faceBtn.setUnChecked();
                filterBtn.setUnChecked();
                break;
        }
    }

    @Override
    public void onBackPressed() {
        exitApp();
    }

    private void exitApp() {
        long currentTime = System.currentTimeMillis();
        if ((currentTime - touchTime) >= WAIT_TIME) {
            Toast.makeText(this, getResources().getString(R.string.app_exit), (int) WAIT_TIME).show();
            touchTime = currentTime;
        } else {
            System.exit(0);
        }
    }


}