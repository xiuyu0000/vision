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


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import com.mindspore.enginelibrary.train.FaceTrain;
import com.mindspore.enginelibrary.utils.ConUtil;
import com.mindspore.facelibrary.R;
import com.mindspore.utilslibrary.ui.BaseActivity;
import com.mindspore.utilslibrary.ui.common.Constants;

public class FaceResultActivity extends BaseActivity {

    ImageView imageView;
    Bitmap originBitmap;
    private int[] boundingbox;

    @Override
    protected void init() {
        imageView = findViewById(R.id.img_result);

        String originPath = getIntent().getStringExtra(Constants.TRANS_ORIGIN);
        originBitmap = BitmapFactory.decodeFile(originPath).copy(Bitmap.Config.ARGB_8888, true);
        imageView.setImageBitmap(originBitmap);
        drawImage();
    }

    @Override
    public int getLayout() {
        return R.layout.activity_face_result;
    }


    private void drawImage() {
        FaceTrain.getInstance().setContext(this);
        boolean load = FaceTrain.getInstance().loadModel();
        if (!load) {
            Toast.makeText(this, "模型加载失败", Toast.LENGTH_SHORT).show();
            return;
        }

        String result = FaceTrain.getInstance().runBoundingBoxMindSporeNet(originBitmap);
        String[] resultArray = result.split(", ");
        boundingbox = new int[resultArray.length];
        for (int i = 0; i < resultArray.length; i++) {
            boundingbox[i] = Integer.parseInt(resultArray[i]);
        }

        //裁剪原始图片
        Bitmap corpBitmap = corpBitmap(boundingbox, originBitmap);
        String landmarksStr = FaceTrain.getInstance().runPFLDMindSporeNet(corpBitmap);
        String[] landmarkStrs = landmarksStr.split(",");
        float[] landmarks = new float[landmarkStrs.length];
        for (int i = 0; i < landmarkStrs.length; i++) {
            landmarks[i] = Float.valueOf(landmarkStrs[i]);
        }

        for (int i = 0; i < landmarks.length; i++) {
            if (i % 2 == 0) {
                landmarks[i] = landmarks[i] + boundingbox[0];
            } else {
                landmarks[i] = landmarks[i] + boundingbox[1];
            }
        }

        // 画点
        drawDot(originBitmap, landmarks);
    }

    public Bitmap corpBitmap(int[] boudingbox, Bitmap bitmap) {
        return Bitmap.createBitmap(bitmap, boudingbox[0], boudingbox[1], boudingbox[2], boudingbox[3]);
    }

    public void drawDot(Bitmap bitmap, float[] fts) {
        Canvas canvas = new Canvas(bitmap);

        Paint paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setStrokeWidth(bitmap.getWidth()/200);
        paint.setAntiAlias(true);
        paint.setStrokeCap(Paint.Cap.ROUND);
        canvas.drawPoints(fts, paint);

        paint.setStyle(Paint.Style.STROKE);
        canvas.drawRect(boundingbox[0], boundingbox[1], boundingbox[2] + boundingbox[0], boundingbox[3] + boundingbox[1], paint);

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        FaceTrain.getInstance().unloadModel();
    }
}