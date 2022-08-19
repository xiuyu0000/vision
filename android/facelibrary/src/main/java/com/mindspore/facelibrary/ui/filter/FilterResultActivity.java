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

package com.mindspore.facelibrary.ui.filter;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.widget.ImageView;

import com.mindspore.facelibrary.R;
import com.mindspore.facelibrary.ui.face.FaceBeautyView;
import com.mindspore.utilslibrary.ui.BaseActivity;
import com.mindspore.utilslibrary.ui.common.Constants;

public class FilterResultActivity extends BaseActivity {
    private ImageView imageView;
    private Bitmap originBitmap;
    private FaceBeautyView beautyView;

    @Override
    protected void init() {
        imageView = findViewById(R.id.img_result);

        String originPath = getIntent().getStringExtra(Constants.TRANS_ORIGIN);
        originBitmap = BitmapFactory.decodeFile(originPath).copy(Bitmap.Config.ARGB_8888, true);
        imageView.setImageBitmap(originBitmap);

        beautyView =findViewById(R.id.faceBeautyView);
        beautyView.setBeautyStateListener((state, progress) -> {
            //state  0:磨皮  1:净肤 2:清晰 3:美白
            //progress  进度条进度
        });
    }

    @Override
    public int getLayout() {
        return R.layout.activity_filter_result;
    }
}