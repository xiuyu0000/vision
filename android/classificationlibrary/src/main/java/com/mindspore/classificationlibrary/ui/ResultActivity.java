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
package com.mindspore.classificationlibrary.ui;

import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import com.mindspore.classificationlibrary.R;
import com.mindspore.utilslibrary.ui.BaseActivity;
import com.mindspore.utilslibrary.ui.view.AppTitleView;
import com.mindspore.utilslibrary.ui.common.Constants;

import static com.mindspore.utilslibrary.ui.common.Constants.TRANS_MODEL;

public class ResultActivity extends BaseActivity {

    @Override
    public int getLayout() {
        return R.layout.activity_result;
    }

    protected void init() {
        int model = getIntent().getIntExtra(TRANS_MODEL, 0);

        AppTitleView appTitleView = findViewById(R.id.toolbar);
        FragmentManager fragmentManager = getSupportFragmentManager();
        FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();

        if (Constants.TAG_CLASSIFICATION == model) {
            appTitleView.setTitleText(R.string.image_camera_title);
            String originPath = getIntent().getStringExtra(Constants.TRANS_ORIGIN);
            fragmentTransaction.add(R.id.frameLayout, ClassificationFragment.newInstance(originPath));
        } else if (Constants.TAG_DETECTION == model) {
            appTitleView.setTitleText(R.string.image_detection_title);
            String originPath = getIntent().getStringExtra(Constants.TRANS_ORIGIN);
            fragmentTransaction.add(R.id.frameLayout, DetectionFragment.newInstance(originPath));
        }
        fragmentTransaction.commitAllowingStateLoss();
    }


}


