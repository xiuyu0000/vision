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

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.preference.PreferenceManager;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.core.util.Pair;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;

import com.bumptech.glide.Glide;
import com.google.android.material.tabs.TabLayout;
import com.mindspore.classificationlibrary.R;
import com.mindspore.classificationlibrary.bean.CommonResultBean;
import com.mindspore.classificationlibrary.bean.ModelLabelBean;
import com.mindspore.classificationlibrary.bean.ObjectResultBean;
import com.mindspore.classificationlibrary.help.ResultJsonHelper;
import com.mindspore.classificationlibrary.help.ResultTabHelper;
import com.mindspore.enginelibrary.train.DetectionTrain;
import com.mindspore.utilslibrary.ui.adapter.BasePagerAdapter;
import com.mindspore.utilslibrary.ui.common.Constants;
import com.mindspore.utilslibrary.ui.utils.BitmapUtils;
import com.mindspore.utilslibrary.ui.utils.DisplayUtil;
import com.mindspore.utilslibrary.ui.utils.PathUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class DetectionFragment extends Fragment {

    private static final String ORIGIN_PATH = "originPath";
    private String originPath;

    private final static int DEALY_TIME = 500;
    private final static int REQUEST_FILE_PATH = 1;

    private final static float DRAW_TEXT_SIZE = 45.0f;
    private final static float DRAW_UP_TEXT_SIZE = 15.0f;
    private final static float DRAW_LIMIT_TEXT_SIZE = 30.0f;
    private final static int NUMBER_ZERO = 0;
    private final static int NUMBER_TWO = 2;
    private final static int NUMBER_THREE = 3;
    private final static int NUMBER_HUNDRED = 100;
    private final static String MS = "ms";
    private final static String JSON = "json";
    private final static String MS_NAME = "detection.ms";

    private static final int[] COLORS = {R.color.white, R.color.text_blue,
            R.color.text_yellow, R.color.text_orange, R.color.text_green};

    private Bitmap originBitmap;
    private ViewPager viewPager;
    private ResultChildFragment defaultFragment;
    private ResultChildFragment customFragment;

    private DetectionTrain commonTrackingMobile;
    private DetectionTrain customTrackingMobile;

    private final List<String> ModelTabList = new ArrayList<>();
    private File MS_FILE;

    private List<Fragment> fragments;
    private BasePagerAdapter adapter;
    private SharedPreferences prefs;

    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    boolean isLandScape;
    private ImageView imageView;


    public static DetectionFragment newInstance(String originPath) {
        DetectionFragment fragment = new DetectionFragment();
        Bundle args = new Bundle();
        args.putString(ORIGIN_PATH, originPath);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.isLandScape =
                (this.getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE);
        if (getArguments() != null) {
            originPath = getArguments().getString(ORIGIN_PATH);
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view = inflater.inflate(R.layout.fragment_classification, container, false);
        imageView = view.findViewById(R.id.img_result);
        imageView.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                imageView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                Pair<Integer, Integer> targetedSize = getTargetSize();
                originBitmap = BitmapUtils.loadFromPath(originPath, targetedSize.first, targetedSize.second)
                        .copy(Bitmap.Config.ARGB_8888, true);
            }
        });

        TabLayout mTabLayout = view.findViewById(R.id.tab_layout);
        viewPager = view.findViewById(R.id.viewPager);
        LinearLayout customLayout = view.findViewById(R.id.custom_layout);
        customLayout.setOnClickListener(this::onClick);
        ModelTabList.add(getString(R.string.tab_common));
        initViewPager(mTabLayout);
        initTabLayout(mTabLayout);
        return view;
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        prefs = PreferenceManager.getDefaultSharedPreferences(getActivity());

        if (prefs.getBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false)) {
            String customPath = prefs.getString(Constants.KEY_CLASSIFICATION_CUSTOM_PATH, "");
            if (!TextUtils.isEmpty(customPath)) {
                addCustomJsonFile(customPath);
            } else {
                prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
                Toast.makeText(getActivity(), R.string.custom_file_error, Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void initTabLayout(TabLayout mTabLayout) {
        ResultTabHelper resultTabHelper = new ResultTabHelper(getActivity(), mTabLayout, ModelTabList);
        resultTabHelper.initTabLayout(new ResultTabHelper.BaseOnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                viewPager.setCurrentItem(tab.getPosition(), true);
                if (tab.getCustomView() != null) {
                    new Handler().postDelayed(() -> initDetectionModel(tab.getPosition()), DEALY_TIME);
                }
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {
                unloadModel();
            }

        });
    }

    private void initViewPager(TabLayout mTabLayout) {
        defaultFragment = ResultChildFragment.newInstance(true);

        fragments = new ArrayList<>();
        fragments.add(defaultFragment);

        adapter = new BasePagerAdapter(getChildFragmentManager(), fragments);
        viewPager.setAdapter(adapter);
        viewPager.addOnPageChangeListener(new TabLayout.TabLayoutOnPageChangeListener(mTabLayout));
        viewPager.setOffscreenPageLimit(NUMBER_THREE);
        viewPager.setCurrentItem(NUMBER_ZERO);
    }


    private void initDetectionModel(int position) {
        if (position == NUMBER_ZERO) {
            commonTrackingMobile = new DetectionTrain(getActivity());
            boolean ret = commonTrackingMobile.loadModelFromBuf(MS_NAME, DetectionTrain.TAG_COMMON);
            if (!ret) {
                Toast.makeText(getActivity(), R.string.load_model_error, Toast.LENGTH_SHORT).show();
                return;
            }
            // run net.
            long startTime = System.currentTimeMillis();
            String result = commonTrackingMobile.MindSpore_runnet(originBitmap);
            long endTime = System.currentTimeMillis();
            List<CommonResultBean> list = new ArrayList<>();

            Glide.with(getActivity()).asBitmap().load(originBitmap).into(imageView);
            List<ObjectResultBean> objectBeanList = ObjectResultBean.getRecognitionList(getContext(), result);

            if (objectBeanList.size() > NUMBER_ZERO) {
                drawRect(originBitmap, objectBeanList);

                for (ObjectResultBean resultBean : objectBeanList) {
                    String positionSB = Math.round(resultBean.getLeft()) +
                            ", " + Math.round(resultBean.getTop()) +
                            ", " + Math.round(resultBean.getRight()) +
                            ", " + Math.round(resultBean.getBottom());
                    CommonResultBean commonResultBean = new CommonResultBean.Builder()
                            .setTitle(resultBean.getRectID() + ". " + resultBean.getObjectName())
                            .setContent(String.format(Locale.CHINA, "%.2f",
                                    (NUMBER_HUNDRED * resultBean.getScore())) + "%")
                            .setScore(resultBean.getScore())
                            .setPosition(positionSB)
                            .build();
                    list.add(commonResultBean);
                }
            } else {

            }
            String time = (endTime - startTime) + MS;
            CommonResultBean commonResultBean = new CommonResultBean.Builder()
                    .setTitle(getString(R.string.inference_time))
                    .setContent(time)
                    .build();
            list.add(commonResultBean);
            defaultFragment.setDataList(list);

        } else {
            //to do
        }
    }

    public void addCustomJsonFile(String path) {
        File JSON_FILE = new File(path);
        if (JSON_FILE.exists() && path.toLowerCase().endsWith(JSON)) {
            ModelLabelBean bean = ResultJsonHelper.gonsAnalyzeJSON(JSON_FILE.getPath());
            if (!TextUtils.isEmpty(bean.getTitle()) && !TextUtils.isEmpty(bean.getFile())
                    && bean.getLabel() != null && bean.getLabel().size() > NUMBER_ZERO) {
                addCustomMSFile(bean, JSON_FILE);
            } else {
                prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
                Toast.makeText(getActivity(), R.string.custom_json_data_error, Toast.LENGTH_SHORT).show();
            }
        } else {
            prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
            Toast.makeText(getActivity(), R.string.custom_json_file_error, Toast.LENGTH_SHORT).show();
        }
    }

    public void addCustomMSFile(ModelLabelBean bean, File JSON_FILE) {
        MS_FILE = new File(JSON_FILE.getParent(), bean.getFile());
        if (MS_FILE.exists() && MS_FILE.getPath().toLowerCase().endsWith(MS)) {
            prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, true).apply();
            prefs.edit().putString(Constants.KEY_CLASSIFICATION_CUSTOM_PATH, JSON_FILE.getPath()).apply();
            ModelTabList.add(bean.getTitle());
            addNewCustom();
        } else {
            prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
            Toast.makeText(getActivity(), R.string.custom_ms_file_error, Toast.LENGTH_SHORT).show();
        }
    }

    public void addNewCustom() {
        ResultTabHelper.addTabView(getContext());
        customFragment = ResultChildFragment.newInstance(true);
        fragments.add(customFragment);
        adapter.notifyDataSetChanged();
    }


    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK && requestCode == REQUEST_FILE_PATH && null != data.getData()) {
            Uri uri = data.getData();
            String path = PathUtils.getFilePathByUri(getActivity(), uri);
            if (!TextUtils.isEmpty(path)) {
                addCustomJsonFile(path);
            } else {
                prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
                Toast.makeText(getActivity(), R.string.custom_file_error, Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        ModelTabList.clear();
    }

    // Returns max width of image.
    private Integer getMaxWidthOfImage() {
        if (this.maxWidthOfImage == null) {
            if (this.isLandScape) {
                this.maxWidthOfImage = ((View) imageView.getParent()).getHeight();
            } else {
                this.maxWidthOfImage = ((View) imageView.getParent()).getWidth();
            }
        }
        return this.maxWidthOfImage;
    }

    // Returns max height of image.
    private Integer getMaxHeightOfImage() {
        if (this.maxHeightOfImage == null) {
            if (this.isLandScape) {
                this.maxHeightOfImage = ((View) imageView.getParent()).getWidth();
            } else {
                this.maxHeightOfImage = ((View) imageView.getParent()).getHeight();
            }
        }
        return this.maxHeightOfImage;
    }

    // Gets the targeted size(width / height).
    private Pair<Integer, Integer> getTargetSize() {
        Integer targetWidth;
        Integer targetHeight;
        Integer maxWidth = this.getMaxWidthOfImage();
        Integer maxHeight = this.getMaxHeightOfImage();
        targetWidth = this.isLandScape ? maxHeight : maxWidth;
        targetHeight = this.isLandScape ? maxWidth : maxHeight;
        return new Pair<>(targetWidth, targetHeight);
    }


    private void drawRect(Bitmap bitmap, List<ObjectResultBean> objectBeanList) {
        Canvas canvas = new Canvas(bitmap);
        Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(DisplayUtil.dp2px(getContext(), NUMBER_TWO));
        Paint textPaint = new Paint();
        textPaint.setTextSize(DRAW_TEXT_SIZE);

        for (int i = NUMBER_ZERO; i < objectBeanList.size(); i++) {
            ObjectResultBean bean = objectBeanList.get(i);
            StringBuilder sb = new StringBuilder();
            sb.append(bean.getRectID()).append(". ").append(bean.getObjectName());
            int paintColor = getResources().getColor(COLORS[i % COLORS.length]);
            boxPaint.setColor(paintColor);
            textPaint.setColor(paintColor);

            RectF rectF = new RectF(bean.getLeft(), bean.getTop(), bean.getRight(), bean.getBottom());
            canvas.drawRect(rectF, boxPaint);

            if (bean.getTop() < DRAW_LIMIT_TEXT_SIZE) {
                canvas.drawText(sb.toString(), bean.getLeft(),
                        bean.getTop() + DRAW_TEXT_SIZE, textPaint);
            } else {
                canvas.drawText(sb.toString(), bean.getLeft(),
                        bean.getTop() - DRAW_UP_TEXT_SIZE, textPaint);
            }
        }
    }


    private void unloadModel() {
        if (commonTrackingMobile != null) {
            commonTrackingMobile.unloadModel();
        }
        if (customTrackingMobile != null) {
            customTrackingMobile.unloadModel();
        }
    }

    private void onClick(View v) {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, REQUEST_FILE_PATH);
    }
}