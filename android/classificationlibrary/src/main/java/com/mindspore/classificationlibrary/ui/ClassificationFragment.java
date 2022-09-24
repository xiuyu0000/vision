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

package com.mindspore.classificationlibrary.ui;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.preference.PreferenceManager;
import android.provider.Settings;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.viewpager.widget.ViewPager;

import com.bumptech.glide.Glide;
import com.google.android.material.tabs.TabLayout;
import com.mindspore.classificationlibrary.R;
import com.mindspore.classificationlibrary.bean.CommonResultBean;
import com.mindspore.classificationlibrary.bean.ModelLabelBean;
import com.mindspore.classificationlibrary.help.ResultJsonHelper;
import com.mindspore.classificationlibrary.help.ResultTabHelper;
import com.mindspore.enginelibrary.train.ClassificationTrain;
import com.mindspore.utilslibrary.ui.adapter.BasePagerAdapter;
import com.mindspore.utilslibrary.ui.common.Constants;
import com.mindspore.utilslibrary.ui.utils.PathUtils;

import java.io.File;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * A simple {@link Fragment} subclass.
 * Use the {@link ClassificationFragment#newInstance} factory method to
 * create an instance of this fragment.
 */
public class ClassificationFragment extends Fragment {

    private static final String ORIGIN_PATH = "originPath";
    private final static int DELAY_TIME = 500;
    private final static int REQUEST_FILE_PATH = 1;
    private static final int REQUEST_R_PERMISSIONS = 2;

    private final static int NUMBER_HUNDRED = 100;
    private final static int NUMBER_ZERO = 0;
    private final static int NUMBER_ONE = 1;
    private final static int NUMBER_THREE = 3;
    private final static float THRESHOLD = 0.5f;
    private final static int HANDLER_ONE = 1;

    private final static String MS = "ms";
    private final static String JSON = "json";
    private final static String MS_NAME = "mobilenetv2.ms";

    private String originPath;
    private Bitmap originClassificationBitmap;
    private ViewPager viewPager;
    private ResultChildFragment defaultFragment;
    private ResultChildFragment customFragment;

    private ClassificationTrain commonTrackingMobile;
    private ClassificationTrain customTrackingMobile;

    private final List<String> modelTabList = new ArrayList<>();
    private File MS_FILE;
    private List<String> customLabelist;

    private List<Fragment> fragments;
    private BasePagerAdapter adapter;
    private SharedPreferences prefs;

    private final MyHandler myHandler = new MyHandler(this);
    public int tabPosition;

    public static ClassificationFragment newInstance(String originPath) {
        ClassificationFragment fragment = new ClassificationFragment();
        Bundle args = new Bundle();
        args.putString(ORIGIN_PATH, originPath);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            originPath = getArguments().getString(ORIGIN_PATH);
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view = inflater.inflate(R.layout.fragment_classification, container, false);

        ImageView imageView = view.findViewById(R.id.img_result);
        originClassificationBitmap = BitmapFactory.decodeFile(originPath);
        Glide.with(this).asBitmap().load(originClassificationBitmap).into(imageView);
        TabLayout mTabLayout = view.findViewById(R.id.tab_layout);
        viewPager = view.findViewById(R.id.viewPager);
        LinearLayout customLayout = view.findViewById(R.id.custom_layout);
        customLayout.setOnClickListener(this::onClick);
        modelTabList.add(getString(R.string.tab_common));
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
        ResultTabHelper resultTabHelper = new ResultTabHelper(getActivity(), mTabLayout, modelTabList);
        resultTabHelper.initTabLayout(new ResultTabHelper.BaseOnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                viewPager.setCurrentItem(tab.getPosition(), true);
                if (tab.getCustomView() != null) {
                    tabPosition = tab.getPosition();
                    myHandler.sendEmptyMessageDelayed(HANDLER_ONE, DELAY_TIME);
                }
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {
                unloadModel();
            }

        });
    }

    public void initModel() {
        initClassificationModel(tabPosition);
    }

    private void initViewPager(TabLayout mTabLayout) {
        defaultFragment = ResultChildFragment.newInstance(false);
        fragments = new ArrayList<>();
        fragments.add(defaultFragment);

        adapter = new BasePagerAdapter(getChildFragmentManager(), fragments);
        viewPager.setAdapter(adapter);
        viewPager.addOnPageChangeListener(new TabLayout.TabLayoutOnPageChangeListener(mTabLayout));
        viewPager.setOffscreenPageLimit(NUMBER_THREE);
        viewPager.setCurrentItem(NUMBER_ZERO);
    }

    public void initClassificationModel(int position) {
        if (position == NUMBER_ZERO) {
            commonTrackingMobile = new ClassificationTrain(getActivity());
            boolean ret = commonTrackingMobile.loadModelFromBuf(MS_NAME, ClassificationTrain.TAG_COMMON);
            if (!ret) {
                Toast.makeText(getActivity(), R.string.load_model_error, Toast.LENGTH_SHORT).show();
                return;
            }
            initCommonData();
        } else {
            customTrackingMobile = new ClassificationTrain(getActivity());
            boolean ret = customTrackingMobile.loadModelFromBuf(MS_FILE.getPath(), ClassificationTrain.TAG_CUSTOM);
            if (!ret) {
                Toast.makeText(getActivity(), R.string.load_model_error, Toast.LENGTH_SHORT).show();
                return;
            }
            initCustomData();
        }
    }

    private void initCommonData() {
        // run net.
        long startTime = System.currentTimeMillis();
        String result = commonTrackingMobile.MindSpore_runnet(originClassificationBitmap, NUMBER_ZERO);
        long endTime = System.currentTimeMillis();
        List<CommonResultBean> list = new ArrayList<>();

        String[] CONTENT_ARRAY = getResources().getStringArray(R.array.image_category);
        if (!TextUtils.isEmpty(result)) {
            String[] resultArray = result.split(";");
            if (resultArray.length > NUMBER_ZERO) {
                for (String singleRecognitionResult : resultArray) {
                    String[] singleResult = singleRecognitionResult.split(":");
                    if (singleResult.length > NUMBER_ONE) {
                        int nameIndex = Integer.parseInt(singleResult[NUMBER_ZERO]);
                        float score = Float.parseFloat(singleResult[NUMBER_ONE]);
                        if (score > THRESHOLD) {
                            CommonResultBean commonResultBean = new CommonResultBean.Builder()
                                    .setTitle(CONTENT_ARRAY[nameIndex])
                                    .setContent(String.format(Locale.CHINA, "%.2f", (NUMBER_HUNDRED * score)) + "%")
                                    .setScore(score)
                                    .build();
                            list.add(commonResultBean);
                        }
                    }
                }
            }
            Collections.sort(list, (t1, t2) -> Float.compare(t2.getScore(), t1.getScore()));
            String time = (endTime - startTime) + MS;
            CommonResultBean commonResultBean = new CommonResultBean.Builder()
                    .setTitle(getString(R.string.inference_time))
                    .setContent(time)
                    .build();
            list.add(commonResultBean);
            defaultFragment.setDataList(list);
        }
    }

    private void initCustomData() {
        // run net.
        long startTime = System.currentTimeMillis();
        String result = customTrackingMobile.MindSpore_runnet(originClassificationBitmap, customLabelist.size());
        long endTime = System.currentTimeMillis();

        List<CommonResultBean> list = new ArrayList<>();
        if (!TextUtils.isEmpty(result)) {
            String[] singleResult = result.split(":");
            if (singleResult.length > NUMBER_ONE) {
                int nameIndex = Integer.parseInt(singleResult[NUMBER_ZERO]);
                float score = Float.parseFloat(singleResult[NUMBER_ONE]);
                CommonResultBean commonResultBean = new CommonResultBean.Builder()
                        .setTitle(customLabelist.get(nameIndex))
                        .setContent(String.format(Locale.CHINA, "%.2f", (NUMBER_HUNDRED * score)) + "%")
                        .setScore(score)
                        .build();
                list.add(commonResultBean);
            }
            String time = (endTime - startTime) + MS;
            list.add(new CommonResultBean.Builder()
                    .setTitle(getString(R.string.inference_time))
                    .setContent(time)
                    .build());
            customFragment.setDataList(list);
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
        customLabelist = bean.getLabel();
        MS_FILE = new File(JSON_FILE.getParent(), bean.getFile());
        if (MS_FILE.exists() && MS_FILE.getPath().toLowerCase().endsWith(MS)) {
            prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, true).apply();
            prefs.edit().putString(Constants.KEY_CLASSIFICATION_CUSTOM_PATH, JSON_FILE.getPath()).apply();
            modelTabList.add(bean.getTitle());
            addNewCustom();
        } else {
            prefs.edit().putBoolean(Constants.KEY_CLASSIFICATION_CUSTOM_STATE, false).apply();
            Toast.makeText(getActivity(), R.string.custom_ms_file_error, Toast.LENGTH_SHORT).show();
        }
    }

    public void addNewCustom() {
        ResultTabHelper.addTabView(getContext());
        customFragment = ResultChildFragment.newInstance(false);
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
        } else if (REQUEST_R_PERMISSIONS == requestCode) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (Environment.isExternalStorageManager()) {
                    openDocument();
                } else {
                    Toast.makeText(getContext(), R.string.app_save_permissions_failed, Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        modelTabList.clear();
    }

    private void unloadModel() {
        if (commonTrackingMobile != null) {
            commonTrackingMobile.unloadCustomModel();
        }
        if (customTrackingMobile != null) {
            customTrackingMobile.unloadCustomModel();
        }
    }

    private void onClick(View v) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (Environment.isExternalStorageManager()) {
                openDocument();
            } else {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + getContext().getPackageName()));
                startActivityForResult(intent, REQUEST_R_PERMISSIONS);
            }
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            openDocument();
        } else {
            openDocument();
        }
    }

    private void openDocument() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, REQUEST_FILE_PATH);
    }


    private static class MyHandler extends Handler {
        private final WeakReference<Fragment> fragmentWeakReference;

        public MyHandler(ClassificationFragment fragmentWeakReference) {
            this.fragmentWeakReference = new WeakReference<>(fragmentWeakReference);
        }

        @Override
        public void handleMessage(@NonNull Message msg) {
            super.handleMessage(msg);
            if (fragmentWeakReference.get() != null) {
                if (HANDLER_ONE == msg.what) {
                    ((ClassificationFragment) fragmentWeakReference.get()).initModel();
                }
            }
        }
    }
}