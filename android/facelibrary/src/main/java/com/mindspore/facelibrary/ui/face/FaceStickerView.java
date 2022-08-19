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

package com.mindspore.facelibrary.ui.face;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;

import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager.widget.ViewPager;

import com.google.android.material.tabs.TabLayout;
import com.mindspore.facelibrary.R;
import com.mindspore.facelibrary.resource.bean.ResourceData;
import com.mindspore.facelibrary.ui.help.FaceStickerTabHelper;
import com.mindspore.utilslibrary.ui.adapter.BasePagerAdapter;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class FaceStickerView extends LinearLayout {

    private static final String[] TITLE_STICKER = new String[]{"收藏", "热门"};
//    private static final String[] TITLE_STICKER = new String[]{"收藏", "热门", "最新", "妆容", "可爱", "滤镜", "轻熟", "创意", "造型", "抖音", "漫画", "搞怪"};

    private final static int NUMBER_HUNDRED = 100;
    private final static int NUMBER_ZERO = 0;
    private final static int NUMBER_ONE = 1;
    private final static int NUMBER_THREE = 3;
    private final static float THRESHOLD = 0.5f;
    private final static int HANDLER_ONE = 1;

    private Context mContext;
    private View mView;
    private ViewPager viewPager;

    private List<String> modelTabList = new ArrayList<>();
    private List<Fragment> fragments;
    private BasePagerAdapter adapter;
    private FragmentActivity activity;

    public FaceStickerView(Context context) {
        this(context, null);
    }

    public FaceStickerView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public FaceStickerView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        this.mContext = context;
        activity = (FragmentActivity) mContext;

        mView = LayoutInflater.from(mContext).inflate(R.layout.layout_face_sticker, this);
        initView(mView);
    }

    public void initView(View view) {
        TabLayout mTabLayout = view.findViewById(R.id.tab_layout);
        viewPager = view.findViewById(R.id.viewPager);
        Collections.addAll(modelTabList, TITLE_STICKER);
        initViewPager(mTabLayout);
        initTabLayout(mTabLayout);
    }

    private void initViewPager(TabLayout mTabLayout) {
        fragments = new ArrayList<>();
        for (int i = 0; i < modelTabList.size(); i++) {
            FaceStickerFragment fragment = FaceStickerFragment.newInstance();
            fragment.addOnChangeResourceListener((data) -> {
                if (mOnResourceChangeListener !=null){
                    mOnResourceChangeListener.onResourceChange(data);
                }
            });
            fragments.add(fragment);
        }

        adapter = new BasePagerAdapter(activity.getSupportFragmentManager(), fragments);
        viewPager.setAdapter(adapter);
        viewPager.addOnPageChangeListener(new TabLayout.TabLayoutOnPageChangeListener(mTabLayout));
        viewPager.setOffscreenPageLimit(NUMBER_THREE);
        viewPager.setCurrentItem(NUMBER_ZERO);
    }


    private void initTabLayout(TabLayout mTabLayout) {
        FaceStickerTabHelper resultTabHelper = new FaceStickerTabHelper(mContext, mTabLayout, modelTabList);
        resultTabHelper.initTabLayout(new FaceStickerTabHelper.BaseOnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                viewPager.setCurrentItem(tab.getPosition(), true);
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {
            }

        });
    }


    /**
     * 资源切换监听器
     */
    public interface OnResourceChangeListener {

        /** 切换资源 */
        void onResourceChange(ResourceData data);
    }

    /**
     * 添加资源切换监听器
     * @param listener
     */
    public void addOnChangeResourceListener(OnResourceChangeListener listener) {
        mOnResourceChangeListener = listener;
    }

    private OnResourceChangeListener mOnResourceChangeListener;
}
