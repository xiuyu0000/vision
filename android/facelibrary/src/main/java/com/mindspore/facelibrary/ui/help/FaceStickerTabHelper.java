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

package com.mindspore.facelibrary.ui.help;

import android.content.Context;
import android.view.View;
import android.widget.TextView;

import com.google.android.material.tabs.TabLayout;
import com.mindspore.facelibrary.R;
import com.mindspore.utilslibrary.ui.view.MSTabEntity;
import com.mindspore.utilslibrary.ui.view.TabEntity;

import java.util.ArrayList;
import java.util.List;

public class FaceStickerTabHelper {

    private final Context mContext;
    private static TabLayout mTabLayout;
    private static List<String> tabList;

    public FaceStickerTabHelper(Context mContext, TabLayout mTabLayout, List<String> tabList) {
        this.mContext = mContext;
        FaceStickerTabHelper.mTabLayout = mTabLayout;
        FaceStickerTabHelper.tabList = tabList;
    }

    public void initTabLayout(BaseOnTabSelectedListener listener) {
        mTabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                if (tab.getCustomView() != null) {
                    TextView tabText = tab.getCustomView().findViewById(R.id.txtModel);
                    tabText.setTextColor(mContext.getResources().getColor(R.color.colorPrimary));
                }
                if (listener != null) {
                    listener.onTabSelected(tab);
                }
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {
                if (tab.getCustomView() != null) {
                    TextView tabText = tab.getCustomView().findViewById(R.id.txtModel);
                    tabText.setTextColor(mContext.getResources().getColor(R.color.white));
                }
                if (listener != null) {
                    listener.onTabUnselected(tab);
                }
            }

            @Override
            public void onTabReselected(TabLayout.Tab tab) {

            }
        });

        addTabView(mContext);
    }

    public static void addTabView(Context mContext) {
        mTabLayout.removeAllTabs();
        ArrayList<MSTabEntity> mTabEntities = getTabEntity();
        for (int i = 0; i < mTabEntities.size(); i++) {
            mTabLayout.addTab(mTabLayout.newTab().setCustomView(getTabView(mContext, mTabEntities.get(i))));
        }
    }

    private static ArrayList<MSTabEntity> getTabEntity() {
        ArrayList<MSTabEntity> mTabEntities = new ArrayList<>();
        for (int i = 0; i < tabList.size(); i++) {
            mTabEntities.add(new TabEntity(tabList.get(i)));
        }
        return mTabEntities;
    }


    private static View getTabView(Context context, MSTabEntity tabEntity) {
        View view = View.inflate(context, R.layout.tab_face_sticker, null);
        TextView tabText = view.findViewById(R.id.txtModel);
        tabText.setText(tabEntity.getMSTabTitle());
        tabText.setTextColor(context.getResources().getColor(R.color.white));
        return view;
    }


    public interface BaseOnTabSelectedListener {

        void onTabSelected(TabLayout.Tab tab);

        void onTabUnselected(TabLayout.Tab tab);
    }
}
