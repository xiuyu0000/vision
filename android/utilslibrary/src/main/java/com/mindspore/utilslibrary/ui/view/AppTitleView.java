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

package com.mindspore.utilslibrary.ui.view;

import android.app.Activity;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;

import com.mindspore.utilslibrary.R;


public class AppTitleView extends LinearLayout {

    private final String TAG = AppTitleView.class.getSimpleName();
    private Activity activity;
    private final CharSequence nameTitle;
    private final Drawable iconLeftBlack;
    private final Drawable iconRightOne;
    private final Drawable iconRightTwo;
    private final Drawable iconRightThree;
    private ImageView titleBack, titleRightOne, titleRightTwo, titleRightThree;
    private TextView titleName;
    private final boolean titleLineView;

    public void setNavigationOnClickListener(OnClickListener onClickListener) {
        titleBack.setOnClickListener(onClickListener);
    }

    public void setTitleRightOne(OnClickListener onClickListener) {
        titleRightOne.setOnClickListener(onClickListener);
    }

    public void setTitleRightTwo(OnClickListener onClickListener) {
        titleRightTwo.setOnClickListener(onClickListener);
    }

    public void setTitleRightThree(OnClickListener onClickListener) {
        titleRightThree.setOnClickListener(onClickListener);
    }

    public void setTitleText(int tileTextId) {
        titleName.setText((int) tileTextId);
    }

    public void setTitleText(String tileText) {
        titleName.setText(tileText);
    }

    public AppTitleView(Context context) {
        this(context, null);
    }

    public AppTitleView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public AppTitleView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs);
        if (context instanceof Activity) {
            this.activity = (Activity) context;
        }
        TypedArray ToolbarTile = context.obtainStyledAttributes(attrs, R.styleable.AppTitleView);
        nameTitle = ToolbarTile.getText(R.styleable.AppTitleView_nameTitle);
        iconLeftBlack = ToolbarTile.getDrawable(R.styleable.AppTitleView_iconLeftBlack);
        iconRightOne = ToolbarTile.getDrawable(R.styleable.AppTitleView_iconRightOne);
        iconRightTwo = ToolbarTile.getDrawable(R.styleable.AppTitleView_iconRightTwo);
        iconRightThree = ToolbarTile.getDrawable(R.styleable.AppTitleView_iconRightThree);
        titleLineView = ToolbarTile.getBoolean(R.styleable.AppTitleView_titleLine, true);
        ToolbarTile.recycle();
    }

    @Override
    protected void onFinishInflate() {
        super.onFinishInflate();
        LinearLayout layout = (LinearLayout) LayoutInflater.from(getContext())
                .inflate(R.layout.title_layout_utils, this, true);

        titleName = layout.findViewById(R.id.title_name);
        titleBack = layout.findViewById(R.id.title_black);
        titleRightOne = layout.findViewById(R.id.icon_right_one);
        titleRightTwo = layout.findViewById(R.id.icon_right_two);
        titleRightThree = layout.findViewById(R.id.icon_right_three);
        View titleLine = layout.findViewById(R.id.title_line);

        titleName.setText(nameTitle);
        titleBack.setImageDrawable(iconLeftBlack);
        titleRightOne.setImageDrawable(iconRightOne);
        titleRightTwo.setImageDrawable(iconRightTwo);
        titleRightThree.setImageDrawable(iconRightThree);
        if (titleLineView) {
            titleLine.setVisibility(VISIBLE);
        } else {
            titleLine.setVisibility(GONE);
        }
        titleBack.setOnClickListener(view -> {
            if (activity != null) {
                activity.finish();
            } else {
                Log.e(TAG, "activity null");
            }
        });

    }

    public void setRightOneImageOnClickListener(OnClickListener onClickListener) {
        titleRightOne.setOnClickListener(onClickListener);
    }

    public void setRightTwoImageOnClickListener(OnClickListener onClickListener) {
        titleRightTwo.setOnClickListener(onClickListener);
    }

    public void setRightThreeImageOnClickListener(OnClickListener onClickListener) {
        titleRightThree.setOnClickListener(onClickListener);
    }

    public void setTitleTextName(int titleTextId) {
        titleName.setText((int) titleTextId);
    }

    public void setTitleTextName(String titleText) {
        titleName.setText(titleText);
    }

    public void setTitleRightOneVisibility(boolean isVisible) {
        if (isVisible) {
            titleRightOne.setVisibility(VISIBLE);
        } else {
            titleRightOne.setVisibility(GONE);
        }
    }

    public void setTitleRightTwoVisibility(boolean isVisible) {
        if (isVisible) {
            titleRightTwo.setVisibility(VISIBLE);
        } else {
            titleRightTwo.setVisibility(GONE);
        }
    }

    public void setTitleRightThreeVisibility(boolean isVisible) {
        if (isVisible) {
            titleRightThree.setVisibility(VISIBLE);
        } else {
            titleRightThree.setVisibility(GONE);
        }
    }

    public void setTitleOneImage(int drawable) {
        titleRightOne.setImageResource(drawable);
    }

    public void setTitleTwoImage(int drawable) {
        titleRightTwo.setImageResource(drawable);
    }

    public void setTitleThreeImage(int drawable) {
        titleRightThree.setImageResource(drawable);
    }
}
