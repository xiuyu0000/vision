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

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;

import com.mindspore.utilslibrary.R;


public class UpImageDownTextView extends LinearLayout {

    private final Drawable imageViewUp;
    private final CharSequence textViewDown;

    private final Drawable imageViewUpChecked;
    private final int textDownColorChecked, textDownColorUnCheck;

    private ImageView iconUp;
    private TextView textDown;

    public UpImageDownTextView(Context context) {
        this(context, null);
    }

    public UpImageDownTextView(Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public UpImageDownTextView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        TypedArray typedArray = context.obtainStyledAttributes(attrs, R.styleable.UpImageDownTextView);
        imageViewUp = typedArray.getDrawable(R.styleable.UpImageDownTextView_UpImageView);
        textViewDown = typedArray.getText(R.styleable.UpImageDownTextView_DownTextView);
        imageViewUpChecked = typedArray.getDrawable(R.styleable.UpImageDownTextView_UpImageViewChecked);
        textDownColorChecked = typedArray.getColor(
                R.styleable.UpImageDownTextView_DownTextViewCheckedColor, Color.BLACK);
        textDownColorUnCheck = typedArray.getColor(
                R.styleable.UpImageDownTextView_DownTextViewUnCheckColor, Color.BLACK);
        typedArray.recycle();
    }

    @Override
    protected void onFinishInflate() {
        super.onFinishInflate();
        LinearLayout layout = (LinearLayout) LayoutInflater.from(getContext())
                .inflate(R.layout.view_up_image_down_text, this, true);
        iconUp = layout.findViewById(R.id.upImageView);
        textDown = layout.findViewById(R.id.downTextView);
        iconUp.setImageDrawable(imageViewUp);
        textDown.setText(textViewDown);
        setUnChecked();
    }

    public void setTextViewDown(String textStr) {
        textDown.setText(textStr);
    }

    public void setChecked() {
        iconUp.setImageDrawable(imageViewUpChecked);
        textDown.setTextColor(textDownColorChecked);
    }

    public void setUnChecked() {
        iconUp.setImageDrawable(imageViewUp);
        textDown.setTextColor(textDownColorUnCheck);
    }

    public void setCheck(boolean check) {
        if (check) {
            setChecked();
        } else {
            setUnChecked();
        }
    }
}
