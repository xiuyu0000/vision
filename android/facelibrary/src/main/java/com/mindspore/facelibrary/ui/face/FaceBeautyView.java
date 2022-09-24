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

package com.mindspore.facelibrary.ui.face;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.mindspore.facelibrary.R;
import com.mindspore.facelibrary.camera.CameraParam;
import com.mindspore.facelibrary.event.FaceCloseEvent;
import com.mindspore.facelibrary.ui.adapter.FaceBeautyItemAdapter;
import com.mindspore.facelibrary.ui.bean.FaceBeautyItemBean;
import com.warkiz.widget.IndicatorSeekBar;
import com.warkiz.widget.OnSeekChangeListener;
import com.warkiz.widget.SeekParams;

import org.greenrobot.eventbus.EventBus;

import java.util.ArrayList;
import java.util.List;

public class FaceBeautyView extends LinearLayout implements View.OnClickListener {

    private Context mContext;
    private View mView;
    private RecyclerView horRecyclerView;

    // 相机参数
    private CameraParam mCameraParam;

    public FaceBeautyView(Context context) {
        this(context, null);
    }

    public FaceBeautyView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public FaceBeautyView(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        this.mContext = context;
        mCameraParam = CameraParam.getInstance();
        mView = LayoutInflater.from(mContext).inflate(R.layout.layout_face_beauty, this);
        init(mView);
    }

    private TextView seekText;
    private Button compareImage;
    private ImageView closeImage;
    private IndicatorSeekBar seekBar;
    private TextView btnBeauty, btnBody, btnMakeup;
    private TextView btnReset;
    private FaceBeautyItemAdapter faceBeautyItemAdapter;

    private void init(View view) {
        seekText = view.findViewById(R.id.seek_text);
        compareImage = view.findViewById(R.id.compareImage);
        seekBar = view.findViewById(R.id.seekBar);

        btnBeauty = view.findViewById(R.id.btn_beauty);
        btnBody = view.findViewById(R.id.btn_body);
        btnMakeup = view.findViewById(R.id.btn_makeup);
        closeImage = view.findViewById(R.id.close_btn);
        btnReset = view.findViewById(R.id.reset);

        btnBeauty.setOnClickListener(this);
        btnBody.setOnClickListener(this);
        btnMakeup.setOnClickListener(this);
        closeImage.setOnClickListener(this);
        btnReset.setOnClickListener(this);

        seekBar.setOnSeekChangeListener(new OnSeekChangeListener() {
            @Override
            public void onSeeking(SeekParams seekParams) {
                if (seekParams.fromUser) {
                    if (beautyStateListener != null){
                        beautyStateListener.beautyChoseState(faceBeautyItemAdapter.getSelectPosition(),seekParams.progress);
                    }
                }
            }

            @Override
            public void onStartTrackingTouch(IndicatorSeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(IndicatorSeekBar seekBar) {

            }
        });
        compareImage.setOnTouchListener((v, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    if (mCompareEffectListener != null) {
                        mCompareEffectListener.onCompareEffect(true);
                    }
                    compareImage.setBackgroundResource(R.drawable.face_compare_press);
                    break;

                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    if (mCompareEffectListener != null) {
                        mCompareEffectListener.onCompareEffect(false);
                    }
                    compareImage.setBackgroundResource(R.drawable.face_compare);
                    break;
            }
            return true;
        });

        List<FaceBeautyItemBean> itemList = new ArrayList<>();
        itemList.add(new FaceBeautyItemBean(1, R.drawable.panel_ic_classic_b, R.drawable.panel_ic_classic_p, "磨皮"));
        itemList.add(new FaceBeautyItemBean(2, R.drawable.panel_ic_long_b, R.drawable.panel_ic_long_p, "净肤"));
        itemList.add(new FaceBeautyItemBean(3, R.drawable.panel_ic_natural_b, R.drawable.panel_ic_natural_p, "清晰"));
        itemList.add(new FaceBeautyItemBean(4, R.drawable.panel_ic_round_b, R.drawable.panel_ic_round_p, "美白"));
        itemList.add(new FaceBeautyItemBean(5, R.drawable.panel_ic_classic_b, R.drawable.panel_ic_classic_p, "经典女神"));
        itemList.add(new FaceBeautyItemBean(6, R.drawable.panel_ic_long_b, R.drawable.panel_ic_long_p, "长脸专属"));
        itemList.add(new FaceBeautyItemBean(7, R.drawable.panel_ic_natural_b, R.drawable.panel_ic_natural_p, "自然脸"));
        itemList.add(new FaceBeautyItemBean(8, R.drawable.panel_ic_round_b, R.drawable.panel_ic_round_p, "圆脸专属"));

        seekText.setText(itemList.get(0).getName());

        horRecyclerView = view.findViewById(R.id.horRecyclerView);
        horRecyclerView.setLayoutManager(new LinearLayoutManager(mContext, LinearLayoutManager.HORIZONTAL, false));
        faceBeautyItemAdapter = new FaceBeautyItemAdapter(mContext, itemList);
        faceBeautyItemAdapter.setmOnClickListener(position -> {
            seekText.setText(itemList.get(position).getName());
            faceBeautyItemAdapter.setSelectPosition(position);
            faceBeautyItemAdapter.notifyDataSetChanged();
        });
        horRecyclerView.setAdapter(faceBeautyItemAdapter);

    }


    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.close_btn) {
            FaceCloseEvent faceCloseEvent = new FaceCloseEvent();
            EventBus.getDefault().post(faceCloseEvent);
        } else if (v.getId() == R.id.btn_beauty) {
            btnBeauty.setTextColor(getResources().getColor(R.color.colorPrimary));
            btnBody.setTextColor(getResources().getColor(R.color.gray_face_text));
            btnMakeup.setTextColor(getResources().getColor(R.color.gray_face_text));
        } else if (v.getId() == R.id.btn_body) {
            btnBeauty.setTextColor(getResources().getColor(R.color.gray_face_text));
            btnBody.setTextColor(getResources().getColor(R.color.colorPrimary));
            btnMakeup.setTextColor(getResources().getColor(R.color.gray_face_text));
        } else if (v.getId() == R.id.btn_makeup) {
            btnBeauty.setTextColor(getResources().getColor(R.color.gray_face_text));
            btnBody.setTextColor(getResources().getColor(R.color.gray_face_text));
            btnMakeup.setTextColor(getResources().getColor(R.color.colorPrimary));
        } else if (v.getId() == R.id.reset) {
            mCameraParam.beauty.reset();
        }
    }


    /**
     * 比较监听器
     */
    public interface OnCompareEffectListener {
        void onCompareEffect(boolean compare);
    }

    /**
     * 添加比较回调监听
     *
     * @param listener
     */
    public void addOnCompareEffectListener(OnCompareEffectListener listener) {
        mCompareEffectListener = listener;
    }

    private OnCompareEffectListener mCompareEffectListener;



    public interface BeautyStateListener {
        void beautyChoseState(int state, int progress);
    }

    private BeautyStateListener beautyStateListener;

    public FaceBeautyView setBeautyStateListener(BeautyStateListener beautyStateListener) {
        this.beautyStateListener = beautyStateListener;
        return this;
    }

}