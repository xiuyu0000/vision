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

package com.mindspore.classificationlibrary.adapter;

import android.content.Context;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import com.mindspore.classificationlibrary.R;
import com.mindspore.classificationlibrary.bean.CommonResultBean;

import java.util.List;


public class ResultChildAdapter extends RecyclerView.Adapter<ResultChildAdapter.ViewHolder> {

    private final List<CommonResultBean> mValues;
    private final Context mContext;

    public ResultChildAdapter(Context mContext, List<CommonResultBean> items) {
        mValues = items;
        this.mContext = mContext;
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        return new ViewHolder(LayoutInflater.from(mContext)
                .inflate(R.layout.adapter_result_child, parent, false));
    }

    @Override
    public void onBindViewHolder(final ViewHolder holder, int position) {
        holder.mTextLeft.setText(mValues.get(position).getTitle());
        holder.mTextRight.setText(mValues.get(position).getContent());
        if (TextUtils.isEmpty(mValues.get(position).getPosition())) {
            holder.mTextMiddle.setVisibility(View.INVISIBLE);
        } else {
            holder.mTextMiddle.setVisibility(View.VISIBLE);
            holder.mTextMiddle.setText(mValues.get(position).getPosition());
        }
    }

    @Override
    public int getItemCount() {
        return mValues.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        public final TextView mTextLeft;
        public final TextView mTextRight;
        public final TextView mTextMiddle;

        public ViewHolder(View itemView) {
            super(itemView);
            mTextLeft = itemView.findViewById(R.id.item_left);
            mTextRight = itemView.findViewById(R.id.item_right);
            mTextMiddle = itemView.findViewById(R.id.item_middle);
        }
    }
}