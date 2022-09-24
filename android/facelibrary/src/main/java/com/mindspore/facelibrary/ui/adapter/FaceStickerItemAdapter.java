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

package com.mindspore.facelibrary.ui.adapter;

import android.content.Context;
import android.net.Uri;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;


import com.mindspore.facelibrary.R;
import com.mindspore.facelibrary.resource.bean.ResourceData;
import com.mindspore.utilslibrary.ui.utils.BitmapUtils;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

public class FaceStickerItemAdapter extends RecyclerView.Adapter<FaceStickerItemAdapter.ViewHolder> {
    private final Context mContext;

    public void setSelectPosition(int selectPosition) {
        this.selectPosition = selectPosition;
    }

    private int selectPosition;

    public int getSelectPosition() {
        return selectPosition;
    }

    private List<ResourceData> itemList = new ArrayList<>();

    public FaceStickerItemAdapter(Context mContext, List<ResourceData> itemList) {
        this.mContext = mContext;
        this.itemList = itemList;
    }

    public ViewHolder onCreateViewHolder(@NonNull @NotNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(mContext)
                .inflate(R.layout.adapter_face_sticker_item, parent, false);
        ViewHolder viewHolder = new ViewHolder(view);

        return viewHolder;
    }

    private ListItemClickListener mOnClickListener;

    public void setmOnClickListener(ListItemClickListener mOnClickListener) {
        this.mOnClickListener = mOnClickListener;
    }

    @Override
    public void onBindViewHolder(@NonNull @NotNull ViewHolder holder, int position) {
        holder.bind();
    }


    @Override
    public int getItemCount() {
        return itemList.size();
    }


    public class ViewHolder extends RecyclerView.ViewHolder {
        ImageView iconView;
        TextView nameText;
        ImageView checkView;
        View view;

        public ViewHolder(@NonNull @NotNull View itemView) {
            super(itemView);
            view = itemView;
            iconView = itemView.findViewById(R.id.icon);
//            nameText = itemView.findViewById(R.id.icon_name);
//            checkView = itemView.findViewById(R.id.checked_view);
            itemView.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    if (mOnClickListener != null) {
                        mOnClickListener.onListItemClick(getAdapterPosition(),itemList.get(getAdapterPosition()));
                    }

                }
            });

        }

        public void bind() {
            ResourceData resource = itemList.get(getAdapterPosition());
            // 如果是asset下面的，则直接解码
            if (!TextUtils.isEmpty(resource.thumbPath) && resource.thumbPath.startsWith("assets://")) {
                iconView.setImageBitmap(BitmapUtils.getImageFromAssetsFile(mContext,
                        resource.thumbPath.substring("assets://".length())));
            } else {

            }


//            nameText.setText(itemList.get(getAdapterPosition()).getName());
//            iconView.setImageResource(itemList.get(getAdapterPosition()).getIconUncheck());

            if (getAdapterPosition() == getSelectPosition()) {
                view.setBackgroundResource(R.drawable.blue_rect_bg);
//                iconView.setImageResource(itemList.get(getAdapterPosition()).getIconChecked());
//                nameText.setTextColor(mContext.getResources().getColor(R.color.colorPrimary));
//                checkView.setVisibility(View.VISIBLE);
            } else {
                view.setBackgroundResource(0);
//                iconView.setImageResource(itemList.get(getAdapterPosition()).getIconUncheck());
//                nameText.setTextColor(mContext.getResources().getColor(R.color.gray));
//                checkView.setVisibility(View.GONE);
            }

        }
    }

    public interface ListItemClickListener {
        void onListItemClick(int position,ResourceData resource);
    }
}
