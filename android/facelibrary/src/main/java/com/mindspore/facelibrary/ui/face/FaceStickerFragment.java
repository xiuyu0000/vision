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

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.mindspore.facelibrary.R;
import com.mindspore.facelibrary.resource.ResourceHelper;
import com.mindspore.facelibrary.resource.bean.ResourceData;
import com.mindspore.facelibrary.ui.adapter.FaceStickerItemAdapter;
import com.mindspore.facelibrary.ui.bean.FaceBeautyItemBean;

import java.util.ArrayList;
import java.util.List;

public class FaceStickerFragment extends Fragment {


    private List<ResourceData> listData = new ArrayList<>();
    private RecyclerView recyclerView;
    private FaceStickerItemAdapter adapter;
    public FaceStickerFragment() {
        // Required empty public constructor
    }


    public static FaceStickerFragment newInstance() {
        FaceStickerFragment fragment = new FaceStickerFragment();
        Bundle args = new Bundle();
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        listData= ResourceHelper.getResourceList();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view = inflater.inflate(R.layout.fragment_face_sticker, container, false);
        recyclerView = view.findViewById(R.id.recycleView);
        recyclerView.setLayoutManager(new GridLayoutManager(getContext(),5));

        adapter = new FaceStickerItemAdapter(getContext(), listData);
        adapter.setmOnClickListener(new FaceStickerItemAdapter.ListItemClickListener() {
            @Override
            public void onListItemClick(int position,ResourceData resource) {
                adapter.setSelectPosition(position);
                adapter.notifyDataSetChanged();
                if (mOnResourceChangeListener != null) {
                    mOnResourceChangeListener.onResourceChange(resource);
                }
            }
        });

        recyclerView.setAdapter(adapter);
        return view;
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