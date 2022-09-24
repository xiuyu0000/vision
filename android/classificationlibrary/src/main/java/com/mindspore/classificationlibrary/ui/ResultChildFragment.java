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

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.RecyclerView;

import com.mindspore.classificationlibrary.R;
import com.mindspore.classificationlibrary.adapter.ResultChildAdapter;
import com.mindspore.classificationlibrary.bean.CommonResultBean;

import java.util.ArrayList;
import java.util.List;

public class ResultChildFragment extends Fragment {

    private static final String ARG_SHOW_OSITION = "param1";
    private boolean isShowPosition;

    private RecyclerView recyclerView;
    private TextView positionText;
    private final List<CommonResultBean> dataList = new ArrayList<>();
    private ResultChildAdapter adapter;

    public static ResultChildFragment newInstance(boolean isShowPosition) {
        ResultChildFragment fragment = new ResultChildFragment();
        Bundle args = new Bundle();
        args.putBoolean(ARG_SHOW_OSITION, isShowPosition);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            isShowPosition = getArguments().getBoolean(ARG_SHOW_OSITION);
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_result_child_list, container, false);
        recyclerView = (RecyclerView) view.findViewById(R.id.list);
        positionText = (TextView) view.findViewById(R.id.title_middle);
        if (isShowPosition) {
            positionText.setVisibility(View.VISIBLE);
        } else {
            positionText.setVisibility(View.INVISIBLE);
        }
        return view;
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        adapter = new ResultChildAdapter(getContext(), dataList);
        recyclerView.setAdapter(adapter);
    }

    public void setDataList(List<CommonResultBean> dataList) {
        this.dataList.clear();
        this.dataList.addAll(dataList);
        adapter.notifyDataSetChanged();
    }
}