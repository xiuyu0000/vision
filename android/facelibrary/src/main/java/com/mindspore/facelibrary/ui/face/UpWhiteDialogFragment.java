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

import android.app.Dialog;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.coordinatorlayout.widget.CoordinatorLayout;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;

import com.google.android.material.bottomsheet.BottomSheetBehavior;
import com.google.android.material.bottomsheet.BottomSheetDialog;
import com.google.android.material.bottomsheet.BottomSheetDialogFragment;
import com.mindspore.facelibrary.R;

import org.jetbrains.annotations.NotNull;

public class UpWhiteDialogFragment extends BottomSheetDialogFragment {



    public static UpWhiteDialogFragment getInstance() {
        // Required empty public constructor
        return new UpWhiteDialogFragment();
    }

    @NonNull
    @NotNull
    @Override
    public Dialog onCreateDialog(@Nullable Bundle savedInstanceState) {
        return new BottomSheetDialog(this.getContext());
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_up_white_dialog, container, false);
    }


    @Override
    public void onStart() {
        super.onStart();
        //??????dialog??????
        BottomSheetDialog dialog = (BottomSheetDialog) getDialog();
        //???windowsd??????????????????????????????????????????????????????
        dialog.getWindow().findViewById(R.id.design_bottom_sheet).setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        //??????diglog????????????
        FrameLayout bottomSheet = dialog.getDelegate().findViewById(R.id.design_bottom_sheet);
        if (bottomSheet != null) {
            //??????????????????LayoutParams??????
            CoordinatorLayout.LayoutParams layoutParams = (CoordinatorLayout.LayoutParams) bottomSheet.getLayoutParams();
            layoutParams.height = getPeekHeight();
            //?????????????????????????????????????????????????????????????????????
            bottomSheet.setLayoutParams(layoutParams);

            final BottomSheetBehavior<FrameLayout> behavior = BottomSheetBehavior.from(bottomSheet);
            //peekHeight????????????????????????
            behavior.setPeekHeight(getPeekHeight());
            // ?????????????????????
            behavior.setState(BottomSheetBehavior.STATE_EXPANDED);
//            ImageView mReBack = view.findViewById(R.id.re_back_img);
//            //????????????
//            mReBack.setOnClickListener(new View.OnClickListener() {
//                @Override
//                public void onClick(View view) {
//                    //????????????
//                    behavior.setState(BottomSheetBehavior.STATE_HIDDEN);
//                }
//            });
        }

    }

    /**
     * ???????????????????????????????????????????????????
     * ??????????????????????????????peekHeight
     *
     * @return height
     */
    protected int getPeekHeight() {
        int peekHeight = getResources().getDisplayMetrics().heightPixels;
        //????????????????????????????????????3/4
        return peekHeight / 3;
    }
}