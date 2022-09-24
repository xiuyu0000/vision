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

package com.mindspore.facelibrary.stickers;

import android.content.Context;

import com.mindspore.facelibrary.base.GLImageFilter;
import com.mindspore.facelibrary.stickers.bean.DynamicSticker;

import java.util.ArrayList;
import java.util.List;

/**
 * 贴纸滤镜基类
 */
public class DynamicStickerBaseFilter extends GLImageFilter {

    // 贴纸数据
    protected DynamicSticker mDynamicSticker;

    // 贴纸加载器列表
    protected List<DynamicStickerLoader> mStickerLoaderList;

    public DynamicStickerBaseFilter(Context context, DynamicSticker sticker,
                                    String vertexShader, String fragmentShader) {
        super(context, vertexShader, fragmentShader);
        mDynamicSticker = sticker;
        mStickerLoaderList = new ArrayList<>();
    }

}
