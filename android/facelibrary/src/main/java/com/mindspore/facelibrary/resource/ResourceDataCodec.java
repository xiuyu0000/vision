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

package com.mindspore.facelibrary.resource;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Pair;

/**
 * 数据解码器
 */
public class ResourceDataCodec extends ResourceCodec {

    public ResourceDataCodec(String indexPath, String dataPath) {
        super(indexPath, dataPath);
    }

    /**
     * 根据文件名加载bitmap图片
     * @param name
     * @return
     */
    public Bitmap loadBitmap(String name) {
        Pair pair = mIndexMap.get(name);
        if (pair == null) {
            return null;
        }
        return BitmapFactory.decodeByteArray(mDataBuffer.array(),
                mDataBuffer.arrayOffset() + (Integer) pair.first, (Integer) pair.second);
    }

    /**
     * 获取文件缓冲
     * @return
     */
    public byte[] getBufferArray() {
        return mDataBuffer.array();
    }

    /**
     * 获取资源描述对象
     * @param path
     * @return
     */
    public Pair<Integer, Integer> getResourcePair(String path) {
        Pair pair = (Pair) mIndexMap.get(path);
        if (pair == null) {
            return null;
        }
        return new Pair<>(((Integer) pair.first) + mDataBuffer.arrayOffset(),
                (Integer) pair.second);
    }
}
