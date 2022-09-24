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

package com.mindspore.facelibrary.color;

import android.content.Context;
import android.text.TextUtils;

import com.mindspore.facelibrary.base.GLImageFilter;
import com.mindspore.facelibrary.color.bean.DynamicColorData;
import com.mindspore.facelibrary.utils.OpenGLUtils;


/**
 * 颜色滤镜基类
 */
public class DynamicColorBaseFilter extends GLImageFilter {

    // 颜色滤镜参数
    protected DynamicColorData mDynamicColorData;
    protected DynamicColorLoader mDynamicColorLoader;

    public DynamicColorBaseFilter(Context context, DynamicColorData dynamicColorData, String unzipPath) {
        super(context, (dynamicColorData == null || TextUtils.isEmpty(dynamicColorData.vertexShader)) ? VERTEX_SHADER
                        : getShaderString(context, unzipPath, dynamicColorData.vertexShader),
                (dynamicColorData == null || TextUtils.isEmpty(dynamicColorData.fragmentShader)) ? FRAGMENT_SHADER
                        : getShaderString(context, unzipPath, dynamicColorData.fragmentShader));
        mDynamicColorData = dynamicColorData;
        mDynamicColorLoader = new DynamicColorLoader(this, mDynamicColorData, unzipPath);
        mDynamicColorLoader.onBindUniformHandle(mProgramHandle);
    }

    @Override
    public void onInputSizeChanged(int width, int height) {
        super.onInputSizeChanged(width, height);
        if (mDynamicColorLoader != null) {
            mDynamicColorLoader.onInputSizeChange(width, height);
        }
    }

    @Override
    public void onDrawFrameBegin() {
        super.onDrawFrameBegin();
        if (mDynamicColorLoader != null) {
            mDynamicColorLoader.onDrawFrameBegin();
        }
    }

    @Override
    public void release() {
        super.release();
        if (mDynamicColorLoader != null) {
            mDynamicColorLoader.release();
        }
    }

    /**
     * 设置强度，调节滤镜的轻重程度
     * @param strength
     */
    public void setStrength(float strength) {
        if (mDynamicColorLoader != null) {
            mDynamicColorLoader.setStrength(strength);
        }
    }

    /**
     * 根据解压路径和shader名称读取shader的字符串内容
     * @param unzipPath
     * @param shaderName
     * @return
     */
    protected static String getShaderString(Context context, String unzipPath, String shaderName) {
        if (TextUtils.isEmpty(unzipPath) || TextUtils.isEmpty(shaderName)) {
            throw new IllegalArgumentException("shader is empty!");
        }
        String path = unzipPath + "/" + shaderName;
        if (path.startsWith("assets://")) {
            return OpenGLUtils.getShaderFromAssets(context, path.substring("assets://".length()));
        } else if (path.startsWith("file://")) {
            return OpenGLUtils.getShaderFromFile(path.substring("file://".length()));
        }
        return OpenGLUtils.getShaderFromFile(path);
    }

}
