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

package com.mindspore.facelibrary.base;

import android.content.Context;
import android.graphics.PointF;
import android.opengl.GLES20;

import com.mindspore.facelibrary.utils.OpenGLUtils;


/**
 * 暗角(虚光照)滤镜
 */
public class GLImageVignetteFilter extends GLImageFilter {

    private int mVignetteCenterHandle;
    private int mVignetteColorHandle;
    private int mVignetteStartHandle;
    private int mVignetteEndHandle;

    private PointF mVignetteCenter;

    private float[] mVignetteColor;

    private float mVignetteStart;

    private float mVignetteEnd;

    public GLImageVignetteFilter(Context context) {
        this(context, VERTEX_SHADER, OpenGLUtils.getShaderFromAssets(context,
                "shader/base/fragment_vignette.glsl"));
    }

    public GLImageVignetteFilter(Context context, String vertexShader, String fragmentShader) {
        super(context, vertexShader, fragmentShader);
    }

    @Override
    public void initProgramHandle() {
        super.initProgramHandle();
        mVignetteCenterHandle = GLES20.glGetUniformLocation(mProgramHandle, "vignetteCenter");
        mVignetteColorHandle = GLES20.glGetUniformLocation(mProgramHandle, "vignetteColor");
        mVignetteStartHandle = GLES20.glGetUniformLocation(mProgramHandle, "vignetteStart");
        mVignetteEndHandle = GLES20.glGetUniformLocation(mProgramHandle, "vignetteEnd");
        setVignetteCenter(new PointF(0.5f, 0.5f));
        setVignetteColor(new float[] {0.0f, 0.0f, 0.0f});
        setVignetteStart(0.3f);
        setVignetteEnd(0.75f);
    }

    /**
     * 设置暗角中心，默认为纹理中心(0.5f, 0.5f)
     * @param vignetteCenter
     */
    public void setVignetteCenter(final PointF vignetteCenter) {
        mVignetteCenter = vignetteCenter;
        setPoint(mVignetteCenterHandle, mVignetteCenter);
    }

    /**
     * 设置暗角的颜色
     * @param vignetteColor
     */
    public void setVignetteColor(final float[] vignetteColor) {
        mVignetteColor = vignetteColor;
        setFloatVec3(mVignetteColorHandle, mVignetteColor);
    }

    /**
     * 设置暗角开始位置
     * @param vignetteStart 0.0f ~ 1.0f
     */
    public void setVignetteStart(final float vignetteStart) {
        mVignetteStart = vignetteStart;
        setFloat(mVignetteStartHandle, mVignetteStart);
    }

    /**
     * 设置暗角结束位置
     * @param vignetteEnd 0.0f ~ 1.0f
     */
    public void setVignetteEnd(final float vignetteEnd) {
        mVignetteEnd = vignetteEnd;
        setFloat(mVignetteEndHandle, mVignetteEnd);
    }
}
