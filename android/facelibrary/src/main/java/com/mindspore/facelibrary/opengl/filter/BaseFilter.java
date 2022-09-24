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

package com.mindspore.facelibrary.opengl.filter;

import android.content.Context;

import com.mindspore.facelibrary.opengl.utils.BufferHelper;
import com.mindspore.facelibrary.opengl.utils.ShaderHelper;
import com.mindspore.facelibrary.opengl.utils.TextResourceReader;

import java.nio.FloatBuffer;

import static android.opengl.GLES20.*;


public class BaseFilter {

    private int mVertexSourceId;
    private int mFragmentSourceId;

    FloatBuffer mVertexBuffer;
    FloatBuffer mTextureBuffer;

    int mProgramId;
    int vPosition;
    int vCoord;
    int vMatrix;
    int vTexture;
    int mWidth;
    int mHeight;

    public BaseFilter(Context context, int mVertexSourceId, int mFragmentSourceId) {
        this.mVertexSourceId = mVertexSourceId;
        this.mFragmentSourceId = mFragmentSourceId;

        float[] VERTEX = {-1.0f, -1.0f,
                1.0f, -1.0f,
                -1.0f, 1.0f,
                1.0f, 1.0f,};
        mVertexBuffer = BufferHelper.getFloatBuffer(VERTEX);

        float[] TEXTURE = {0.0f, 1.0f,
                1.0f, 1.0f,
                0.0f, 0.0f,
                1.0f, 0.0f,};
        mTextureBuffer = BufferHelper.getFloatBuffer(TEXTURE);

        init(context);
        changeTextureData();
    }

    /**
     * 修改纹理坐标 textureData（有需求可以重写该方法）
     */
    protected void changeTextureData() {

    }

    private void init(Context context) {
        String vertexSource = TextResourceReader.readTextFileFromResource(context, mVertexSourceId);
        String fragmentSource = TextResourceReader.readTextFileFromResource(context, mFragmentSourceId);

        int vertexShaderId = ShaderHelper.compileVertexShader(vertexSource);
        int fragmentShaderId = ShaderHelper.compileFragmentShader(fragmentSource);

        mProgramId = ShaderHelper.linkProgram(vertexShaderId, fragmentShaderId);
        glDeleteShader(vertexShaderId);
        glDeleteShader(fragmentShaderId);

        vPosition = glGetAttribLocation(mProgramId, "vPosition");
        vCoord = glGetAttribLocation(mProgramId, "vCoord");
        vMatrix = glGetUniformLocation(mProgramId, "vMatrix");
        vTexture = glGetUniformLocation(mProgramId, "vTexture");
    }

    public void onReady(int width, int height) {
        mWidth = width;
        mHeight = height;
    }

    public int onDrawFrame(int textureId) {
        glViewport(0, 0, mWidth, mHeight);
        glUseProgram(mProgramId);

        mVertexBuffer.position(0);
        glVertexAttribPointer(vPosition, 2, GL_FLOAT, false, 0, mVertexBuffer);
        glEnableVertexAttribArray(vPosition);

        mTextureBuffer.position(0);
        glVertexAttribPointer(vCoord, 2, GL_FLOAT, false, 0, mTextureBuffer);
        glEnableVertexAttribArray(vCoord);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glUniform1i(vTexture, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindTexture(GL_TEXTURE_2D, 0);
        return textureId;
    }

    public void release() {
        glDeleteProgram(mProgramId);
    }
}
