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

package com.mindspore.facelibrary.base;

import android.content.Context;
import android.graphics.PointF;
import android.opengl.GLES30;
import android.text.TextUtils;


import com.mindspore.facelibrary.utils.OpenGLUtils;
import com.mindspore.facelibrary.utils.TextureRotationUtils;

import java.nio.FloatBuffer;
import java.util.LinkedList;



public class GLImageFilter {

    protected static final String VERTEX_SHADER = "" +
            "attribute vec4 aPosition;                                  \n" +
            "attribute vec4 aTextureCoord;                              \n" +
            "varying vec2 textureCoordinate;                            \n" +
            "void main() {                                              \n" +
            "    gl_Position = aPosition;                               \n" +
            "    textureCoordinate = aTextureCoord.xy;                  \n" +
            "}                                                          \n";

    protected static final String FRAGMENT_SHADER = "" +
            "precision mediump float;                                   \n" +
            "varying vec2 textureCoordinate;                            \n" +
            "uniform sampler2D inputTexture;                                \n" +
            "void main() {                                              \n" +
            "    gl_FragColor = texture2D(inputTexture, textureCoordinate); \n" +
            "}                                                          \n";


    protected String TAG = getClass().getSimpleName();

    protected Context mContext;

    private final LinkedList<Runnable> mRunOnDraw;

    // ???????????????
    protected String mVertexShader;
    protected String mFragmentShader;

    // ?????????????????????
    protected boolean mIsInitialized;
    // ?????????????????????????????????
    protected boolean mFilterEnable = true;

    // ?????????????????????????????????
    protected int mCoordsPerVertex = TextureRotationUtils.CoordsPerVertex;
    // ??????????????????
    protected int mVertexCount = TextureRotationUtils.CubeVertices.length / mCoordsPerVertex;

    // ??????
    protected int mProgramHandle;
    protected int mPositionHandle;
    protected int mTextureCoordinateHandle;
    protected int mInputTextureHandle;

    // ?????????Image?????????
    protected int mImageWidth;
    protected int mImageHeight;

    // ?????????????????????
    protected int mDisplayWidth;
    protected int mDisplayHeight;

    // FBO???????????????????????????????????????????????????
    protected int mFrameWidth = -1;
    protected int mFrameHeight = -1;

    // FBO
    protected int[] mFrameBuffers;
    protected int[] mFrameBufferTextures;

    public GLImageFilter(Context context) {
        this(context, VERTEX_SHADER, FRAGMENT_SHADER);
    }

    public GLImageFilter(Context context, String vertexShader, String fragmentShader) {
        mContext = context;
        mRunOnDraw = new LinkedList<>();
        // ??????shader??????
        mVertexShader = vertexShader;
        mFragmentShader = fragmentShader;
        // ?????????????????????
        initProgramHandle();
    }

    /**
     * ?????????????????????
     */
    public void initProgramHandle() {
        // ?????????shader????????????????????????????????????????????????
        if (!TextUtils.isEmpty(mVertexShader) && !TextUtils.isEmpty(mFragmentShader)) {
            mProgramHandle = OpenGLUtils.createProgram(mVertexShader, mFragmentShader);
            mPositionHandle = GLES30.glGetAttribLocation(mProgramHandle, "aPosition");
            mTextureCoordinateHandle = GLES30.glGetAttribLocation(mProgramHandle, "aTextureCoord");
            mInputTextureHandle = GLES30.glGetUniformLocation(mProgramHandle, "inputTexture");
            mIsInitialized = true;
        } else {
            mPositionHandle = OpenGLUtils.GL_NOT_INIT;
            mTextureCoordinateHandle = OpenGLUtils.GL_NOT_INIT;
            mInputTextureHandle = OpenGLUtils.GL_NOT_TEXTURE;
            mIsInitialized = false;
        }
    }

    /**
     * Surface?????????????????????
     * @param width
     * @param height
     */
    public void onInputSizeChanged(int width, int height) {
        mImageWidth = width;
        mImageHeight = height;
    }

    /**
     *  ?????????????????????????????????
     * @param width
     * @param height
     */
    public void onDisplaySizeChanged(int width, int height) {
        mDisplayWidth = width;
        mDisplayHeight = height;
    }

    /**
     * ??????Frame
     * @param textureId
     * @param vertexBuffer
     * @param textureBuffer
     */
    public boolean drawFrame(int textureId, FloatBuffer vertexBuffer,
                          FloatBuffer textureBuffer) {
        // ????????????????????????????????????????????????????????????????????????
        if (!mIsInitialized || textureId == OpenGLUtils.GL_NOT_INIT || !mFilterEnable) {
            return false;
        }

        // ??????????????????
        GLES30.glViewport(0, 0, mDisplayWidth, mDisplayHeight);
        GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);

        // ???????????????program
        GLES30.glUseProgram(mProgramHandle);
        // ??????????????????
        runPendingOnDrawTasks();

        // ????????????
        onDrawTexture(textureId, vertexBuffer, textureBuffer);

        return true;
    }

    /**
     * ?????????FBO
     * @param textureId
     * @param vertexBuffer
     * @param textureBuffer
     * @return FBO?????????Texture
     */
    public int drawFrameBuffer(int textureId, FloatBuffer vertexBuffer, FloatBuffer textureBuffer) {
        // ??????FBO???????????????????????????????????????????????????????????????????????????
        if (textureId == OpenGLUtils.GL_NOT_TEXTURE || mFrameBuffers == null
                || !mIsInitialized || !mFilterEnable) {
            return textureId;
        }

        // ??????FBO
        bindFrameBuffer();

        // ????????????
        onDrawTexture(textureId, vertexBuffer, textureBuffer);

        // ??????FBO
        return unBindFrameBuffer();
    }

    public void bindFrameBuffer() {
        // ??????FBO
        GLES30.glViewport(0, 0, mFrameWidth, mFrameHeight);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, mFrameBuffers[0]);
        // ???????????????program
        GLES30.glUseProgram(mProgramHandle);
        // ????????????????????????????????????glUseProgram?????????????????????????????????????????????
        runPendingOnDrawTasks();
    }

    public int unBindFrameBuffer() {
        GLES30.glUseProgram(0);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        return mFrameBufferTextures[0];
    }

    public int drawFrameBufferClear(int textureId, FloatBuffer vertexBuffer, FloatBuffer textureBuffer) {
        // ??????FBO???????????????????????????????????????????????????????????????????????????
        if (textureId == OpenGLUtils.GL_NOT_TEXTURE || mFrameBuffers == null
                || !mIsInitialized || !mFilterEnable) {
            return textureId;
        }

        // ??????FBO
        GLES30.glViewport(0, 0, mFrameWidth, mFrameHeight);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, mFrameBuffers[0]);
        GLES30.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
        // ???????????????program
        GLES30.glUseProgram(mProgramHandle);
        // ????????????????????????????????????glUseProgram?????????????????????????????????????????????
        runPendingOnDrawTasks();

        // ????????????
        onDrawTexture(textureId, vertexBuffer, textureBuffer);

        // ??????FBO
        return unBindFrameBuffer();
    }

    /**
     * ??????
     * @param textureId
     * @param vertexBuffer
     * @param textureBuffer
     */
    public void onDrawTexture(int textureId, FloatBuffer vertexBuffer, FloatBuffer textureBuffer) {
        // ????????????????????????
        vertexBuffer.position(0);
        GLES30.glVertexAttribPointer(mPositionHandle, mCoordsPerVertex,
                GLES30.GL_FLOAT, false, 0, vertexBuffer);
        GLES30.glEnableVertexAttribArray(mPositionHandle);
        // ????????????????????????
        textureBuffer.position(0);
        GLES30.glVertexAttribPointer(mTextureCoordinateHandle, 2,
                GLES30.GL_FLOAT, false, 0, textureBuffer);
        GLES30.glEnableVertexAttribArray(mTextureCoordinateHandle);
        // ????????????
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0);
        GLES30.glBindTexture(getTextureType(), textureId);
        GLES30.glUniform1i(mInputTextureHandle, 0);
        onDrawFrameBegin();
        onDrawFrame();
        onDrawFrameAfter();
        // ??????
        GLES30.glDisableVertexAttribArray(mPositionHandle);
        GLES30.glDisableVertexAttribArray(mTextureCoordinateHandle);
        GLES30.glBindTexture(getTextureType(), 0);

    }

    /**
     * ??????glDrawArrays/glDrawElements?????????????????????????????????
     */
    public void onDrawFrameBegin() {

    }

    /**
     * ????????????
     */
    protected void onDrawFrame() {
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, mVertexCount);
    }

    /**
     * glDrawArrays/glDrawElements???????????????????????????????????????
     */
    public void onDrawFrameAfter() {

    }

    protected void onUnbindTextureValue() {

    }

    /**
     * ??????Texture??????
     * GLES30.TEXTURE_2D / GLES11Ext.GL_TEXTURE_EXTERNAL_OES???
     */
    public int getTextureType() {
        return GLES30.GL_TEXTURE_2D;
    }

    /**
     * ????????????
     */
    public void release() {
        if (mIsInitialized) {
            GLES30.glDeleteProgram(mProgramHandle);
            mProgramHandle = OpenGLUtils.GL_NOT_INIT;
        }
        destroyFrameBuffer();
    }

    /**
     * ??????FBO
     * @param width
     * @param height
     */
    public void initFrameBuffer(int width, int height) {
        if (!isInitialized()) {
            return;
        }
        if (mFrameBuffers != null && (mFrameWidth != width || mFrameHeight != height)) {
            destroyFrameBuffer();
        }
        if (mFrameBuffers == null) {
            mFrameWidth = width;
            mFrameHeight = height;
            mFrameBuffers = new int[1];
            mFrameBufferTextures = new int[1];
            OpenGLUtils.createFrameBuffer(mFrameBuffers, mFrameBufferTextures, width, height);
        }
    }

    /**
     * ????????????
     */
    public void destroyFrameBuffer() {
        if (!mIsInitialized) {
            return;
        }
        if (mFrameBufferTextures != null) {
            GLES30.glDeleteTextures(1, mFrameBufferTextures, 0);
            mFrameBufferTextures = null;
        }

        if (mFrameBuffers != null) {
            GLES30.glDeleteFramebuffers(1, mFrameBuffers, 0);
            mFrameBuffers = null;
        }
        mFrameWidth = -1;
        mFrameWidth = -1;
    }

    /**
     * ?????????????????????
     * @return
     */
    public boolean isInitialized() {
        return mIsInitialized;
    }

    /**
     * ????????????????????????
     * @param enable
     */
    public void setFilterEnable(boolean enable) {
        mFilterEnable = enable;
    }

    /**
     * ??????????????????
     * @return
     */
    public int getDisplayWidth() {
        return mDisplayWidth;
    }

    /**
     * ??????????????????
     * @return
     */
    public int getDisplayHeight() {
        return mDisplayHeight;
    }
    // ------------------ ????????????(uniform)?????? ------------------------ //
    protected void setInteger(final int location, final int intValue) {
        runOnDraw(new Runnable() {
            @Override
            public void run() {
                GLES30.glUniform1i(location, intValue);
            }
        });
    }

    protected void setFloat(final int location, final float floatValue) {
        runOnDraw(new Runnable() {
            @Override
            public void run() {
                GLES30.glUniform1f(location, floatValue);
            }
        });
    }

    protected void setFloatVec2(final int location, final float[] arrayValue) {
        runOnDraw(new Runnable() {
            @Override
            public void run() {
                GLES30.glUniform2fv(location, 1, FloatBuffer.wrap(arrayValue));
            }
        });
    }

    protected void setFloatVec3(final int location, final float[] arrayValue) {
        runOnDraw(new Runnable() {
            @Override
            public void run() {
                GLES30.glUniform3fv(location, 1, FloatBuffer.wrap(arrayValue));
            }
        });
    }

    protected void setFloatVec4(final int location, final float[] arrayValue) {
        runOnDraw(new Runnable() {
            @Override
            public void run() {
                GLES30.glUniform4fv(location, 1, FloatBuffer.wrap(arrayValue));
            }
        });
    }

    protected void setFloatArray(final int location, final float[] arrayValue) {
        runOnDraw(new Runnable() {
            @Override
            public void run() {
                GLES30.glUniform1fv(location, arrayValue.length, FloatBuffer.wrap(arrayValue));
            }
        });
    }

    protected void setPoint(final int location, final PointF point) {
        runOnDraw(new Runnable() {

            @Override
            public void run() {
                float[] vec2 = new float[2];
                vec2[0] = point.x;
                vec2[1] = point.y;
                GLES30.glUniform2fv(location, 1, vec2, 0);
            }
        });
    }

    protected void setUniformMatrix3f(final int location, final float[] matrix) {
        runOnDraw(new Runnable() {

            @Override
            public void run() {
                GLES30.glUniformMatrix3fv(location, 1, false, matrix, 0);
            }
        });
    }

    protected void setUniformMatrix4f(final int location, final float[] matrix) {
        runOnDraw(new Runnable() {

            @Override
            public void run() {
                GLES30.glUniformMatrix4fv(location, 1, false, matrix, 0);
            }
        });
    }

    /**
     * ??????????????????
     * @param runnable
     */
    protected void runOnDraw(final Runnable runnable) {
        synchronized (mRunOnDraw) {
            mRunOnDraw.addLast(runnable);
        }
    }

    /**
     * ??????????????????
     */
    protected void runPendingOnDrawTasks() {
        while (!mRunOnDraw.isEmpty()) {
            mRunOnDraw.removeFirst().run();
        }
    }

    protected static float clamp(float value, float min, float max) {
        if (value < min) {
            return min;
        } else if (value > max) {
            return max;
        }
        return value;
    }
}
