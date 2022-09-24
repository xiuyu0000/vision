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

package com.mindspore.facelibrary.opengl.view;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;

public class BigEyeGLSurfaceView extends GLSurfaceView {

    private BigEyeRenderer bigEyeRenderer;

    public BigEyeGLSurfaceView(Context context) {
        this(context,null);
    }

    public BigEyeGLSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init(){
        setEGLContextClientVersion(2);
        bigEyeRenderer = new BigEyeRenderer(this);
        setRenderer(bigEyeRenderer);
        setRenderMode(RENDERMODE_WHEN_DIRTY);
    }
}
