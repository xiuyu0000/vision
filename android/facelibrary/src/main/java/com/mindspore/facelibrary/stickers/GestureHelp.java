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

package com.mindspore.facelibrary.stickers;

import com.badlogic.gdx.math.Camera;
import com.badlogic.gdx.math.Vector3;
import com.mindspore.facelibrary.base.GLImageFilter;

import java.util.List;

/**
 * 描述：相机和屏幕坐标转换，用于触摸控制贴纸的旋转，平移，缩放操作
 */
public class GestureHelp {


    /**
     * 屏幕坐标转本地坐标
     * @param camera
     * @param screenCoords
     * @return
     */
    public static Vector3 screenToStageCoordinates (Camera camera, final Vector3 screenCoords) {
        camera.unproject(screenCoords);
        return screenCoords;
    }

    /**
     * 本地坐标转屏幕坐标
     * @param camera
     * @param stageCoords
     * @return
     */
    public static Vector3 stageToScreenCoordinates (final Camera camera,final Vector3 stageCoords) {
        camera.project(stageCoords);
        stageCoords.y = camera.getScreenHeight() - stageCoords.y;
        return stageCoords;
    }


    public static StaticStickerNormalFilter hit(final Vector3 target,final List<GLImageFilter> mFilters){
        for(GLImageFilter glImageFilter:mFilters){
            if(glImageFilter instanceof StaticStickerNormalFilter){
                StaticStickerNormalFilter staticStickerNormalFilter=((StaticStickerNormalFilter)glImageFilter);
                //屏幕坐标转本地坐标
                screenToStageCoordinates(staticStickerNormalFilter.camera,target);
                //获取触摸到的贴纸
                return  ((StaticStickerNormalFilter)glImageFilter).hit(target);
            }
        }
        return null;
    }
}
