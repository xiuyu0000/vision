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

package com.mindspore.landmarklibrary;

import android.graphics.PointF;

/**
 * 人脸关键点计算工具
 */
public class FacePointsUtils {

    /**
     * 求两点之间的距离
     * @param p1
     * @param p2
     * @return
     */
    public static double getDistance(PointF p1, PointF p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    /**
     * 求两点之间的距离
     * @param point1
     * @param point2
     * @return
     */
    public static double getDistance(float[] point1, float[] point2) {
        return Math.sqrt(Math.pow(point1[0] - point2[0], 2) + Math.pow(point1[1] - point2[1], 2));
    }

    /**
     * 求两点之间的距离
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     * @return
     */
    public static double getDistance(float x1, float y1, float x2, float y2) {
        return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
    }

    /**
     * 获取两点中心点
     * @param p1
     * @param p2
     * @return
     */
    public static PointF getCenter(PointF p1, PointF p2) {
        PointF pointF = new PointF();
        pointF.x = (p1.x + p2.x) / 2;
        pointF.y = (p1.y + p2.y) / 2;
        return pointF;
    }

    /**
     * 获取两点中心点
     * @param point1
     * @param point2
     * @return
     */
    public static float[] getCenter(float[] point1, float[] point2) {
        float[] point = new float[2];
        point[0] = (point1[0] + point2[0]) / 2.0f;
        point[1] = (point1[1] + point2[1]) / 2.0f;
        return point;
    }

    /**
     * 获取两点的中心点
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     * @return
     */
    public static float[] getCenter(float x1, float y1, float x2, float y2) {
        float[] point = new float[2];
        point[0] = (x1 + x2) / 2.0f;
        point[1] = (y1 + y2) / 2.0f;
        return point;
    }

    /**
     * 计算中心点
     * @param result
     * @param x1
     * @param y1
     * @param x2
     * @param y2
     */
    public static void getCenter(float[] result, float x1, float y1, float x2, float y2) {
        result[0] = (x1 + x2) / 2.0f;
        result[1] = (y1 + y2) / 2.0f;
    }
}
