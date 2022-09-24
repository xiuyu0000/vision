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

package com.mindspore.enginelibrary;

import android.graphics.PointF;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
public class ExampleUnitTest {
    @Test
    public void addition_isCorrect() {
        float[] landmarks = new float[]{7, 189, 24, 250, 23, 333, 32, 408, 45, 470, 84, 523, 128, 551,
                173, 614, 225, 617, 262, 595, 310, 550, 333, 530, 355, 478, 368, 407, 387, 340,
                381, 265, 389, 214, 78, 138, 83, 114, 130, 99, 171, 101, 215, 125, 262, 117, 301,
                107, 334, 111, 361, 105, 380, 147, 240, 186, 236, 235, 241, 288, 245, 333, 192,
                352, 219, 365, 230, 371, 259, 369, 278, 363, 100, 196, 130, 176, 162, 180, 193,
                207, 161, 208, 133, 208, 285, 207, 293, 183, 322, 189, 339, 192, 317, 209, 292,
                211, 142, 425, 181, 414, 213, 408, 248, 429, 253, 414, 280, 426, 301, 451, 280,
                462, 241, 482, 229, 494, 209, 489, 177, 474, 153, 448, 216, 450, 227, 438, 248,
                438, 287, 445, 259, 453, 234, 462, 214, 453};

        PointF[] point = new PointF[landmarks.length / 2];
        for (int i = 0; i < point.length; i=i+2) {
                point[i] = new PointF(landmarks[i], landmarks[i + 1]);

        }
    }


}