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

package com.mindspore.enginelibrary.utils;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

public class SensorEventUtil implements SensorEventListener {
    private SensorManager mSensorManager;
    private Sensor mSensor;

    public int orientation = 0;

    public SensorEventUtil(Context context) {
        mSensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        mSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);// TYPE_GRAVITY
        // 参数三，检测的精准度
        mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_NORMAL);// SENSOR_DELAY_GAME
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        final double G = 9.81;
        final double SQRT2 = 1.414213;
        if (event.sensor == null) {
            return;
        }

        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            float x = event.values[0];
            float y = event.values[1];
            float z = event.values[2];
            if (z >= G / SQRT2) { //screen is more likely lying on the table
                if (x >= G / 2) {
                    orientation = 1;
                } else if (x <= -G / 2) {
                    orientation = 2;
                } else if (y <= -G / 2) {
                    orientation = 3;
                } else {
                    orientation = 0;
                }
            } else {
                if (x >= G / SQRT2) {
                    orientation = 1;
                } else if (x <= -G / SQRT2) {
                    orientation = 2;
                } else if (y <= -G / SQRT2) {
                    orientation = 3;
                } else {
                    orientation = 0;
                }
            }
        }
    }
}
