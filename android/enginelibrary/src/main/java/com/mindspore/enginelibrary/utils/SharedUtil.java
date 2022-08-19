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

package com.mindspore.enginelibrary.utils;

import android.content.Context;
import android.content.SharedPreferences;

import java.util.Map;


/**
 * Save Data To SharePreference Or Get Data from SharePreference
 * <p>
 * 通过SharedPreferences来存储数据，自定义类型
 */
public class SharedUtil {
    private Context ctx;
    private String FileName = "vision";

    public SharedUtil(Context ctx) {
        this.ctx = ctx;
    }

    public void saveIntValue(String key, int value) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharePre.edit();
        editor.putInt(key, value);
        editor.commit();
    }

    public void saveLongValue(String key, long value) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharePre.edit();
        editor.putLong(key, value);
        editor.commit();
    }

    public void writeDownStartApplicationTime() {
        SharedPreferences sp = ctx.getSharedPreferences(FileName, Context.MODE_PRIVATE);
        long now = System.currentTimeMillis();
        //Calendar calendar = Calendar.getInstance();
        //Date now = calendar.getTime();
        //SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd:hh-mm-ss");
        SharedPreferences.Editor editor = sp.edit();
        //editor.putString("启动时间", now.toString());
        editor.putLong("nowtimekey", now);
        editor.commit();

    }

    public void saveBooleanValue(String key, boolean value) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharePre.edit();
        editor.putBoolean(key, value);
        editor.commit();
    }

    public void removeSharePreferences(String key) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharePre.edit();
        editor.remove(key);
        editor.commit();
    }

    public boolean contains(String key) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        return sharePre.contains(key);
    }

    public Map<String, Object> getAllMap() {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        return (Map<String, Object>) sharePre.getAll();
    }

    public Integer getIntValueByKey(String key) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        return sharePre.getInt(key, -1);
    }

    public Long getLongValueByKey(String key) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        return sharePre.getLong(key, -1);
    }

    public void saveStringValue(String key, String value) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharePre.edit();
        editor.putString(key, value);
        editor.commit();
    }

    public String getStringValueByKey(String key) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        return sharePre.getString(key, null);
    }

    public Boolean getBooleanValueByKey(String key) {
        SharedPreferences sharePre = ctx.getSharedPreferences(FileName,
                Context.MODE_PRIVATE);
        return sharePre.getBoolean(key, false);
    }

    public Integer getIntValueAndRemoveByKey(String key) {
        Integer value = getIntValueByKey(key);
        removeSharePreferences(key);
        return value;
    }

    public void setUserkey(String userkey) {
        this.saveStringValue("params_userkey", userkey);
    }

    public String getUserkey() {
        return this.getStringValueByKey("params_userkey");
    }

}
