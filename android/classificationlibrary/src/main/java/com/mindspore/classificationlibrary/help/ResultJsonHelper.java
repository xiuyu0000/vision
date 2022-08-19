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

package com.mindspore.classificationlibrary.help;


import com.google.gson.Gson;
import com.mindspore.classificationlibrary.bean.ModelLabelBean;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class ResultJsonHelper {

    public static ModelLabelBean gonsAnalyzeJSON(String filePath) {
        String jsonData = readFile(filePath);
        Gson gson = new Gson();
        return gson.fromJson(jsonData, ModelLabelBean.class);
    }

    private static String readFile(String fileName) {
        String result = null;
        try {
            InputStream inputStream = new FileInputStream(fileName);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] bytes = new byte[inputStream.available()];
            inputStream.read(bytes);
            baos.write(bytes, 0, bytes.length);
            result = new String(baos.toByteArray());
            baos.close();
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }

}
