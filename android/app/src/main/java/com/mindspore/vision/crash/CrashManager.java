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

package com.mindspore.vision.crash;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Environment;
import android.util.Log;

import com.mindspore.utilslibrary.ui.utils.StorageUtils;

import org.jetbrains.annotations.NotNull;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class CrashManager implements Thread.UncaughtExceptionHandler {
    public static final String TAG = "CrashManager";
    public static final String PATH = StorageUtils.CRASH_PATH;
    public static final String FILE_NAME = "crash_";
    public static final String FILE_NAME_SUFFIX = ".txt";
    private Thread.UncaughtExceptionHandler mDefaultHandler;
    private Context mContext;


    private CrashManager() {
    }

    public static CrashManager getInstance() {
        CrashManager INSTANCE = new CrashManager();
        return INSTANCE;
    }

    public void init(Context context) {
        mContext = context;
        mDefaultHandler = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler(this);
    }

    @Override
    public void uncaughtException(@NotNull Thread thread, @NotNull Throwable ex) {
        dumpExceptionToSD(ex);

        if (mDefaultHandler != null) {
            mDefaultHandler.uncaughtException(thread, ex);
        } else {
            android.os.Process.killProcess(android.os.Process.myPid());
            System.exit(1);
        }
    }


    private void dumpExceptionToSD(Throwable ex) {
        if (!Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {

        } else {
            File dir = new File(PATH);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            long currentData = System.currentTimeMillis();
            String time = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(new Date(currentData));
            File file = new File(PATH + FILE_NAME + time.replace(" ", "_") + FILE_NAME_SUFFIX);
            Log.e(TAG, "crash file path:" + file.getAbsolutePath());
            try {
                PrintWriter printWriter = new PrintWriter(new BufferedWriter(new FileWriter(file)));
                printWriter.println(time);
                phoneInformation(printWriter);
                printWriter.println();
                ex.printStackTrace(printWriter);
                printWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
                Log.e(TAG, "writer crash log failed");
            } catch (PackageManager.NameNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * phone info
     */
    private void phoneInformation(PrintWriter pw) throws PackageManager.NameNotFoundException {
        PackageManager pm = mContext.getPackageManager();
        PackageInfo pi = pm.getPackageInfo(mContext.getPackageName(), PackageManager.GET_ACTIVITIES);
        pw.println("App Version: " + pi.versionName + "_versionCode:" + pi.versionCode);
        pw.println("OS Version: " + Build.VERSION.RELEASE + "_SDK:" + Build.VERSION.SDK_INT);
        pw.println("Vendor: " + Build.MANUFACTURER);
        pw.println("Model: " + Build.MODEL);
        pw.println("CPU ABI: " + Build.CPU_ABI);

    }

}
