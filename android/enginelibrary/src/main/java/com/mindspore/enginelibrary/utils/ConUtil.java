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

import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager.NameNotFoundException;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.media.ExifInterface;
import android.os.Build;
import android.os.Environment;
import android.os.PowerManager;
import android.text.TextUtils;
import android.util.Base64;
import android.view.Gravity;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Toast;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.UUID;

public class ConUtil {

    public static boolean isReadKey(Context context) {
        InputStream inputStream = null;
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int count = -1;
        try {
            inputStream = context.getAssets().open("key");
            while ((count = inputStream.read(buffer)) != -1) {
                byteArrayOutputStream.write(buffer, 0, count);
            }
            byteArrayOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        String str = new String(byteArrayOutputStream.toByteArray());
        String key = null;
        String screct = null;
        try {
            String[] strs = str.split(";");
            key = strs[0].trim();
            screct = strs[1].trim();
        } catch (Exception e) {
        }
        FaceppConstraints.API_KEY = key;
        FaceppConstraints.API_SECRET = screct;
        if (FaceppConstraints.API_KEY == null || FaceppConstraints.API_SECRET == null) {
            return false;
        }

        return true;
    }


    public static void toggleHideyBar(Activity activity) {
        int uiOptions = activity.getWindow().getDecorView().getSystemUiVisibility();
        int newUiOptions = uiOptions;


        if (Build.VERSION.SDK_INT >= 14) {
            newUiOptions ^= View.SYSTEM_UI_FLAG_HIDE_NAVIGATION;
        }

        if (Build.VERSION.SDK_INT >= 16) {
            newUiOptions ^= View.SYSTEM_UI_FLAG_FULLSCREEN;
        }

        if (Build.VERSION.SDK_INT >= 19) {
            newUiOptions ^= View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
        }

        activity.getWindow().getDecorView().setSystemUiVisibility(newUiOptions);
    }

    /**
     * ???????????????(????????????)
     */
    public static String getFormatterDate(long time) {
        Date d = new Date(time);
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
        String data = formatter.format(d);
        return data;
    }

    public static String getUUIDString(Context mContext) {
        String KEY_UUID = "key_uuid";
        SharedUtil sharedUtil = new SharedUtil(mContext);
        String uuid = sharedUtil.getStringValueByKey(KEY_UUID);
        if (uuid != null && uuid.trim().length() != 0)
            return uuid;

        uuid = UUID.randomUUID().toString();
        uuid = Base64.encodeToString(uuid.getBytes(),
                Base64.DEFAULT);

        sharedUtil.saveStringValue(KEY_UUID, uuid);
        return uuid;
    }

    public static Bitmap decodeToBitMap(byte[] data, Camera _camera) {
        Camera.Size size = _camera.getParameters().getPreviewSize();
        try {
            YuvImage image = new YuvImage(data, ImageFormat.NV21, size.width, size.height, null);
            if (image != null) {
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                image.compressToJpeg(new Rect(0, 0, size.width, size.height), 80, stream);
                Bitmap bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());
                stream.close();
                return bmp;
            }
        } catch (Exception ex) {
        }
        return null;
    }

    /**
     * ???????????????
     */
    public static void isGoneKeyBoard(Activity activity) {
        if (activity.getCurrentFocus() != null) {
            // ???????????????
            ((InputMethodManager) activity
                    .getSystemService(activity.INPUT_METHOD_SERVICE))
                    .hideSoftInputFromWindow(activity.getCurrentFocus()
                                    .getWindowToken(),
                            InputMethodManager.HIDE_NOT_ALWAYS);
        }
    }

    public static PowerManager.WakeLock wakeLock = null;

    public static void acquireWakeLock(Context context) {
        if (wakeLock == null) {
            PowerManager powerManager = (PowerManager) (context
                    .getSystemService(Context.POWER_SERVICE));
            wakeLock = powerManager.newWakeLock(
                    PowerManager.SCREEN_DIM_WAKE_LOCK, "My Tag");
            wakeLock.acquire();
        }
    }

    public static void releaseWakeLock() {
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
            wakeLock = null;
        }
    }

    /**
     * ??????bitmap???????????????
     */
    public static byte[] getGrayscale(Bitmap bitmap) {
        if (bitmap == null)
            return null;

        byte[] ret = new byte[bitmap.getWidth() * bitmap.getHeight()];
        for (int j = 0; j < bitmap.getHeight(); ++j)
            for (int i = 0; i < bitmap.getWidth(); ++i) {
                int pixel = bitmap.getPixel(i, j);
                int red = ((pixel & 0x00FF0000) >> 16);
                int green = ((pixel & 0x0000FF00) >> 8);
                int blue = pixel & 0x000000FF;
                ret[j * bitmap.getWidth() + i] = (byte) ((299 * red + 587
                        * green + 114 * blue) / 1000);
            }
        return ret;
    }

    public static byte[] convertYUV21FromRGB(Bitmap bitmap) {
        bitmap = rotaingImageView(90, bitmap);

        int inputWidth = bitmap.getWidth();
        int inputHeight = bitmap.getHeight();

        int[] argb = new int[inputWidth * inputHeight];

        bitmap.getPixels(argb, 0, inputWidth, 0, 0, inputWidth, inputHeight);

        byte[] yuv = new byte[inputWidth * inputHeight * 3 / 2];

        encodeYUV420SP(yuv, argb, inputWidth, inputHeight);

        bitmap.recycle();

        return yuv;

    }

    private static void encodeYUV420SP(byte[] yuv420sp, int[] argb, int width, int height) {
        final int frameSize = width * height;

        int yIndex = 0;
        int uvIndex = frameSize;

        int a, R, G, B, Y, U, V;
        int index = 0;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {

                a = (argb[index] & 0xff000000) >> 24; // a is not used obviously
                R = (argb[index] & 0xff0000) >> 16;
                G = (argb[index] & 0xff00) >> 8;
                B = (argb[index] & 0xff) >> 0;

                // well known RGB to YUV algorithm
                Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16;
                U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128;
                V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128;

                // NV21 has a plane of Y and interleaved planes of VU each sampled by a factor of 2
                //    meaning for every 4 Y pixels there are 1 V and 1 U.  Note the sampling is every other
                //    pixel AND every other scanline.
                yuv420sp[yIndex++] = (byte) ((Y < 0) ? 0 : ((Y > 255) ? 255 : Y));
                if (j % 2 == 0 && index % 2 == 0) {
                    yuv420sp[uvIndex++] = (byte) ((V < 0) ? 0 : ((V > 255) ? 255 : V));
                    yuv420sp[uvIndex++] = (byte) ((U < 0) ? 0 : ((U > 255) ? 255 : U));
                }

                index++;
            }
        }
    }


    public static byte[] getFileContent(Context context, int id) {
        InputStream inputStream = null;
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int count = -1;
        try {
            inputStream = context.getResources().openRawResource(id);
            while ((count = inputStream.read(buffer)) != -1) {
                byteArrayOutputStream.write(buffer, 0, count);
            }
            byteArrayOutputStream.close();
        } catch (IOException e) {
            return null;
        } finally {
            // closeStreamSilently(inputStream);
            inputStream = null;
        }
        return byteArrayOutputStream.toByteArray();
    }

    /**
     * ??????toast
     */
    public static void showToast(Context context, String str) {
        if (context != null) {
            Toast toast = Toast.makeText(context, str, Toast.LENGTH_SHORT);
            // ????????????toast???????????????
            toast.setGravity(Gravity.TOP, 0, 30);
            toast.show();
        }
    }

    /**
     * ???????????????toast
     */
    public static void showLongToast(Context context, String str) {
        if (context != null) {
            Toast toast = Toast.makeText(context, str, Toast.LENGTH_LONG);
            // ????????????toast???????????????
            toast.setGravity(Gravity.TOP, 0, 30);
            toast.show();
        }
    }

    /**
     * ??????APP?????????
     */
    public static String getVersionName(Context context) {
        try {
            String versionName = context.getPackageManager().getPackageInfo(
                    context.getPackageName(), 0).versionName;
            return versionName;
        } catch (NameNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * ????????????
     */
    public static Bitmap convert(Bitmap bitmap, boolean mIsFrontalCamera) {
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        Bitmap newbBitmap = Bitmap.createBitmap(w, h, Config.ARGB_8888);// ?????????????????????SRC???????????????????????????
        Canvas cv = new Canvas(newbBitmap);
        Matrix m = new Matrix();
        // m.postScale(1, -1); //??????????????????
        if (mIsFrontalCamera) {
            m.postScale(-1, 1); // ??????????????????
        }
        // m.postRotate(-90); //??????-90???
        Bitmap bitmap2 = Bitmap.createBitmap(bitmap, 0, 0, w, h, m, true);
        cv.drawBitmap(bitmap2,
                new Rect(0, 0, bitmap2.getWidth(), bitmap2.getHeight()),
                new Rect(0, 0, w, h), null);
        return newbBitmap;
    }

    public static byte[] readYUVInfo(Context ctx) {

        String path = getDiskCachePath(ctx);
        String pathName = path + "/yuv.img";
        File file = new File(pathName);

        if (!file.exists()) return null;

        InputStream inputStream = null;
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        byte[] buffer = new byte[1024];
        int count = -1;
        try {
            inputStream = new FileInputStream(file);
            while ((count = inputStream.read(buffer)) != -1) {
                byteArrayOutputStream.write(buffer, 0, count);
            }
            byteArrayOutputStream.close();
        } catch (IOException e) {
            return null;
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return byteArrayOutputStream.toByteArray();
    }


    public static void saveYUVInfo(Context ctx, byte[] arr) {
        if (arr == null) return;

        String path = getDiskCachePath(ctx);
        String pathName = path + "/yuv.img";
        File file = new File(pathName);

        FileOutputStream fileOutputStream = null;

        try {
            fileOutputStream = new FileOutputStream(file);
            fileOutputStream.write(arr);
            fileOutputStream.flush();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

    }


    /**
     * ??????bitmap?????????Picture?????????
     */
    public static String saveBitmap(Context mContext, Bitmap bitmaptosave) {
        if (bitmaptosave == null)
            return null;

        File mediaStorageDir = mContext.getExternalFilesDir("megvii");

        if (!mediaStorageDir.exists()) {
            if (!mediaStorageDir.mkdirs()) {
                return null;
            }
        }
        // String bitmapFileName = System.currentTimeMillis() + ".jpg";
        String bitmapFileName = System.currentTimeMillis() + "";
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mediaStorageDir + "/" + bitmapFileName);
            boolean successful = bitmaptosave.compress(
                    Bitmap.CompressFormat.JPEG, 75, fos);

            if (successful)
                return mediaStorageDir.getAbsolutePath() + "/" + bitmapFileName;
            else
                return null;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static Bitmap revitionImage(String path, int width, int height) {
        if (null == path || TextUtils.isEmpty(path) || !new File(path).exists())
            return null;
        BufferedInputStream in = null;
        try {
            // ??????????????????????????????
            int degree = readPictureDegree(path);
            in = new BufferedInputStream(new FileInputStream(new File(path)));
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeStream(in, null, options);
            // ??????????????????????????????
            options.inSampleSize = calculateInSampleSize(options, width, height);

            in.close();
            in = new BufferedInputStream(new FileInputStream(new File(path)));
            options.inJustDecodeBounds = false;
            Bitmap bitmap = BitmapFactory.decodeStream(in, null, options);
            Bitmap newbitmap = rotaingImageView(degree, bitmap);
            return newbitmap;
        } catch (Exception e) {
            return null;
        } finally {
            if (null != in) {
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                in = null;
            }
        }
    }

    public static int readPictureDegree(String path) {
        int degree = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degree = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degree = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degree = 270;
                    break;
            }
        } catch (IOException e) {
            // Logger.getLogger(PhotoHelper.class).e(e.getMessage());
        }
        return degree;
    }

    public static int calculateInSampleSize(BitmapFactory.Options options,
                                            int reqWidth, int reqHeight) {
        // ???????????????????????????
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;
        if (height > reqHeight || width > reqWidth) {
            // ?????????????????????????????????????????????
            final int heightRatio = Math.round((float) height / (float) reqHeight);
            final int widthRatio = Math.round((float) width / (float) reqWidth);
            // ???????????????????????????????????????inSampleSize???????????????????????????????????????????????????
            // ?????????????????????????????????????????????
            inSampleSize = heightRatio < widthRatio ? heightRatio : widthRatio;
        }
        return inSampleSize;
    }

    public static Bitmap rotaingImageView(int angle, Bitmap bitmap) {
        if (null == bitmap) {
            return null;
        }
        // ???????????? ??????
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        // ??????????????????
        Bitmap resizedBitmap = Bitmap.createBitmap(bitmap, 0, 0,
                bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        return resizedBitmap;
    }

    /**
     * ??????cache??????
     */
    public static String getDiskCachePath(Context context) {
        if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState()) || !Environment.isExternalStorageRemovable()) {
            return context.getExternalCacheDir().getPath();
        } else {
            return context.getCacheDir().getPath();
        }
    }

    public static String getSDRootPath() {
        if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState()) || !Environment.isExternalStorageRemovable()) {
            return Environment.getExternalStorageDirectory().getPath();
        } else {
            return null;
        }

    }


    public static int dip2px(Context context, float dipValue) {
        final float scale = context.getResources().getDisplayMetrics().density;
        return (int) (dipValue * scale + 0.5f);
    }

    public static int px2dip(Context context, float pxValue) {
        final float scale = context.getResources().getDisplayMetrics().density;
        return (int) (pxValue / scale + 0.5f);
    }

}
