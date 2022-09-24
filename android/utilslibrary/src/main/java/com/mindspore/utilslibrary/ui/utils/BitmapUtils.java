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

package com.mindspore.utilslibrary.ui.utils;

import android.app.Activity;
import android.content.ContentProviderClient;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.graphics.drawable.Drawable;
import android.media.ExifInterface;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.RemoteException;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;


public class BitmapUtils {

    private static final String TAG = "BitmapUtils";

    private static final int NUM_HUNDRED = 100;
    private static final int NUM_256 = 256;
    private static final float NUM_1920 = 1920f;
    private static final float NUM_1080 = 1080f;
    private static final int ROTATE_90 = 90;
    private static final int ROTATE_180 = 180;
    private static final int ROTATE_270 = 270;

    public static final String[] EXIF_TAGS = {
            "FNumber",
            ExifInterface.TAG_DATETIME,
            "ExposureTime",
            ExifInterface.TAG_FLASH,
            ExifInterface.TAG_FOCAL_LENGTH,
            "GPSAltitude", "GPSAltitudeRef",
            ExifInterface.TAG_GPS_DATESTAMP,
            ExifInterface.TAG_GPS_LATITUDE,
            ExifInterface.TAG_GPS_LATITUDE_REF,
            ExifInterface.TAG_GPS_LONGITUDE,
            ExifInterface.TAG_GPS_LONGITUDE_REF,
            ExifInterface.TAG_GPS_PROCESSING_METHOD,
            ExifInterface.TAG_GPS_TIMESTAMP,
            ExifInterface.TAG_IMAGE_LENGTH,
            ExifInterface.TAG_IMAGE_WIDTH, "ISOSpeedRatings",
            ExifInterface.TAG_MAKE, ExifInterface.TAG_MODEL,
            ExifInterface.TAG_WHITE_BALANCE,
    };

    /**
     * 从Buffer中创建Bitmap
     *
     * @param buffer
     * @param width
     * @param height
     * @return
     */
    public static Bitmap getBitmapFromBuffer(ByteBuffer buffer, int width, int height) {
        return getBitmapFromBuffer(buffer, width, height, false, false);
    }

    /**
     * 从Buffer中创建Bitmap
     *
     * @param buffer
     * @param width
     * @param height
     * @param flipX
     * @param flipY
     * @return
     */
    public static Bitmap getBitmapFromBuffer(ByteBuffer buffer, int width, int height,
                                             boolean flipX, boolean flipY) {
        if (buffer == null) {
            return null;
        }
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);
        if (flipX || flipY) {
            Bitmap result = flipBitmap(bitmap, flipX, flipY, true);
            return result;
        } else {
            return bitmap;
        }
    }

    /**
     * 从byte[]中创建Bitmap
     *
     * @param imgByte
     * @return
     */
    public static Bitmap getBitmapFromByte(byte[] imgByte) {
//        if (imgByte.length != 0) {
//            return BitmapFactory.decodeByteArray(imgByte, 0, imgByte.length);
//        } else {
//            return null;
//        }


        YuvImage yuvimage=new YuvImage(imgByte, ImageFormat.NV21, 20,20, null);//20、20分别是图的宽度与高度
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        yuvimage.compressToJpeg(new Rect(0, 0,20, 20), 80, baos);//80--JPG图片的质量[0-100],100最高
        byte[] jdata = baos.toByteArray();

        Bitmap bmp = BitmapFactory.decodeByteArray(jdata, 0, jdata.length);
        return bmp;
    }

    /**
     * 从普通文件中读入图片
     *
     * @param fileName
     * @return
     */
    public static Bitmap getBitmapFromFile(String fileName) {
        Bitmap bitmap;
        File file = new File(fileName);
        if (!file.exists()) {
            return null;
        }
        try {
            bitmap = BitmapFactory.decodeFile(fileName);
        } catch (Exception e) {
            Log.e(TAG, "getBitmapFromFile: ", e);
            bitmap = null;
        }
        return bitmap;
    }

    /**
     * 加载Assets文件夹下的图片
     *
     * @param context
     * @param fileName
     * @return
     */
    public static Bitmap getImageFromAssetsFile(Context context, String fileName) {
        Bitmap bitmap = null;
        AssetManager manager = context.getResources().getAssets();
        try {
            InputStream is = manager.open(fileName);
            bitmap = BitmapFactory.decodeStream(is);
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bitmap;
    }

    /**
     * 加载Assets文件夹下的图片
     *
     * @param context
     * @param fileName
     * @return
     */
    public static Bitmap getImageFromAssetsFile(Context context, String fileName, Bitmap inBitmap) {
        Bitmap bitmap = null;
        AssetManager manager = context.getResources().getAssets();
        try {
            InputStream is = manager.open(fileName);
            if (inBitmap != null && !inBitmap.isRecycled()) {
                BitmapFactory.Options options = new BitmapFactory.Options();
                // 使用inBitmap时，inSampleSize得设置为1
                options.inSampleSize = 1;
                // 这个属性一定要在inBitmap之前使用，否则会弹出一下异常
                // BitmapFactory: Unable to reuse an immutable bitmap as an image decoder target.
                options.inMutable = true;
                options.inBitmap = inBitmap;
                bitmap = BitmapFactory.decodeStream(is, null, options);
            } else {
                bitmap = BitmapFactory.decodeStream(is);
            }
            is.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bitmap;
    }


    /**
     * 从文件读取Bitmap
     *
     * @param dst       目标路径
     * @param maxWidth  读入最大宽度, 为0时，直接读入原图
     * @param maxHeight 读入最大高度，为0时，直接读入原图
     * @return
     */
    public static Bitmap getBitmapFromFile(File dst, int maxWidth, int maxHeight) {
        if (null != dst && dst.exists()) {
            BitmapFactory.Options opts = null;
            if (maxWidth > 0 && maxHeight > 0) {
                opts = new BitmapFactory.Options();
                opts.inJustDecodeBounds = true;
                BitmapFactory.decodeFile(dst.getPath(), opts);
                // 计算图片缩放比例
                opts.inSampleSize = calculateInSampleSize(opts, maxWidth, maxHeight);
                opts.inJustDecodeBounds = false;
                opts.inInputShareable = true;
                opts.inPurgeable = true;
            }
            try {
                return BitmapFactory.decodeFile(dst.getPath(), opts);
            } catch (OutOfMemoryError e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    /**
     * 从文件读取Bitmap
     *
     * @param dst                目标路径
     * @param maxWidth           读入最大宽度，为0时，直接读入原图
     * @param maxHeight          读入最大高度，为0时，直接读入原图
     * @param processOrientation 是否处理图片旋转角度
     * @return
     */
    public static Bitmap getBitmapFromFile(File dst, int maxWidth, int maxHeight, boolean processOrientation) {
        if (null != dst && dst.exists()) {
            BitmapFactory.Options opts = null;
            if (maxWidth > 0 && maxHeight > 0) {
                opts = new BitmapFactory.Options();
                opts.inJustDecodeBounds = true;
                BitmapFactory.decodeFile(dst.getPath(), opts);
                // 计算图片缩放比例
                opts.inSampleSize = calculateInSampleSize(opts, maxWidth, maxHeight);
                opts.inJustDecodeBounds = false;
                opts.inInputShareable = true;
                opts.inPurgeable = true;
            }
            try {
                Bitmap bitmap = BitmapFactory.decodeFile(dst.getPath(), opts);
                if (!processOrientation) {
                    return bitmap;
                }
                int orientation = getOrientation(dst.getPath());
                if (orientation == 0) {
                    return bitmap;
                } else {
                    Matrix matrix = new Matrix();
                    matrix.postRotate(orientation);
                    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                }
            } catch (OutOfMemoryError e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    /**
     * 从Drawable中获取Bitmap图片
     *
     * @param drawable
     * @return
     */
    public static Bitmap getBitmapFromDrawable(Drawable drawable) {
        int w = drawable.getIntrinsicWidth();
        int h = drawable.getIntrinsicHeight();
        Bitmap.Config config =
                drawable.getOpacity() != PixelFormat.OPAQUE ? Bitmap.Config.ARGB_8888
                        : Bitmap.Config.RGB_565;
        Bitmap bitmap = Bitmap.createBitmap(w, h, config);
        // 在View或者SurfaceView里的canvas.drawBitmap会看不到图，需要用以下方式处理
        Canvas canvas = new Canvas(bitmap);
        drawable.setBounds(0, 0, w, h);
        drawable.draw(canvas);

        return bitmap;
    }

    /**
     * 图片等比缩放
     *
     * @param bitmap
     * @param newWidth
     * @param newHeight
     * @param isRecycled
     * @return
     */
    public static Bitmap zoomBitmap(Bitmap bitmap, int newWidth, int newHeight, boolean isRecycled) {
        if (bitmap == null) {
            return null;
        }
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        if (scaleWidth < scaleHeight) {
            matrix.postScale(scaleWidth, scaleWidth);
        } else {
            matrix.postScale(scaleHeight, scaleHeight);
        }
        Bitmap result = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
        if (!bitmap.isRecycled() && isRecycled) {
            bitmap.recycle();
            bitmap = null;
        }
        return result;
    }

    /**
     * 保存图片
     *
     * @param context
     * @param path
     * @param bitmap
     */
    public static void saveBitmap(Context context, String path, Bitmap bitmap) {
        saveBitmap(context, path, bitmap, true);
    }

    /**
     * 保存图片
     *
     * @param context
     * @param path
     * @param bitmap
     * @param addToMediaStore
     */
    public static void saveBitmap(Context context, String path, Bitmap bitmap,
                                  boolean addToMediaStore) {
        final File file = new File(path);
        if (!file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }

        FileOutputStream fOut = null;
        try {
            fOut = new FileOutputStream(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        boolean compress = true;
        if (path.endsWith(".png")) {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fOut);
        } else if (path.endsWith(".jpeg") || path.endsWith(".jpg")) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fOut);
        } else { // 除了png和jpeg之外的图片格式暂时不支持
            compress = false;
        }
        try {
            fOut.flush();
            fOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // 添加到媒体库
        if (addToMediaStore && compress) {
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DATA, path);
            values.put(MediaStore.Images.Media.DISPLAY_NAME, file.getName());
            context.getContentResolver().insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        }
    }

    /**
     * 保存图片
     *
     * @param filePath
     * @param buffer
     * @param width
     * @param height
     */
    public static void saveBitmap(String filePath, ByteBuffer buffer, int width, int height) {
        BufferedOutputStream bos = null;
        try {
            bos = new BufferedOutputStream(new FileOutputStream(filePath));
            Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            bitmap.copyPixelsFromBuffer(buffer);
            bitmap = BitmapUtils.rotateBitmap(bitmap, 180, true);
            bitmap = BitmapUtils.flipBitmap(bitmap, true);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bos);
            bitmap.recycle();
            bitmap = null;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if (bos != null) {
                try {
                    bos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * 保存图片
     *
     * @param filePath
     * @param bitmap
     */
    public static void saveBitmap(String filePath, Bitmap bitmap) {
        if (bitmap == null) {
            return;
        }
        BufferedOutputStream bos = null;
        try {
            bos = new BufferedOutputStream(new FileOutputStream(filePath));
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bos);
            bitmap.recycle();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if (bos != null) {
                try {
                    bos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * 获取图片旋转角度
     *
     * @param path
     * @return
     */
    public static int getOrientation(final String path) {
        int rotation = 0;
        try {
            File file = new File(path);
            ExifInterface exif = new ExifInterface(file.getAbsolutePath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotation = 90;
                    break;

                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotation = 180;
                    break;

                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotation = 270;
                    break;

                default:
                    rotation = 0;
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rotation;
    }

    /**
     * 获取Uri路径图片的旋转角度
     *
     * @param context
     * @param uri
     * @return
     */
    public static int getOrientation(Context context, Uri uri) {
        final String scheme = uri.getScheme();
        ContentProviderClient provider = null;
        if (scheme == null || ContentResolver.SCHEME_FILE.equals(scheme)) {
            return getOrientation(uri.getPath());
        } else if (scheme.equals(ContentResolver.SCHEME_CONTENT)) {
            try {
                provider = context.getContentResolver().acquireContentProviderClient(uri);
            } catch (SecurityException e) {
                return 0;
            }
            if (provider != null) {
                Cursor cursor;
                try {
                    cursor = provider.query(uri, new String[]{
                                    MediaStore.Images.ImageColumns.ORIENTATION,
                                    MediaStore.Images.ImageColumns.DATA},
                            null, null, null);
                } catch (RemoteException e) {
                    e.printStackTrace();
                    return 0;
                }
                if (cursor == null) {
                    return 0;
                }

                int orientationIndex = cursor
                        .getColumnIndex(MediaStore.Images.ImageColumns.ORIENTATION);
                int dataIndex = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);

                try {
                    if (cursor.getCount() > 0) {
                        cursor.moveToFirst();

                        int rotation = 0;

                        if (orientationIndex > -1) {
                            rotation = cursor.getInt(orientationIndex);
                        }

                        if (dataIndex > -1) {
                            String path = cursor.getString(dataIndex);
                            rotation |= getOrientation(path);
                        }
                        return rotation;
                    }
                } finally {
                    cursor.close();
                }
            }
        }
        return 0;
    }

    /**
     * 获取图片大小
     *
     * @param path
     * @return
     */
    public static Point getBitmapSize(String path) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path, options);
        return new Point(options.outWidth, options.outHeight);
    }

    /**
     * 将Bitmap图片旋转90度
     *
     * @param data
     * @return
     */
    public static Bitmap rotateBitmap(byte[] data) {
        return rotateBitmap(data, 90);
    }

    /**
     * 将Bitmap图片旋转一定角度
     *
     * @param data
     * @param rotate
     * @return
     */
    public static Bitmap rotateBitmap(byte[] data, int rotate) {
        Bitmap bitmap = BitmapFactory.decodeByteArray(data, 0, data.length);
        Matrix matrix = new Matrix();
        matrix.reset();
        matrix.postRotate(rotate);
        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(),
                bitmap.getHeight(), matrix, true);
        bitmap.recycle();
        System.gc();
        return rotatedBitmap;
    }

    /**
     * 将Bitmap图片旋转90度
     *
     * @param bitmap
     * @param isRecycled
     * @return
     */
    public static Bitmap rotateBitmap(Bitmap bitmap, boolean isRecycled) {
        return rotateBitmap(bitmap, 90, isRecycled);
    }

    /**
     * 将Bitmap图片旋转一定角度
     *
     * @param bitmap
     * @param rotate
     * @param isRecycled
     * @return
     */
    public static Bitmap rotateBitmap(Bitmap bitmap, int rotate, boolean isRecycled) {
        if (bitmap == null) {
            return null;
        }
        Matrix matrix = new Matrix();
        matrix.reset();
        matrix.postRotate(rotate);
        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(),
                bitmap.getHeight(), matrix, true);
        if (!bitmap.isRecycled() && isRecycled) {
            bitmap.recycle();
            bitmap = null;
        }
        return rotatedBitmap;
    }

    /**
     * 镜像翻转图片
     *
     * @param bitmap
     * @param isRecycled
     * @return
     */
    public static Bitmap flipBitmap(Bitmap bitmap, boolean isRecycled) {
        return flipBitmap(bitmap, true, false, isRecycled);
    }

    /**
     * 翻转图片
     *
     * @param bitmap
     * @param flipX
     * @param flipY
     * @param isRecycled
     * @return
     */
    public static Bitmap flipBitmap(Bitmap bitmap, boolean flipX, boolean flipY, boolean isRecycled) {
        if (bitmap == null) {
            return null;
        }
        Matrix matrix = new Matrix();
        matrix.setScale(flipX ? -1 : 1, flipY ? -1 : 1);
        matrix.postTranslate(bitmap.getWidth(), 0);
        Bitmap result = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(),
                bitmap.getHeight(), matrix, false);
        if (isRecycled && bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
            bitmap = null;
        }
        return result;
    }

    /**
     * 裁剪
     *
     * @param bitmap
     * @param x
     * @param y
     * @param width
     * @param height
     * @param isRecycled
     * @return
     */
    public static Bitmap cropBitmap(Bitmap bitmap, int x, int y, int width, int height, boolean isRecycled) {

        if (bitmap == null) {
            return null;
        }
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        // 保证裁剪区域
        if ((w - x) < width || (h - y) < height) {
            return null;
        }
        Bitmap result = Bitmap.createBitmap(bitmap, x, y, width, height, null, false);
        if (!bitmap.isRecycled() && isRecycled) {
            bitmap.recycle();
            bitmap = null;
        }
        return result;
    }

    /**
     * 获取Exif参数
     *
     * @param path
     * @param bundle
     * @return
     */
    public static boolean loadExifAttributes(String path, Bundle bundle) {
        ExifInterface exifInterface;
        try {
            exifInterface = new ExifInterface(path);
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
        for (String tag : EXIF_TAGS) {
            bundle.putString(tag, exifInterface.getAttribute(tag));
        }
        return true;
    }

    /**
     * 保存Exif属性
     *
     * @param path
     * @param bundle
     * @return 是否保存成功
     */
    public static boolean saveExifAttributes(String path, Bundle bundle) {
        ExifInterface exif;
        try {
            exif = new ExifInterface(path);
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        for (String tag : EXIF_TAGS) {
            if (bundle.containsKey(tag)) {
                exif.setAttribute(tag, bundle.getString(tag));
            }
        }
        try {
            exif.saveAttributes();
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        return true;
    }


    public static boolean isImageFile(String filePath) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(filePath, options);
        return options.outWidth != -1;
    }

    public static void recycleBitmap(Bitmap... bitmaps) {
        for (Bitmap bitmap : bitmaps) {
            if (bitmap != null && !bitmap.isRecycled()) {
                bitmap.recycle();
            }
        }
    }

    private static String getImagePath(Activity activity, Uri uri) {
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = activity.managedQuery(uri, projection, null, null, null);
        int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        return cursor.getString(columnIndex);
    }

    public static Bitmap loadFromPath(Activity activity, int id, int width, int height) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        InputStream is = activity.getResources().openRawResource(id);
        options.inSampleSize = calculateInSampleSize(options, width, height);
        options.inJustDecodeBounds = false;
        return zoomImage(BitmapFactory.decodeStream(is), width, height);
    }

    public static Bitmap loadFromPath(Activity activity, Uri uri, int width, int height) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;

        String path = getImagePath(activity, uri);
        BitmapFactory.decodeFile(path, options);
        options.inSampleSize = calculateInSampleSize(options, width, height);
        options.inJustDecodeBounds = false;

        Bitmap bitmap = zoomImage(BitmapFactory.decodeFile(path, options), width, height);
        return rotateBitmap(bitmap, getRotationAngle(path));
    }

    public static Bitmap loadFromPath(String path, int width, int height) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;

        BitmapFactory.decodeFile(path, options);
        options.inSampleSize = calculateInSampleSize(options, width, height);
        options.inJustDecodeBounds = false;

        Bitmap bitmap = zoomImage(BitmapFactory.decodeFile(path, options), width, height);
        return rotateBitmap(bitmap, getRotationAngle(path));
    }

    private static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int width = options.outWidth;
        final int height = options.outHeight;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            // Calculate height and required height scale.
            final int heightRatio = Math.round((float) height / (float) reqHeight);
            // Calculate width and required width scale.
            final int widthRatio = Math.round((float) width / (float) reqWidth);
            // Take the larger of the values.
            inSampleSize = Math.max(heightRatio, widthRatio);
        }
        return inSampleSize;
    }

    // Scale pictures to screen width.
    public static Bitmap zoomImage(Bitmap imageBitmap, int targetWidth, int maxHeight) {
        float scaleFactor =
                Math.max(
                        (float) imageBitmap.getWidth() / (float) targetWidth,
                        (float) imageBitmap.getHeight() / (float) maxHeight);

        return Bitmap.createScaledBitmap(
                imageBitmap,
                (int) (imageBitmap.getWidth() / scaleFactor),
                (int) (imageBitmap.getHeight() / scaleFactor),
                true);
    }

    public static Bitmap changeBitmapSize(Bitmap bitmap, int targetWidth, int targetHeight) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        float scaleWidth = ((float) targetWidth) / width;
        float scaleHeight = ((float) targetHeight) / height;

        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);

        bitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
        bitmap.getWidth();
        bitmap.getHeight();
        return bitmap;
    }

    /**
     * Get the rotation angle of the photo.
     *
     * @param path photo path.
     * @return angle.
     */
    public static int getRotationAngle(String path) {
        int rotation = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotation = ROTATE_90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotation = ROTATE_180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotation = ROTATE_270;
                    break;
                default:
                    break;
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to get rotation: " + e.getMessage());
        }
        return rotation;
    }

    public static Bitmap rotateBitmap(Bitmap bitmap, int angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        Bitmap result = null;
        try {
            result = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        } catch (OutOfMemoryError e) {
            Log.e(TAG, "Failed to rotate bitmap: " + e.getMessage());
        }
        if (result == null) {
            return bitmap;
        }
        return result;
    }

    public static Bitmap getBitmapFormUri(Activity ac, Uri uri) {
        Bitmap bitmap = null;
        try {
            InputStream input = ac.getContentResolver().openInputStream(uri);
            BitmapFactory.Options onlyBoundsOptions = new BitmapFactory.Options();
            onlyBoundsOptions.inJustDecodeBounds = true;
            onlyBoundsOptions.inDither = true;//optional
            onlyBoundsOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;//optional
            BitmapFactory.decodeStream(input, null, onlyBoundsOptions);
            input.close();
            int originalWidth = onlyBoundsOptions.outWidth;
            int originalHeight = onlyBoundsOptions.outHeight;
            if ((originalWidth == -1) || (originalHeight == -1)) {
                return null;
            }
            float hh = NUM_1920;
            float ww = NUM_1080;
            int be = 1;
            if (originalWidth > originalHeight && originalWidth > ww) {
                be = (int) (originalWidth / ww);
            } else if (originalWidth < originalHeight && originalHeight > hh) {
                be = (int) (originalHeight / hh);
            }
            if (be <= 0) {
                be = 1;
            }
            BitmapFactory.Options bitmapOptions = new BitmapFactory.Options();
            bitmapOptions.inSampleSize = be;
            bitmapOptions.inDither = true;//optional
            bitmapOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;//optional
            input = ac.getContentResolver().openInputStream(uri);
            bitmap = BitmapFactory.decodeStream(input, null, bitmapOptions);
            input.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
        return compressImage(bitmap);
    }


    public static Bitmap compressImage(Bitmap image) {
        if (image != null) {
            ByteArrayOutputStream bas = new ByteArrayOutputStream();
            image.compress(Bitmap.CompressFormat.JPEG, NUM_HUNDRED, bas);
            int options = NUM_HUNDRED;
            while (bas.toByteArray().length / 1024 > NUM_HUNDRED) {
                bas.reset();
                image.compress(Bitmap.CompressFormat.JPEG, options, bas);
                options -= 10;
            }
            ByteArrayInputStream isBm = new ByteArrayInputStream(bas.toByteArray());
            return BitmapFactory.decodeStream(isBm, null, null);
        } else {
            return null;
        }
    }

    public static File getFileFromMediaUri(Context ac, Uri uri) {
        if (uri.getScheme().compareTo("content") == 0) {
            ContentResolver cr = ac.getContentResolver();
            Cursor cursor = cr.query(uri, null, null, null, null);
            if (cursor != null) {
                cursor.moveToFirst();
                String filePath = cursor.getString(cursor.getColumnIndex("_data"));
                cursor.close();
                if (filePath != null) {
                    return new File(filePath);
                }
            }
        } else if (uri.getScheme().compareTo("file") == 0) {
            return new File(uri.toString().replace("file://", ""));
        }
        return null;
    }

    public static int getBitmapDegree(String path) {
        int degree = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degree = ROTATE_90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degree = ROTATE_180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degree = ROTATE_270;
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return degree;
    }

    public static Bitmap rotateBitmapByDegree(Bitmap bm, int degree) {
        Bitmap returnBm = null;
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);
        try {
            returnBm = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), matrix, true);
        } catch (OutOfMemoryError e) {
            e.printStackTrace();
        }
        if (returnBm == null) {
            returnBm = bm;
        }
        if (bm != returnBm) {
            bm.recycle();
        }
        return returnBm;
    }


    // Save the picture to the system album and refresh it.
    public static void saveToAlbum(final Context context, Bitmap bitmap) {
        File file = null;
        String fileName = System.currentTimeMillis() + ".jpg";
        File root = new File(StorageUtils.ABSOLUTE_FILE, context.getPackageName());
        File dir = new File(root, "image");
        if (dir.mkdirs() || dir.isDirectory()) {
            file = new File(dir, fileName);
        }
        FileOutputStream os = null;
        try {
            os = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, NUM_HUNDRED, os);
            os.flush();

        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        } finally {
            try {
                if (os != null) {
                    os.close();
                }
            } catch (IOException e) {
                Log.e(TAG, e.getMessage());
            }
        }
        if (file == null) {
            return;
        }
        String path = null;
        try {
            path = file.getCanonicalPath();
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        }
        MediaScannerConnection.scanFile(context, new String[]{path}, null,
                (path1, uri) -> {
                    Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                    mediaScanIntent.setData(uri);
                    context.sendBroadcast(mediaScanIntent);
                });

    }

    /**
     * 将Base64编码转换为图片
     *
     * @param base64Str
     * @param path
     * @return true
     */
    public static boolean base64ToFile(String base64Str, String path) {
        byte[] data = Base64.decode(base64Str, Base64.NO_WRAP);
        for (int i = 0; i < data.length; i++) {
            if (data[i] < 0) {
                //调整异常数据
                data[i] += NUM_256;
            }
        }
        OutputStream os;
        try {
            os = new FileOutputStream(path);
            os.write(data);
            os.flush();
            os.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }
}
