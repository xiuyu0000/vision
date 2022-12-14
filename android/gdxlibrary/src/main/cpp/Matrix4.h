/*
 * Copyright 2022
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

#include <jni.h>
/* Header for class com_badlogic_gdx_math_Matrix4 */

#ifndef _Included_com_badlogic_gdx_math_Matrix4
#define _Included_com_badlogic_gdx_math_Matrix4
#ifdef __cplusplus
extern "C" {
#endif
#undef com_badlogic_gdx_math_Matrix4_serialVersionUID
#define com_badlogic_gdx_math_Matrix4_serialVersionUID -2717655254359579617LL
#undef com_badlogic_gdx_math_Matrix4_M00
#define com_badlogic_gdx_math_Matrix4_M00 0L
#undef com_badlogic_gdx_math_Matrix4_M01
#define com_badlogic_gdx_math_Matrix4_M01 4L
#undef com_badlogic_gdx_math_Matrix4_M02
#define com_badlogic_gdx_math_Matrix4_M02 8L
#undef com_badlogic_gdx_math_Matrix4_M03
#define com_badlogic_gdx_math_Matrix4_M03 12L
#undef com_badlogic_gdx_math_Matrix4_M10
#define com_badlogic_gdx_math_Matrix4_M10 1L
#undef com_badlogic_gdx_math_Matrix4_M11
#define com_badlogic_gdx_math_Matrix4_M11 5L
#undef com_badlogic_gdx_math_Matrix4_M12
#define com_badlogic_gdx_math_Matrix4_M12 9L
#undef com_badlogic_gdx_math_Matrix4_M13
#define com_badlogic_gdx_math_Matrix4_M13 13L
#undef com_badlogic_gdx_math_Matrix4_M20
#define com_badlogic_gdx_math_Matrix4_M20 2L
#undef com_badlogic_gdx_math_Matrix4_M21
#define com_badlogic_gdx_math_Matrix4_M21 6L
#undef com_badlogic_gdx_math_Matrix4_M22
#define com_badlogic_gdx_math_Matrix4_M22 10L
#undef com_badlogic_gdx_math_Matrix4_M23
#define com_badlogic_gdx_math_Matrix4_M23 14L
#undef com_badlogic_gdx_math_Matrix4_M30
#define com_badlogic_gdx_math_Matrix4_M30 3L
#undef com_badlogic_gdx_math_Matrix4_M31
#define com_badlogic_gdx_math_Matrix4_M31 7L
#undef com_badlogic_gdx_math_Matrix4_M32
#define com_badlogic_gdx_math_Matrix4_M32 11L
#undef com_badlogic_gdx_math_Matrix4_M33
#define com_badlogic_gdx_math_Matrix4_M33 15L
/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    mul
 * Signature: ([F[F)V
 */
JNIEXPORT void JNICALL Java_com_badlogic_gdx_math_Matrix4_mul
        (JNIEnv *, jclass, jfloatArray, jfloatArray);

/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    mulVec
 * Signature: ([F[F)V
 */
JNIEXPORT void JNICALL Java_com_badlogic_gdx_math_Matrix4_mulVec___3F_3F
        (JNIEnv *, jclass, jfloatArray, jfloatArray);

/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    mulVec
 * Signature: ([F[FIII)V
 */
JNIEXPORT void JNICALL Java_com_badlogic_gdx_math_Matrix4_mulVec___3F_3FIII
        (JNIEnv *, jclass, jfloatArray, jfloatArray, jint, jint, jint);

/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    prj
 * Signature: ([F[F)V
 */
JNIEXPORT void JNICALL Java_com_badlogic_gdx_math_Matrix4_prj___3F_3F
        (JNIEnv *, jclass, jfloatArray, jfloatArray);

/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    prj
 * Signature: ([F[FIII)V
 */
JNIEXPORT void JNICALL Java_com_badlogic_gdx_math_Matrix4_prj___3F_3FIII
        (JNIEnv *, jclass, jfloatArray, jfloatArray, jint, jint, jint);

/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    rot
 * Signature: ([F[F)V
 */
JNIEXPORT void JNICALL Java_com_badlogic_gdx_math_Matrix4_rot___3F_3F
        (JNIEnv *, jclass, jfloatArray, jfloatArray);

/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    rot
 * Signature: ([F[FIII)V
 */
JNIEXPORT void JNICALL Java_com_badlogic_gdx_math_Matrix4_rot___3F_3FIII
        (JNIEnv *, jclass, jfloatArray, jfloatArray, jint, jint, jint);

/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    inv
 * Signature: ([F)Z
 */
JNIEXPORT jboolean JNICALL Java_com_badlogic_gdx_math_Matrix4_inv
        (JNIEnv *, jclass, jfloatArray);

/*
 * Class:     com_badlogic_gdx_math_Matrix4
 * Method:    det
 * Signature: ([F)F
 */
JNIEXPORT jfloat JNICALL Java_com_badlogic_gdx_math_Matrix4_det
        (JNIEnv *, jclass, jfloatArray);

#ifdef __cplusplus
}
#endif
#endif
