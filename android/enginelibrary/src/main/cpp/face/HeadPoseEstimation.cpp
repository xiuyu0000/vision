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

#include <android/log.h>
#include "HeadPoseEstimation.h"

#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)


/*
 * euler_angle[0]: X, means Pitch.
 * euler_angle[1]: Y, means Yaw.
 * euler_angle[2]: Z, means Roll.
 */
bool HeadPoseEstimation(cv::Mat img, const float *face, cv::Mat &euler_angle) {
    double focal = img.cols; // Approximate focal length.
    cv::Point2d center = cv::Point2d(img.cols / 2, img.rows / 2);
    cv::Mat cam_matrix = (cv::Mat_<double>(3, 3) << focal, 0, center.x, 0, focal, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

    //fill in 3D ref 14 points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    std::vector<cv::Point3d> object_pts;
    object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
    object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
    object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
    object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
    object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
    object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
    object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
    object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
    object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

    //2D ref points(image coordinates), referenced from detected facial feature
    std::vector<cv::Point2d> image_pts;

    //result
    cv::Mat rotation_vec;                           //3 x 1
    cv::Mat rotation_mat;                           //3 x 3 R
    cv::Mat translation_vec;                        //3 x 1 T
    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    // cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

    //reproject 3D （8）points world coordinate axis to verify result pose
    std::vector<cv::Point3d> reprojectsrc;
    reprojectsrc.push_back(cv::Point3d(10.0, 10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, 10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, -10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, -10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, 10.0));

    //reprojected 2D points
    std::vector<cv::Point2d> reproject_dst;
    reproject_dst.resize(8);

    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

    //fill in 2D ref points（14Point), annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
    image_pts.push_back(cv::Point2d(face[34], face[35])); //#17 left brow left corner
    image_pts.push_back(cv::Point2d(face[42], face[43])); //#21 left brow right corner
    image_pts.push_back(cv::Point2d(face[44], face[45])); //#22 right brow left corner
    image_pts.push_back(cv::Point2d(face[52], face[53])); //#26 right brow right corner
    image_pts.push_back(cv::Point2d(face[72], face[73])); //#36 left eye left corner
    image_pts.push_back(cv::Point2d(face[78], face[79])); //#39 left eye right corner
    image_pts.push_back(cv::Point2d(face[84], face[85])); //#42 right eye left corner
    image_pts.push_back(cv::Point2d(face[90], face[91])); //#45 right eye right corner
    image_pts.push_back(cv::Point2d(face[62], face[63])); //#31 nose left corner
    image_pts.push_back(cv::Point2d(face[70], face[71])); //#35 nose right corner
    image_pts.push_back(cv::Point2d(face[96], face[97])); //#48 mouth left corner
    image_pts.push_back(cv::Point2d(face[108], face[109])); //#54 mouth right corner
    image_pts.push_back(cv::Point2d(face[114], face[115])); //#57 mouth central bottom corner
    image_pts.push_back(cv::Point2d(face[16], face[17]));   //#8 chin corner

    //calc pose
    cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

    //reproject
    cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reproject_dst);

    //calc euler angle
    cv::Rodrigues(rotation_vec, rotation_mat);
    cv::hconcat(rotation_mat, translation_vec, pose_mat);
    cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

    //show angle result
    // std::cout << "angle X: " << std::setprecision(3) << euler_angle.at<double>(0);
    // std::cout << "angle Y: " << std::setprecision(3) << euler_angle.at<double>(1);
    // std::cout << "angle Z: " << std::setprecision(3) << euler_angle.at<double>(2);

    image_pts.clear();
    return true;
}