/**
 * @file triangulate.cpp
 * @author yuntian li (yuntianlee91@hotmail.com)
 * @brief code for vio_cource Ch6, modified by yuntian li. triangulated one landmark by several cam pose
 * @version 0.1
 * @date 2020-03-18
 * 
 * @copyright Copyright (c) 2020
 * 
 */
//
// Created by hyj on 18-11-11.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
using namespace std;

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};

int main(int argc, char** argv)
{
    for(int pixel_noise = 0; pixel_noise <= 60; pixel_noise+=3){

    // 10帧相机位姿
    int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R,t));
    }

    // 随机数生成 1 个 三维特征点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);
    // 这个特征从第三帧相机开始被观测，i=3
    int start_frame_id = 3;
    int end_frame_id = poseNums;
    
    cout << "current noise is " << pixel_noise << " pixel.\n";
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();
        // 生成随机误差（误差大于0.2个像素就会使后续估计出现较大误差，故暂时放弃）
        // 像素误差要转化为归一化平面误差
        std::normal_distribution<double> uv_rand(0, pixel_noise / 460.);
        Eigen::Vector2d uv_noise;
        uv_noise.x() = uv_rand(generator);
        uv_noise.y() = uv_rand(generator);
        camera_pose[i].uv = Eigen::Vector2d(x/z,y/z) + uv_noise;
    }
    
    /// TODO::homework; 请完成三角化估计深度的代码
    // 遍历所有的观测数据，并三角化
    Eigen::Vector3d P_est;           // 结果保存到这个变量
    P_est.setZero();
    /* your code begin */
    // 构建系数矩阵
    int frame_count = poseNums - start_frame_id;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * frame_count, 4) ;
    Eigen::Matrix<double, 3, 4> Tcw; // 位姿矩阵T^c_w;
    double sigma_34; // 最后两维度奇异值的比值。
    double estimated_error; // 估计误差
    ofstream ofs_sigma;
    ofs_sigma.open("./sigma.txt", fstream::app | fstream::out);
    ofstream ofs_result;
    ofs_result.open("./result.txt", fstream::app | fstream::out);
    for (int i=start_frame_id; i< poseNums; i++){
        Tcw.block<3, 3>(0, 0) = camera_pose[i].Rwc.transpose();
        Tcw.block<3, 1>(0, 3) = -camera_pose[i].Rwc.transpose() * camera_pose[i].twc;

        A.row(2 * (i-start_frame_id)) = camera_pose[i].uv.x() * Tcw.row(2) - Tcw.row(0);
        A.row(2 * (i-start_frame_id) + 1) = camera_pose[i].uv.y() * Tcw.row(2) - Tcw.row(1);
        // SVD分解
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix4d V = svd.matrixV();

        double scale = V.rightCols(1)(3); // 第四维归一化尺度
        P_est.x() = V.rightCols(1)(0) / scale;
        P_est.y() = V.rightCols(1)(1) / scale;
        P_est.z() = V.rightCols(1)(2) / scale;
     
        estimated_error = (P_est - Pw).norm();
        /* your code end */
        sigma_34 = svd.singularValues()(2) / svd.singularValues()(3);
        ofs_sigma << sigma_34 << " ";
        ofs_result << estimated_error << " ";
        // std::cout << "==========  loop " << i-2 << " ==========" << endl;
        // std::cout <<"sigular values: " << svd.singularValues().transpose() << std::endl;
        // std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
        // std::cout <<"your result: \n"<< P_est.transpose() <<std::endl;
    }
    ofs_sigma << endl;
    ofs_result << endl;

    ofs_sigma.close();
    ofs_result.close();
   // TODO:: 请如课程讲解中提到的判断三角化结果好坏的方式，绘制奇异值比值变化曲线
    }
    return 0;
}
