/**
 * @file rotation.cpp
 * @author yuntian li (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2020-02-18
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <sophus/so3.hpp>
using namespace std;

int main(int argc, char** argv){
    Eigen::AngleAxisd rot_vec(M_PI / 2, Eigen::Vector3d(0, 0, 1));

    Eigen::Vector3d omega; //声明角速度矢量
    omega = {0.01, 0.02, 0.03}; //赋值
    // ================ 李代数更新 =================== //
    Eigen::Matrix3d R0 = rot_vec.toRotationMatrix();
    Sophus::SO3d SO3_R0(R0); //由R0构造SO3
    Sophus::SO3d SO3_R1 = SO3_R0 * Sophus::SO3d::exp(omega);
    cout << "李代数更新前的so3：\n" << SO3_R0.log().transpose() << endl;
    cout << "李代数更新后的so3：\n" << SO3_R1.log().transpose() << endl;
    // ================ 四元数更新 =================== //
    Eigen::Quaterniond q1, dq; //声明原四元数，更新后四元数及更新四元数

    Eigen::Quaterniond q0(rot_vec); //
    dq.w() = 1; dq.vec() = omega / 2; //为更新四元数赋值
    q1 = q0 * dq;
    Sophus::SO3d SO3_q1(q1);

    cout << "四元数更新的so3：\n" << SO3_q1.log().transpose() << endl;
    return 0;
}