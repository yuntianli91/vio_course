
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include <Eigen/StdVector>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 1;
string sData_path = "../../vio_data_simulation/bin/";
string sConfig_path = "../config/";

std::shared_ptr<System> pSystem;

void loadSimFeatures(const double dStampNSec, const string sImgFileName, const double prev_frame_time, const vector<Vector2d> &prev_un_pts,  
                    vector<int> &pt_ids, vector<Vector2d> &cur_un_pts, vector<Vector2d> &cur_pts, vector<Vector2d> &cur_velocity);
/**
 * @brief 发布仿真IMU数据
 * 
 */
void PubImuSimData()
{
	string sImu_data_file = sData_path + "imu_pose_noise.txt";
	cout << "1 PubImuData start sImu_data_filea: " << sImu_data_file << endl;
	ifstream fsImu;
	fsImu.open(sImu_data_file.c_str());
	if (!fsImu.is_open())
	{
		cerr << "Failed to open imu file! " << sImu_data_file << endl;
		return;
	}

	std::string sImu_line;
	double dStampNSec = 0.0;
    Quaterniond quat; //ground truth轨迹四元数
    Vector3d t; // ground truth 位置
	Vector3d vAcc;
	Vector3d vGyr;
	// 存储轨迹真值
	ofstream ofs_gt;
	system("rm ./gt_output.txt");
	ofs_gt.open("./gt_output.txt", fstream::app|fstream::out);
	// 清空轨迹真值容器	
	pSystem->pos_gt_.clear();
	while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
	{
		std::istringstream ssImuData(sImu_line);
        
		ssImuData >> dStampNSec 
                >> quat.w() >> quat.x() >> quat.y() >> quat.z()
                >> t.x() >> t.y() >> t.z()
                >> vGyr.x() >> vGyr.y() >> vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
		// cout << "Imu t: " << fixed << dStampNSec << " gyr: " << vGyr.transpose() << " acc: " << vAcc.transpose() << endl;
		pSystem->PubImuData(dStampNSec, vGyr, vAcc);
		pSystem->pos_gt_.push_back(t);
		// 以TUM格式存储gt
		ofs_gt << fixed << dStampNSec << " "
			<< t.x() << " " << t.y() << " " << t.z() << " "
			<< quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << endl;
		usleep(4500*nDelayTimes); //时间需要调整
	}

	fsImu.close();
}

/**
 * @brief 发布仿真图像特征数据
 * 
 */
void PubImageSimData()
{
	string sImage_file = sData_path + "cam_pose.txt";

	cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		cerr << "Failed to open cam pose file! " << sImage_file << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;
	
	// cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
	int img_cnt = 0;
    double prev_frame_time = 0.;
    vector<Vector2d> prev_un_pts; // 上一帧归一化平面特征点坐标容器
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec;
		// cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
		sImgFileName = sData_path + "keyframe/all_points_" + to_string(img_cnt) + ".txt";
        // 读取特征点数据
        vector<int> pt_ids;
        vector<Vector2d> cur_un_pts; // 归一化平面特征点坐标容器
        vector<Vector2d> cur_pts; //像素平面特征点坐标容器
        vector<Vector2d> cur_velocity; // 归一化平面特征点速度容器
        loadSimFeatures(dStampNSec, sImgFileName, prev_frame_time, prev_un_pts, 
                        pt_ids, cur_un_pts, cur_pts, cur_velocity);

        // 更新各容器
        img_cnt++;
        prev_un_pts = cur_un_pts;
        prev_frame_time = dStampNSec;
        pSystem->PubSimFeature(dStampNSec, pt_ids, cur_un_pts, cur_pts, cur_velocity);
		usleep(50000*nDelayTimes);
	}
	fsImage.close();
}

/**
 * @brief 从仿真文件中读取每一帧的所有特征点
 * 
 * @param dStampNSec ： 当前帧时间戳 
 * @param sImgFileName ： 当前帧特征点文件名
 * @param cur_un_pts ： 当前点归一化坐标容器
 * @param cur_pts ： 当前点像素坐标容器
 * @param cur_velocity ： 当前点归一化像平面速度容器
 */
void loadSimFeatures(const double dStampNSec, const string sImgFileName, const double prev_frame_time, const vector<Vector2d> &prev_un_pts,  
                    vector<int> &pt_ids, vector<Vector2d> &cur_un_pts, vector<Vector2d> &cur_pts, vector<Vector2d> &cur_velocity)
{
    ifstream fsFeature;
	fsFeature.open(sImgFileName.c_str());
	if (!fsFeature.is_open())
	{
		cerr << "Failed to open image file! " << sImgFileName << endl;
		return;
	}

	std::string sFeature_line;
    int feature_id = 0;
    // 内参矩阵
    Eigen::Matrix3d K;
    K << 460.0, 0, 320,
		0, 460.0, 320,
		0, 0, 0;
    while(getline(fsFeature, sFeature_line) && !sFeature_line.empty()){
        istringstream ssFeatureData(sFeature_line);
        // 前4维齐次坐标不用
        double tmp;
        for(int i=0;i<4;i++){ssFeatureData >> tmp;}
        // 特征点ids
        pt_ids.push_back(feature_id);
        // 归一化坐标
        Vector2d un_pt;
        ssFeatureData >> un_pt.x() >> un_pt.y();
        cur_un_pts.push_back(un_pt);
        // 像素坐标
        Vector2d uv_pt;
        uv_pt.x() = K(0,0) * un_pt.x() + K(0,2);
        uv_pt.y() = K(1,1) * un_pt.y() + K(1,2);
        cur_pts.push_back(uv_pt);
        // 速度
        if(prev_un_pts.empty()){
            cur_velocity.push_back(Vector2d(0.,0.));
        }
        else{
            Vector2d pt_velocity;
            pt_velocity.x() = (un_pt.x()-prev_un_pts[feature_id].x()) / (dStampNSec - prev_frame_time);
            pt_velocity.y() = (un_pt.y()-prev_un_pts[feature_id].y()) / (dStampNSec - prev_frame_time);
        
            cur_velocity.push_back(pt_velocity);
        }
        //更新id，读取下一个点    
        feature_id++;
    }
}

#ifdef __APPLE__
// support for MacOS
void DrawIMGandGLinMainThrd(){
	string sImage_file = sConfig_path + "MH_05_cam0.txt";

	cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		cerr << "Failed to open image file! " << sImage_file << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;

	pSystem->InitDrawGL();
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		// cout << "Image t : " << fixed << dStampNSec << " Name: " << sImgFileName << endl;
		string imagePath = sData_path + "cam0/data/" + sImgFileName;

		Mat img = imread(imagePath.c_str(), 0);
		if (img.empty())
		{
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		//pSystem->PubImageData(dStampNSec / 1e9, img);
		cv::Mat show_img;
		cv::cvtColor(img, show_img, CV_GRAY2RGB);
		if (SHOW_TRACK)
		{
			for (unsigned int j = 0; j < pSystem->trackerData[0].cur_pts.size(); j++)
			{
				double len = min(1.0, 1.0 *  pSystem->trackerData[0].track_cnt[j] / WINDOW_SIZE);
				cv::circle(show_img,  pSystem->trackerData[0].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
			}

			cv::namedWindow("IMAGE", CV_WINDOW_AUTOSIZE);
			cv::imshow("IMAGE", show_img);
		  // cv::waitKey(1);
		}

		pSystem->DrawGLFrame();
		usleep(50000*nDelayTimes);
	}
	fsImage.close();

} 
#endif

int main(int argc, char **argv)
{
	// if(argc != 3)
	// {
	// 	cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n" 
	// 		<< "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/"<< endl;
	// 	return -1;
	// }
	// sData_path = argv[1];
	// sConfig_path = argv[2];

    string sConfig_file = sConfig_path + "simulation_config.yaml";
	pSystem.reset(new System(sConfig_file));
	
	std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);
		
	// sleep(5);
	std::thread thd_PubImuData(PubImuSimData);

	std::thread thd_PubImageData(PubImageSimData);

#ifdef __linux__	
	std::thread thd_Draw(&System::Draw, pSystem);
#elif __APPLE__
	DrawIMGandGLinMainThrd();
#endif

	thd_PubImuData.join();
	thd_PubImageData.join();

	// thd_BackEnd.join();
	// thd_Draw.join();

	cout << "main end... see you ..." << endl;
	return 0;
}
