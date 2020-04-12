#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <initializer_list> //列表初始化，用于可变参数函数
#include <thread> // 多线程
#include <mutex>  //互斥锁
#include <functional>
#include <condition_variable> // 条件变量/

#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"

using namespace std;

typedef unsigned long ulong;

namespace myslam {
namespace backend {

typedef unsigned long ulong;
//    typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem {
public:

    /**
     * 问题的类型
     * SLAM问题还是通用的问题
     *
     * 如果是SLAM问题那么pose和landmark是区分开的，Hessian以稀疏方式存储
     * SLAM问题只接受一些特定的Vertex和Edge
     * 如果是通用问题那么hessian是稠密的，除非用户设定某些vertex为marginalized
     */
    enum class ProblemType {
        SLAM_PROBLEM,
        GENERIC_PROBLEM
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Problem(ProblemType problemType);

    ~Problem();

    bool AddVertex(std::shared_ptr<Vertex> vertex);

    /**
     * remove a vertex
     * @param vertex_to_remove
     */
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);

    bool AddEdge(std::shared_ptr<Edge> edge);

    bool RemoveEdge(std::shared_ptr<Edge> edge);

    /**
     * 取得在优化中被判断为outlier部分的边，方便前端去除outlier
     * @param outlier_edges
     */
    void GetOutlierEdges(std::vector<std::shared_ptr<Edge>> &outlier_edges);

    /**
     * @brief 使用非线性求解器求解
     * 
     * @param iterations 最大迭代次数 
     * @param type 采用的求解器类型，0-LM，1-DogLeg
     * @return true 
     * @return false 
     */
    bool Solve(int type, int iterations = 10);
    /**
     * @brief 采用LM策略求解
     * 
     * @param iterations 
     * @return true 
     * @return false 
     */
    bool SolveLM(int iterations = 10);
    /**
     * @brief 采用DogLeg策略求解
     * 
     * @param iterations 
     * @return true 
     * @return false 
     */
    bool SolveDogLeg(int iterations = 10);

    /// 边缘化一个frame和以它为host的landmark
    bool Marginalize(std::shared_ptr<Vertex> frameVertex,
                     const std::vector<std::shared_ptr<Vertex>> &landmarkVerticies);

    bool Marginalize(const std::shared_ptr<Vertex> frameVertex);
    bool Marginalize(const std::vector<std::shared_ptr<Vertex> > frameVertex,int pose_dim);

    MatXX GetHessianPrior(){ return H_prior_;}
    VecX GetbPrior(){ return b_prior_;}
    VecX GetErrPrior(){ return err_prior_;}
    MatXX GetJtPrior(){ return Jt_prior_inv_;}

    void SetHessianPrior(const MatXX& H){H_prior_ = H;}
    void SetbPrior(const VecX& b){b_prior_ = b;}
    void SetErrPrior(const VecX& b){err_prior_ = b;}
    void SetJtPrior(const MatXX& J){Jt_prior_inv_ = J;}

    void ExtendHessiansPriorSize(int dim);

    //test compute prior
    void TestComputePrior();
    // 返回求解器耗时
    double getSolverCost(){return solve_cost_;}

private:

    /// Solve的实现，解通用问题
    bool SolveGenericProblem(int iterations);

    /// Solve的实现，解SLAM问题
    bool SolveSLAMProblem(int iterations);

    /// 设置各顶点的ordering_index
    void SetOrdering();

    /// set ordering for new vertex in slam problem
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    /// 构造Hessian矩阵，总函数
    void MakeHessian();
    /// 构造大H矩阵，单线程
    void MakeHessianSingle();
    /// 构造大H矩阵，多线程
    void MakeHessianMulti();
    // 每个线程中用于计算Hessian的函数
    void thdCalcHessian(int thd_id, int thd_num);
    /// 构造大矩阵，采用OpenMP
    void MakeHessianOpenMP();

    /// schur求解SBA
    void SchurSBA();

    /// 解线性方程
    void SolveLinearSystem();
    /// 求解Dogleg步长
    void SolveDogLegStep();
    /**
     * @brief 使用Schur complement加速求解线性方程组Hdelta_x=b（针对类slam问题，即存在主对角线上存在分块对角矩阵）
     * 
     * @param Hessian： H矩阵 
     * @param b ：b
     * @param x ：x
     * @param reserve_num：保留的变量数目 
     * @param schur_num ：schur的变量数目
     * @param schur_vertices：schur变量的顶点map, std::map<unsigned long, std::shared_ptr<Vertex>>
     * @param lambda ：是否添加阻尼
     */
    void SolveLinearWithSchur(MatXX & Hessian, VecX &b, VecX & delta_x, int reserve_size, int schur_size,
                        std::map<unsigned long, std::shared_ptr<Vertex>> & schur_vertices,  double lambda = 0.);

    /// 更新状态变量
    void UpdateStates();

    void RollbackStates(); // 有时候 update 后残差会变大，需要退回去，重来

    /// 计算并更新Prior部分
    void ComputePrior();

    /// 判断一个顶点是否为Pose顶点
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    /// 判断一个顶点是否为landmark顶点
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    /// 在新增顶点后，需要调整几个hessian的大小
    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

    /// 检查ordering是否正确
    bool CheckOrdering();

    void LogoutVectorSize();

    /// 获取某个顶点连接到的边
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    /// Levenberg
    /// 计算LM算法的初始Lambda
    void ComputeLambdaInitLM();
    /// 计算DogLeg算法的初始误差及半径
    void ComputeRadiusInitDogLeg();

    /// Hessian 对角线加上或者减去  Lambda
    void AddLambdatoHessianLM();

    void RemoveLambdaHessianLM();

    /// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
    bool IsGoodStepInLM();
    /// DogLeg 算法中用于判断上次迭代效果及信赖域半径如何缩放
    bool IsGoodStepInDogLeg();
    /// PCG 迭代线性求解器
    VecX PCGSolver(const MatXX &A, const VecX &b, int maxIter);
    /// 存储时间
    void saveCost(initializer_list<double> times);

    double currentChi_;
    double solve_cost_; // 求解器每次迭代耗时
    // DogLeg 相关参数
    double currentRadius_;
    double stopThresholdDogLeg_;
    VecX h_gn_; // 高斯牛顿法步长
    VecX h_sd_; // 最速下降法步长
    VecX h_dl_; // DogLeg步长
    double alpha_ = 0.0;
    double beta_ = 0.0;
    // LM相关参数
    double currentLambda_;
    double stopThresholdLM_;    // LM 迭代退出阈值条件
    double ni_;                 //控制 Lambda 缩放大小
    double L_up_ = 6.;
    double L_down_ = 5.;

    ProblemType problemType_;

    /// 整个信息矩阵
    MatXX Hessian_;
    MatXX diagHessian_;
    VecX b_;
    VecX delta_x_;

    /// 用于多线程计算的变量
    MatXX multi_H_;
    VecX multi_b_;
    mutex m_hessian_;
    vector<unsigned long> edges_idx_;

    /// 先验部分信息
    MatXX H_prior_;
    VecX b_prior_;
    VecX b_prior_backup_;
    VecX err_prior_backup_;

    MatXX Jt_prior_inv_;
    VecX err_prior_;

    /// SBA的Pose部分
    MatXX H_pp_schur_;
    VecX b_pp_schur_;
    // Heesian 的 Landmark 和 pose 部分
    MatXX H_pp_;
    VecX b_pp_;
    MatXX H_ll_;
    VecX b_ll_;

    /// all vertices
    HashVertex verticies_;

    /// all edges
    HashEdge edges_;

    /// 由vertex id查询edge
    HashVertexIdToEdge vertexToEdge_;

    /// Ordering related
    ulong ordering_poses_ = 0;
    ulong ordering_landmarks_ = 0;
    ulong ordering_generic_ = 0;
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;        // 以ordering排序的pose顶点
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;    // 以ordering排序的landmark顶点

    // verticies need to marg. <Ordering_id_, Vertex>
    HashVertex verticies_marg_;

    bool bDebug = false;
    double t_hessian_cost_ = 0.0;
    double t_PCGsovle_cost_ = 0.0;
};

}
}

#endif
