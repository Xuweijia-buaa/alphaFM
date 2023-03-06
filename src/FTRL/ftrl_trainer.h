#ifndef FTRL_TRAINER_H_
#define FTRL_TRAINER_H_

#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"
#include "../Lock/lock_pool.h"


struct trainer_option
{
    // 构造函数。各参数默认取值
    trainer_option() : k0(true), k1(true), factor_num(8), init_mean(0.0), init_stdev(0.1), w_alpha(0.05), w_beta(1.0), w_l1(0.1), w_l2(5.0),
               v_alpha(0.05), v_beta(1.0), v_l1(0.1), v_l2(5.0), model_format("txt"), initial_model_format("txt"),
               threads_num(1), b_init(false), force_v_sparse(false) {}
    string model_path, model_format, init_model_path, initial_model_format, model_number_type;
    double init_mean, init_stdev;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    int threads_num, factor_num;
    bool k0, k1, b_init, force_v_sparse;
    
    void parse_option(const vector<string>& args)
    {
        int argc = args.size();
        if(0 == argc) throw invalid_argument("invalid command\n");
        for(int i = 0; i < argc; ++i)
        {
            // 逐个解析传入的参数
            // 后边必须跟模型输出路径
            if(args[i].compare("-m") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_path = args[++i];
            }
            // 模型输出格式：bin/txt
            else if(args[i].compare("-mf") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_format = args[++i];
                if("bin" != model_format && "txt" != model_format)
                    throw invalid_argument("invalid command\n");
            }
            // 是否用bias,是否用一阶参数w1,二阶参数维度（）
            else if(args[i].compare("-dim") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                vector<string> strVec;
                string tmpStr = args[++i];  // 1,1,8
                utils::split_string(tmpStr, ',', &strVec);// 传入vector对象地址，用来存解析后的string。
                if(strVec.size() != 3)
                    throw invalid_argument("invalid command\n");
                k0 = 0 == stoi(strVec[0]) ? false : true;   // 是否用w0
                k1 = 0 == stoi(strVec[1]) ? false : true;   // 是否用wi
                factor_num = stoi(strVec[2]);               // vi的维度
            }
            // 特征向量的初始化方差
            else if(args[i].compare("-init_stdev") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_stdev = stod(args[++i]);
            }
            // 一阶参数（ftrl）的超参
            else if(args[i].compare("-w_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-w_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_beta = stod(args[++i]);
            }
            // 一阶参数的正则
            else if(args[i].compare("-w_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-w_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l2 = stod(args[++i]);
            }
            // 二阶参数(ftrl)的超参
            else if(args[i].compare("-v_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-v_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_beta = stod(args[++i]);
            }
            // 二阶参数的正则
            else if(args[i].compare("-v_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-v_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_l2 = stod(args[++i]);
            }
            // 多线程线程数
            else if(args[i].compare("-core") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                threads_num = stoi(args[++i]);
            }
            // 初始模型路径
            else if(args[i].compare("-im") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_model_path = args[++i];
                b_init = true; //if im field exits, that means b_init = true
            }
            // 初始模型格式
            else if(args[i].compare("-imf") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                initial_model_format = args[++i];
                if("bin" != initial_model_format && "txt" != initial_model_format)
                    throw invalid_argument("invalid command\n");
            }
            // w为0，是否强制把对应vi向量置为0
            else if(args[i].compare("-fvs") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                int fvs = stoi(args[++i]);
                force_v_sparse = (1 == fvs) ? true : false;
            }
            // 模型浮点数精度
            else if(args[i].compare("-mnt") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_number_type = args[++i];
            }
            else
            {
                throw invalid_argument("invalid command\n");
                break;
            }
        }
    }
};


// 用于训练的主类。继承自pc_task。是个模板类，可以是float、double
template<typename T>
class ftrl_trainer : public pc_task
{
public:
    ftrl_trainer(const trainer_option& opt);//构造函数
    virtual void run_task(vector<string>& dataBuffer);// 可重写的父类函数，默认仍是虚函数类型
    bool load_model(const string& modelPath, const string& modelFormat);    // 自己独有的函数
    bool output_model(const string& modelPath, const string& modelFormat);
    
private:
    void train(int y, const vector<pair<string, double> >& x);   //
    
private:
    ftrl_model<T>* pModel;              // 模型本身
    lock_pool* pLockPool;
    double w_alpha, w_beta, w_l1, w_l2; // 模型超参
    double v_alpha, v_beta, v_l1, v_l2;
    bool k0;
    bool k1;
    bool force_v_sparse;
};

// 初始化
// 模板函数，是这个trainer的初始化构造函。用入参初始化
template<typename T>
ftrl_trainer<T>::ftrl_trainer(const trainer_option& opt)
{
    w_alpha = opt.w_alpha;  // ftrl的超参数（对一阶参数）
    w_beta = opt.w_beta;
    w_l1 = opt.w_l1;
    w_l2 = opt.w_l2;
    v_alpha = opt.v_alpha;  // ftrl对向量参数的超参数
    v_beta = opt.v_beta;
    v_l1 = opt.v_l1;
    v_l2 = opt.v_l2;
    k0 = opt.k0;
    k1 = opt.k1;
    force_v_sparse = opt.force_v_sparse;
    pModel = new ftrl_model<T>(opt.factor_num, opt.init_mean, opt.init_stdev);// 初始化Model
    pLockPool = new lock_pool();   // 新建一些锁
}


// dataBuffer: 读入的每行数据(每个元素对应一行数据line:string). 特征名是string，特征值是float/double
//    buffer[0]：1 sex:1 age:0.3
//    buffer[1]：0 sex:0 age:0.7
template<typename T>
void ftrl_trainer<T>::run_task(vector<string>& dataBuffer)
{
    for(size_t i = 0; i < dataBuffer.size(); ++i)
    {
        fm_sample sample(dataBuffer[i]); // 从每行数据中获得样本[i]
        train(sample.y, sample.x);       // 每个样本逐个训练
    }
}


template<typename T>
bool ftrl_trainer<T>::load_model(const string& modelPath, const string& modelFormat)
{
    return pModel->load_model(modelPath, modelFormat);
}


template<typename T>
bool ftrl_trainer<T>::output_model(const string& modelPath, const string& modelFormat)
{
    return pModel->output_model(modelPath, modelFormat);
}


//输入一个样本，更新参数
// x: [(特征1:取值),(特征2:取值),...]
// y: -1,1
template<typename T>
void ftrl_trainer<T>::train(int y, const vector<pair<string, double> >& x)
{

    // 对应bias这个参数 （以及2个ftrl参数）
    ftrl_model_unit<T>* thetaBias = pModel->get_or_init_model_unit_bias(); // 得到已有的或者初始化的一个特征unit指针

    // 该样本的每个特征都应该有一个特征unit（含该特征的所有一二阶参数，以及每个参数对应的2个ftrl参数）
    vector<ftrl_model_unit<T>*> theta(x.size(), NULL);
    int xLen = x.size(); // 该样本（非稀疏）特征数目

    vector<mutex*> feaLocks(xLen + 1, NULL);  // 不同特征对应的锁（连同bias）（同一特征，更新需和别的线程抢）

    // 每个特征
    for(int i = 0; i < xLen; ++i)
    {
        const string& index = x[i].first;                   // 特征名 (may after one-hot)
        theta[i] = pModel->get_or_init_model_unit(index);   // 根据特征名找这个特征已有de参数单元（没有就初始化）
        feaLocks[i] = pLockPool->get_feature_lock(index);   // 获取该特征对应的锁（不同特征更新不影响，但不同线程更新同一特征需获得锁）
    }
    feaLocks[xLen] = pLockPool->get_bias_lock();            // bias参数对应的锁

    //update w via FTRL （基于历史梯度，依次更新每个特征的一阶参数wi）
    // 该样本对所有特征的一阶参数的贡献，先做完
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit<T>& mu = i < xLen ? *(theta[i]) : *thetaBias; // 该特征对应的特征unit
        if((i < xLen && k1) || (i == xLen && k0))              // bias/一阶参数对应的w:
        {
            feaLocks[i]->lock();           // 对该特征上锁（否则阻塞）
            if(fabs(mu.w_zi) <= w_l1)      // 按ftrl截断 （zi < l）
            {
                mu.wi = 0.0;
            }
            else
            {
                // 该特征一阶参数wi==0,（累积梯度>0） 二阶参数重新初始化
                if(force_v_sparse && mu.w_ni > 0 && 0.0 == mu.wi)
                {
                    mu.reinit_vi(pModel->init_mean, pModel->init_stdev);
                }
                // 最终该特征的一阶参数 y= wixi。存在该特征对应的全局feature unit里。
                mu.wi = (-1) *                                               // 一阶参数共用同样的超参数（学习了，l1等）
                    (1 / (w_l2 + (w_beta + sqrt(mu.w_ni)) / w_alpha)) *
                    (mu.w_zi - utils::sgn(mu.w_zi) * w_l1);
            }
            feaLocks[i]->unlock();
        }
    }

    //update v via FTRL。更新该样本下的每个特征向量
    for(int i = 0; i < xLen; ++i) // 遍历每个特征
    {
        ftrl_model_unit<T>& mu = *(theta[i]);       // 该特征对应的单元
        for(int f = 0; f < pModel->factor_num; ++f) // 的每个维度
        {
            feaLocks[i]->lock();
            T& vif = mu.vi(f);                     // 第i特征向量的第f个维度
            T& v_nif = mu.v_ni(f);
            T& v_zif = mu.v_zi(f);
            if(v_nif > 0)                          // 首次没有累计梯度，先不更新参数值。先记下这次的梯度。
            {
                if(force_v_sparse && 0.0 == mu.wi) // 强制稀疏
                {
                    vif = 0.0;
                }
                else if(fabs(v_zif) <= v_l1) {      // 截断
                    vif = 0.0;
                }
                else                               // 更新该维度
                {
                    vif = (-1) *
                        (1 / (v_l2 + (v_beta + sqrt(v_nif)) / v_alpha)) *
                        (v_zif - utils::sgn(v_zif) * v_l1);
                }
            }
            feaLocks[i]->unlock();
        }
    }

    // 更新每个参数对应的累积参数
    vector<double> sum(pModel->factor_num);
    double bias = thetaBias->wi;
    double p = pModel->predict(x, bias, theta, sum);    // 该样本预测值
    double mult = y * (1 / (1 + exp(-p * y)) - 1);   // y * ( sigmoid( y*y_pred) -1 )
    //update w_n, w_z
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit<T>& mu = i < xLen ? *(theta[i]) : *thetaBias;
        double xi = i < xLen ? x[i].second : 1.0;  // 特征xi取值
        if((i < xLen && k1) || (i == xLen && k0))
        {
            feaLocks[i]->lock();
            double w_gi = mult * xi;  // 一阶特征的梯度本身
            double w_si = 1 / w_alpha * (sqrt(mu.w_ni + w_gi * w_gi) - sqrt(mu.w_ni));// 学习率倒数的差。可累加
            mu.w_zi += w_gi - w_si * mu.wi;  // 累积zi
            mu.w_ni += w_gi * w_gi;          // 累计梯度平方
            feaLocks[i]->unlock();
        }
    }
    //更新所有特征向量的所有维度对应的ftrl累积参数
    //update v_n（累积梯度平方）, v_z（累积阈值梯度）
    for(int i = 0; i < xLen; ++i)
    {
        ftrl_model_unit<T>& mu = *(theta[i]);
        const double& xi = x[i].second;              // 特征i的取值
        for(int f = 0; f < pModel->factor_num; ++f)  // 每个维度
        {
            feaLocks[i]->lock();
            T& vif = mu.vi(f);
            T& v_nif = mu.v_ni(f);
            T& v_zif = mu.v_zi(f);
            double v_gif = mult * (sum[f] * xi - vif * xi * xi);  // 梯度本身。sum[f]：之前算出来的第f维的 sum_i (xi*vf)
            double v_sif = 1 / v_alpha * (sqrt(v_nif + v_gif * v_gif) - sqrt(v_nif));
            v_zif += v_gif - v_sif * vif;
            v_nif += v_gif * v_gif;        // 第i个特征向量第f维的累计梯度
            //有的特征在整个训练集中只出现一次，这里还需要对vif做一次处理
            if(force_v_sparse && v_nif > 0 && 0.0 == mu.wi)
            {
                vif = 0.0;
            }
            feaLocks[i]->unlock();
        }
    }
}


#endif /*FTRL_TRAINER_H_*/
