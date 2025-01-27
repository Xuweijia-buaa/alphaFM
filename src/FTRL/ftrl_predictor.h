#ifndef FTRL_PREDICTOR_H_
#define FTRL_PREDICTOR_H_

#include "../Frame/pc_frame.h"
#include "predict_model.h"
#include "../Sample/fm_sample.h"

// 指定要加载的模型，类型
struct predictor_option
{
    predictor_option() : factor_num(8), threads_num(1), model_format("txt") {}
    string model_path, model_format, predict_path, model_number_type;
    int threads_num, factor_num;
    
    void parse_option(const vector<string>& args)
    {
        int argc = args.size();
        if(0 == argc)
            throw invalid_argument("invalid command\n");
        for(int i = 0; i < argc; ++i)
        {
            if(args[i].compare("-m") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_path = args[++i];
            }
            else if(args[i].compare("-mf") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_format = args[++i];
                if("bin" != model_format && "txt" != model_format)
                    throw invalid_argument("invalid command\n");
            }
            else if(args[i].compare("-dim") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                factor_num = stoi(args[++i]);
            }
            else if(args[i].compare("-core") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                threads_num = stoi(args[++i]);
            }
            else if(args[i].compare("-out") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                predict_path = args[++i];
            }
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

// 同样继承pc_task
template<typename T>
class ftrl_predictor : public pc_task
{
public:
    ftrl_predictor(const predictor_option& opt);
    ~ftrl_predictor();
    virtual void run_task(vector<string>& dataBuffer);
    
private:
    predict_model<T>* pModel;// 加载训好的模型
    ofstream fPredict;// 输出流
    mutex outMtx;
};


template<typename T>
ftrl_predictor<T>::ftrl_predictor(const predictor_option& opt)
{
    // 加载训好的模型
    pModel = new predict_model<T>(opt.factor_num);
    cout << "load model..." << endl;
    if(!pModel->load_model(opt.model_path, opt.model_format))
    {
        cerr << "load model error!" << endl;
        exit(1);
    }
    cout << "model loading finished" << endl;
    // 打开输出文件准备写
    fPredict.open(opt.predict_path, ofstream::out);
    if(!fPredict)
    {
        cerr << "open output file error!" << endl;
        exit(1);
    }
}


template<typename T>
ftrl_predictor<T>::~ftrl_predictor()
{
    fPredict.close();
}

// 被frame调用. con_thread里
// 核心预测函数，处理输入的测试样本。每行一个样本 ()
template<typename T>
void ftrl_predictor<T>::run_task(vector<string>& dataBuffer)
{
    vector<string> outputVec(dataBuffer.size());
    for(size_t i = 0; i < dataBuffer.size(); ++i)
    {
        fm_sample sample(dataBuffer[i]);   // 得到每个预测样本
        // 计算该样本得分：(输入样本和模型，输出逻辑回归后的score)
        double score = pModel->get_score(sample.x, pModel->muBias->wi, pModel->muMap);
        outputVec[i] = to_string(sample.y) + " " + to_string(score);
    }
    outMtx.lock();
    for(size_t i = 0; i < outputVec.size(); ++i)
    {
        fPredict << outputVec[i] << endl;        // 每个样本： 真实label（1、-1）  score(0-1之间的sigmoid后的结果)
    }
    outMtx.unlock();
}


#endif /*FTRL_PREDICTOR_H_*/
