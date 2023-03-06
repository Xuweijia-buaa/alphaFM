#include <iostream>
#include <map>
#include <fstream>
#include "src/Frame/pc_frame.h"
#include "src/FTRL/ftrl_trainer.h"

using namespace std;
// y= wo + wi *xi   + sum i,j  <vi,vj> xi xj
string train_help()
{
    return string(
            "\nusage: cat sample | ./fm_train [<options>]"
            "\n"
            "\n"
            "options:\n"
            "-m <model_path>: 模型输出路径 set the output model path\n"
            "-mf <model_format>: 模型输出格式 set the output model format, txt or bin\tdefault:txt\n"
            "-dim <k0,k1,k2>: k0=use bias, k1=use 1-way interactions, k2=dim of 2-way interactions\tdefault:1,1,8\n"
            "-init_stdev <stdev>: stdev for initialization of 2-way factors 特征向量的初始化方差\tdefault:0.1\n"
            "-w_alpha <w_alpha>: 控制ftrl学习率的超参数 w is updated via FTRL, alpha is one of the learning rate parameters\tdefault:0.05\n"
            "-w_beta <w_beta>: 控制ftrl学习率的超参数 w is updated via FTRL, beta is one of the learning rate parameters\tdefault:1.0\n"
            "-w_l1 <w_L1_reg>: 一阶参数的ftrl截断阈值。小于该值截断 L1 regularization parameter of w\tdefault:0.1\n"
            "-w_l2 <w_L2_reg>: 一阶参数的ftrl L2惩罚前的系数  L2 regularization parameter of w\tdefault:5.0\n"
            "-v_alpha <v_alpha>: 类似超参，但用于特征向量参数 v is updated via FTRL, alpha is one of the learning rate parameters\tdefault:0.05\n"
            "-v_beta <v_beta>: v is updated via FTRL, beta is one of the learning rate parameters\tdefault:1.0\n"
            "-v_l1 <v_L1_reg>: L1 regularization parameter of v\tdefault:0.1\n"
            "-v_l2 <v_L2_reg>: L2 regularization parameter of v\tdefault:5.0\n"
            "-core <threads_num>: set the number of threads\tdefault:1\n"
            "-im <initial_model_path>: set the initial model path\n"
            "-imf <initial_model_format>: set the initial model format, txt or bin\tdefault:txt\n"
            "-fvs <force_v_sparse>: 1：特征wi取0，把对应向量vi置为0. if fvs is 1, set vi = 0 whenever wi = 0\tdefault:0\n"
            "-mnt <model_number_type>: double or float\tdefault:double\n"
    );
}


template<typename T>
int train(const trainer_option& opt)
{
    // 初始化trainer  float/double
    ftrl_trainer<T> trainer(opt);

    // 在之前基础上加载
    if(opt.b_init)
    {
        cout << "load model..." << endl;
        if(!trainer.load_model(opt.init_model_path, opt.initial_model_format))
        {
            cerr << "failed to load model" << endl;
            return 1;
        }
        cout << "model loading finished" << endl;
    }

    // 多线程训练
    pc_frame frame;
    frame.init(trainer, opt.threads_num);
    frame.run();// 所有线程执行完

    // 输出模型
    cout << "output model..." << endl;
    if(!trainer.output_model(opt.model_path, opt.model_format))
    {
        cerr << "failed to output model" << endl;
        return 1;
    }
    cout << "model outputting finished" << endl;
    return 0;
}

// 输入单样本：cat sample | ./fm_train [<options>]
// 基于之前模型增量训练：cat sample | ./fm_train -core 10 -dim 1,1,8 -m fm_model.txt -im fm_model.txt

// argc是命令行总的参数个数(包含可执行文件本身)  argv[]为保存命令行参数的字符串指针
// 可以在一个文件中一次放所有样本（一次性输出到cin）,整体按10个线程读取+训练
// 也可以一边从hadoop下载，一边计算：  hadoop fs -cat train_data_hdfs_path |  ./fm_train
// (将路径train_data_hdfs_path指定的文件内容输出到stdout)

// 训练时，所有原始label都被转化成1/-1,这样好把损失函数写成y*y_pred的形式。但本质上还是交叉熵损失，label是0/-1均可以，都只是代表负例
// 预测时，输出的是sigmoid.依然按照0.5为阈值看正负例。（label=-1只是负例的名称）
int main(int argc, char* argv[])
{
    static_assert(sizeof(void *) == 8, "only 64-bit code generation is supported."); // 字长是8个字节（64bits）
    cin.sync_with_stdio(false);//打消iostream的输入 输出缓存，可以节省许多时间，使效率与scanf与printf相差无几
    cout.sync_with_stdio(false);
    srand(time(NULL));// 设置随机种子。time(NULL):当前时间
    trainer_option opt;          // 入参 结构体
    try
    {
        // 把输入的入参都解析到opt中，备用
        opt.parse_option(utils::argv_to_args(argc, argv));
    }
    catch(const invalid_argument& e)
    {
        cerr << "invalid_argument:" << e.what() << endl;
        cerr << train_help() << endl;// 提示信息
        return 1;
    }

    // opt作为入参，调用train函数，开始训练
    if("float" == opt.model_number_type)
    {
        return train<float>(opt);
    }
    return train<double>(opt);// 默认是duoble类型

    // 最后的模型参数
    // 首行是bias和对应的2个ftrl参数
    // 之后每行对应一个特征：
    // 特征名： 一阶参数 二阶参数（n个）  一阶参数的ftrl参数(2个)   二阶参数的ftrl参数1（n个）  二阶参数的ftrl参数2（n个）
}

