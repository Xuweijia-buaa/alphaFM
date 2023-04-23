#ifndef FTRL_MODEL_H_
#define FTRL_MODEL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include "../Utils/utils.h"
#include "../Mem/mem_pool.h"
#include "../Mem/my_allocator.h"
#include "model_bin_file.h"

using namespace std;

//每一个特征对应的unit单元: 含该特征对应的一阶参数(1个)，二阶特征参数(K个)本身，以及对应的每个参数的2种累积梯度。
template<typename T>
class ftrl_model_unit
{
public:
    // 该特征的一阶参数(或bias参数)
    T wi;        //  参数真实取值
    T w_ni;      //  参数梯度平方的累积   ni = sum gi^2
    T w_zi;      //  参数梯度累积（近似）  zi (初始是0)= sum (gi -  sigma * w)

    
private:
    // 该特征的二阶参数。同样含梯度累积量z,梯度平方累积n. 
    // 所有特征单元的k,offset都是相同，所有特征的实例共享(static)。算类的特征。创建新对象时，不分配这些元素对应内存。
    static size_t offset_v;
    static size_t offset_vn;
    static size_t offset_vz;
    static size_t class_size;
    static int ext_mem_size;
    static int factor_num;// 该特征对应的向量维度

public:
    // 计算特征unit所需空间：一个wi,k个vi  (*3)
    static void static_init(int _factor_num)
    {

        factor_num = _factor_num;              // k
        offset_v = 0;
        offset_vn = offset_v + _factor_num;    // 2k
        offset_vz = offset_vn + _factor_num;   // 3k
        class_size = sizeof(ftrl_model_unit<T>); // unit中，一阶参数占用内存(字节数)。sizeof(unit)是unit对象本身的大小，不算static。

        // Cache_hash_code:是否连同hash_code也存着。这样比较两个对象时候就可以直接比较hash_code.
        // _Value 是 pair<char*,ftrl_model_unit>。预先分配这样的内存出来，可以根据->_M_v()得到地址
        using node_type = std::__detail::_Hash_node<std::pair<const char* const, ftrl_model_unit<T> >, false>;

        // offset_this: node的初始地址到unit的首地址之间的offset。
        size_t offset_this = get_value_offset_in_Hash_node((node_type*)NULL);

        // padding: node中unit之后剩余的空间
        //  | ----------- |  ----unit ---- |  paddding
        //    offfset_this    class_size
        size_t padding = sizeof(node_type) - offset_this - class_size;

        if(padding > 4)
        {
            cerr << "ftrl_model_unit::static_init: padding size exception" << endl;
            exit(1);
        }
        // 需要的额外空间: 每个特征对应向量中的每一维vi，都需要3个参数（值，2个ftrl参数）。需要额外在unit之后分配的空间 （unit只分了3个一阶参数）
        ext_mem_size = (int)(3 * _factor_num * sizeof(T)) - (int)padding;
    }
    
    
    inline static int get_ext_mem_size()
    {
        return ext_mem_size;
    }
    
    
    static size_t get_mem_size()
    {
        return class_size + 3 * factor_num * sizeof(T);
    }
    
    
    static ftrl_model_unit* create_instance(int _factor_num, double v_mean, double v_stdev)
    {
        size_t mem_size = get_mem_size();
        void* pMem = malloc(mem_size);                              // 在堆上分配
        ftrl_model_unit* pInstance = new(pMem) ftrl_model_unit();   // 通过placement new,在这片内存上，初始化一个特征unit,并返回对应指针
        //  设置某该特征unit,每个参数的初始取值。返回该特征unit的指针
        pInstance->instance_init(_factor_num, v_mean, v_stdev);     // 按均值方差初始化该特征单元
        return pInstance;
    }
    

    // （load中用）文本用每一行特征，去初始化特征unit（包括特征取值，和已经累积的特征梯度）
    static ftrl_model_unit* create_instance(int _factor_num, const vector<string>& modelLineSeg)
    {
        size_t mem_size = get_mem_size();// 该特征单元总的mem，包含特征对应的所有参数（以及ftrl参数）
        void* pMem = malloc(mem_size);
        ftrl_model_unit* pInstance = new(pMem) ftrl_model_unit(); // 分配一个新的空间
        pInstance->instance_init(_factor_num, modelLineSeg);
        return pInstance;
    }
    
    
    ftrl_model_unit()
    {}
    
    // 设置该特征unit中：每个参数的初始取值。内存已分配好，只需要设置值。
    void instance_init(int _factor_num, double v_mean, double v_stdev)
    {
        wi = 0.0;
        w_ni = 0.0;
        w_zi = 0.0;
        for(int f = 0; f < _factor_num; ++f)
        {
            vi(f) = utils::gaussian(v_mean, v_stdev);  // 所需的内存，已经通过malloc分配在堆上了。
                                                       // 该单元中，v中第k个参数vk，用offset找到地址，返回对应引用，初始化对应内存。 每个vi,高斯分布
            v_ni(f) = 0.0;
            v_zi(f) = 0.0;
        }
    }
    
    // loadmodel，用文本初始化每个特征unit
    void instance_init(int _factor_num, const vector<string>& modelLineSeg)
    {
        wi = stod(modelLineSeg[1]);                 // 1  一阶特征参数
        w_ni = stod(modelLineSeg[2 + _factor_num]);
        w_zi = stod(modelLineSeg[3 + _factor_num]);
        for(int f = 0; f < _factor_num; ++f)
        {
            vi(f) = stod(modelLineSeg[2 + f]);     // 2-2+f 特征向量
            v_ni(f) = stod(modelLineSeg[4 + _factor_num + f]);
            v_zi(f) = stod(modelLineSeg[4 + 2 * _factor_num + f]);
        }
    }
    
    
    inline T& vi(size_t f) const
    {
        // 返回二阶参数中（共k个），第f个参数（的引用）。用来初始化vf
        char* p = (char*)this + class_size;  // p:二阶参数的首地址。 this:该特征单元对象的地址。加上class_size:是走过所有一阶参数地址。
        return *((T*)p + offset_v + f);      // vi直接放一阶参数之后。放f个T，位于p-p+f。 
                                             // 其中第f个地址，对地址(p+f)解引用，返回。返回的是对单个vf元素的引用（还未初始化）。但内存已分配在栈中，通过malloc.
    }
    
    
    inline T& v_ni(size_t f) const
    {
        // 返回k个二阶参数，对应的k个v_ni中。第f个参数
        char* p = (char*)this + class_size;   // v_ni放在vi之后，对应特征向量各参数的ftrl参数. p依然是二阶参数的首地址。
        return *((T*)p + offset_vn + f);      // p+offset_vn(2k):  v_ni的首地址。返回之后的第f个参数。
    }
    
    
    inline T& v_zi(size_t f) const
    {   // 返回k个二阶参数，对应的k个v_zi中。第f个参数
        char* p = (char*)this + class_size;  // v_zi放在vn_i之后，对应特征向量各参数的另一个ftrl参数 连续放。
        return *((T*)p + offset_vz + f);
    }
    
    
    void reinit_vi(double v_mean, double v_stdev)
    {
        for(int f = 0; f < factor_num; ++f)
        {
            vi(f) = utils::gaussian(v_mean, v_stdev);
        }
    }
    
    
    inline bool is_none_zero()
    {
        if(0.0 != wi) return true;
        for(int f = 0; f < factor_num; ++f)
        {
            if(0.0 != vi(f)) return true;
        }
        return false;
    }
    
    // 重载了unit的打印函数<<
    // 先打印 wi和特征向量，再打印ftrl参数
    friend inline ostream& operator <<(ostream& os, const ftrl_model_unit& mu)
    {
         // 每个参数，除了打印自身。还打印累计梯度。用于后续学习

        os << mu.wi;                                            // 每个特征，对应的参数本身 （wi和k个vi）      
        for(int f = 0; f < ftrl_model_unit::factor_num; ++f)
        {
            os << " " << mu.vi(f);                                 
        }

        os << " " << mu.w_ni << " " << mu.w_zi;                 // 每个参数，对应的累计梯度。用于后续学习
        for(int f = 0; f < ftrl_model_unit::factor_num; ++f)
        {
            os << " " << mu.v_ni(f);
        }
        for(int f = 0; f < ftrl_model_unit::factor_num; ++f)
        {
            os << " " << mu.v_zi(f);
        }
        return os;
    }
};

template<typename T>
size_t ftrl_model_unit<T>::offset_v;
template<typename T>
size_t ftrl_model_unit<T>::offset_vn;
template<typename T>
size_t ftrl_model_unit<T>::offset_vz;
template<typename T>
size_t ftrl_model_unit<T>::class_size;
template<typename T>
int ftrl_model_unit<T>::ext_mem_size;
template<typename T>
int ftrl_model_unit<T>::factor_num;


// 用来存所有特征的特征单元 特征名：特征unit。  自己写的hash和equal,分配内存单元
template<typename T>
using my_hash_map = unordered_map<const char*, ftrl_model_unit<T>, 
                                               my_hash, 
                                               my_equal, 
                                               my_allocator<pair<const char*, ftrl_model_unit<T> >, T, ftrl_model_unit>
                                                >;


// model类，用来实现具体的model逻辑
template<typename T>
class ftrl_model
{
public:
    // 每个模型的参数：wi, vi b
    my_hash_map<T> muMap;       // 模型的所有特征,每个参数一个unit: {特征名: 特征对应unit}
                                // 每个unit:含特征i的一二阶参数wi,k个vi,以及对应的累积梯度
                                // 新特征，会初始化一个unit


    ftrl_model_unit<T>* muBias; // 模型总的bias对应的特征unit   sum (wixi) +b   单独一个参数（对应的unit） 

    int factor_num;            // 模型k
    double init_stdev;         // 指定vi如何初始化. mean std
    double init_mean;

public:
    ftrl_model(int _factor_num);
    ftrl_model(int _factor_num, double _mean, double _stdev);  // 构造函数：指定初始化维度，向量均值方差
    ftrl_model_unit<T>* get_or_init_model_unit(const string& index);
    ftrl_model_unit<T>* get_or_init_model_unit_bias();

    double predict(const vector<pair<string, double> >& x, double bias, vector<ftrl_model_unit<T>*>& theta, vector<double>& sum);
    bool output_model(const string& modelPath, const string& modelFormat);
    void output_model_one_line(ostream& out, const char* feaName, ftrl_model_unit<T>* pMu, bool isBias = false);
    bool load_model(const string& modelPath, const string& modelFormat);
    inline bool convert_one_line_of_txt_model_to_vec(ifstream& in, vector<string>& strVec, bool& dataFmtErr, bool isBias = false);
    size_t get_unit_mem_size();
    static const string& get_bias_fea_name();

private:
    bool load_txt_model(const string& modelPath);
    bool load_txt_model(ifstream& in);
    bool load_bin_model(const string& modelPath);
    bool output_txt_model(const string& modelPath);
    bool output_bin_model(const string& modelPath);
    inline char* create_fea_c_str(const char* key);
    
private:
    mutex mtx;
    mutex mtx_bias;
    static const string biasFeaName;
};

template<typename T>
const string ftrl_model<T>::biasFeaName = "bias";


// 初始化
template<typename T>
ftrl_model<T>::ftrl_model(int _factor_num)
{
    factor_num = _factor_num;
    init_mean = 0.0;
    init_stdev = 0.0;
    muBias = NULL;
    ftrl_model_unit<T>::static_init(_factor_num);
}


template<typename T>
ftrl_model<T>::ftrl_model(int _factor_num, double _mean, double _stdev)
{
    factor_num = _factor_num;// k
    init_mean = _mean;
    init_stdev = _stdev;
    muBias = NULL;
    ftrl_model_unit<T>::static_init(_factor_num); // 按vi特征维度，初始化一个ftrl_model_unit。计算好每个vi需要的空间
}


// 按特征名，返回该特征对应的unit。 没有就
template<typename T>
ftrl_model_unit<T>* ftrl_model<T>::get_or_init_model_unit(const string& index)
{
    auto iter = muMap.find(index.c_str());
    if(iter == muMap.end())        // 该特征还没有unit，初始化一个
    {
        mtx.lock();
        ftrl_model_unit<T>* pMU = NULL;
        iter = muMap.find(index.c_str());
        if(iter != muMap.end())
        {
            pMU = &(iter->second);                                      // 上锁后，发现该特征已有对应unit（地址），直接返回
        }
        else
        {
            // 给key分配一个等长的字符串空间，作为map的key
            char* pKey = create_fea_c_str(index.c_str());
            muMap[pKey].instance_init(factor_num, init_mean, init_stdev);// 没有，初始化一个unit
            pMU = &muMap[pKey];
        }
        mtx.unlock();
        return pMU;
    }
    else
    {
        return &(iter->second);                                        // 找到该特征对应的unit(地址)，直接返回
    }
}

// 返回已有的特征单元(没有就初始化一个)
template<typename T>
ftrl_model_unit<T>* ftrl_model<T>::get_or_init_model_unit_bias()
{
    if(NULL == muBias)
    {
        mtx_bias.lock();
        if(NULL == muBias)
        {
            // 初始化一个特征unit,每个参数的值初始化。返回该特征unit的指针。
            // 内存通过malloc+placement new分配在堆上.再依次用高斯分布初始化。
            muBias = ftrl_model_unit<T>::create_instance(0, init_mean, init_stdev);
        }
        mtx_bias.unlock();
    }
    return muBias;
}


template<typename T>
double ftrl_model<T>::predict(const vector<pair<string, double> >& x, double bias, vector<ftrl_model_unit<T>*>& theta, vector<double>& sum)
{
    double result = 0;
    result += bias;
    for(int i = 0; i < x.size(); ++i)
    {
        result += theta[i]->wi * x[i].second;     // sum wi*xi 用每个特征unit中的wi
    }
    double sum_sqr, d;

    // 拆街到第f维去计算
    // sum(xivi) ** 2 -  sum (xivi**2)
    for(int f = 0; f < factor_num; ++f)
    {
        sum[f] = sum_sqr = 0.0;
        for(int i = 0; i < x.size(); ++i)
        {
            d = theta[i]->vi(f) * x[i].second;  // x * v
            sum[f] += d;                        // sum_i (xi*vf)
            sum_sqr += d * d;
        }
        result += 0.5 * (sum[f] * sum[f] - sum_sqr);
    }
    return result;  // 最终的预测结果y
}


// 给key分配一个字符串空间，返回首地址
template<typename T>
char* ftrl_model<T>::create_fea_c_str(const char* key)
{
    size_t len = strlen(key);
    char* p = (char*)mem_pool::get_mem(len + 1);
    strncpy(p, key, len);
    p[len] = 0;
    return p;
}


template<typename T>
bool ftrl_model<T>::output_model(const string& modelPath, const string& modelFormat)
{
    if("txt" == modelFormat) return output_txt_model(modelPath);
    else if("bin" == modelFormat) return output_bin_model(modelPath);
    else return false;
}

// 按照固定格式，输出模型参数：
// 不仅输出每个参数本身，还输出参数对应的2个累积梯度。用于后续增量学习
template<typename T>
bool ftrl_model<T>::output_txt_model(const string& modelPath)
// modelPath:入参传入的模型输出路径
{
    ofstream out(modelPath, ofstream::out); // 输出到out
    if(!out) return false;
    // 第一行是bias: bias  value  ni  zi
    // 其他行：(遍历map得所有特征.每行一个特征，特征无序。重载了<< )
    //             特征名  wi  特征向量（v1 v2  ... vf）  剩下是累计梯度：w_n   w_z   特征向量累积梯度（f个） 特征向量累积梯度平方（f个）

    out << biasFeaName << " " << muBias->wi << " " << muBias->w_ni << " " << muBias->w_zi << endl;  // bias参数，和对应的2个梯度

    for(auto iter = muMap.begin(); iter != muMap.end(); ++iter)                                     // 每个特征单元： 一个wi,k个vi，以及对应的梯度
    {
        out << iter->first << " " << iter->second << endl; // 打印每个特征的参数。  first是特征名。 second是该特征unit,<<重载，用来打印wi,vi及其累积梯度
    }
    out.close();
    return true;
}

// 模型写成bin文件,直接写入内存内容（按字节）
template<typename T>
bool ftrl_model<T>::output_bin_model(const string& modelPath)
{
    model_bin_file mbf;
    if(!mbf.open_file_for_write(modelPath, sizeof(T), factor_num, get_unit_mem_size())) return false;
    if(!mbf.write_one_fea_unit(get_bias_fea_name().c_str(), muBias, true)) return false;// write bias
    for(auto iter = muMap.begin(); iter != muMap.end(); ++iter)
    {
        if(!mbf.write_one_fea_unit(iter->first, &(iter->second), iter->second.is_none_zero())) return false;
    }
    if(!mbf.close_file()) return false;
    return true;
}


template<typename T>
void ftrl_model<T>::output_model_one_line(ostream& out, const char* feaName, ftrl_model_unit<T>* pMu, bool isBias)
{
    if(isBias)
    {
        out << feaName << " " << pMu->wi << " " << pMu->w_ni << " " << pMu->w_zi << endl;
    }
    else
    {
        out << feaName << " " << *pMu << endl;
    }
}


template<typename T>
bool ftrl_model<T>::load_model(const string& modelPath, const string& modelFormat)
{
    if("txt" == modelFormat) return load_txt_model(modelPath);
    else if("bin" == modelFormat) return load_bin_model(modelPath);
    else return false;
}


template<typename T>
bool ftrl_model<T>::load_txt_model(const string& modelPath)
{
    ifstream in(modelPath);
    if(!in) return false;
    bool res = load_txt_model(in);
    in.close();
    return res;
}

// 读model文件，解析到unit里
template<typename T>
bool ftrl_model<T>::load_txt_model(ifstream& in)
{
    string line;
    if(!getline(in, line))
    {
        return false;
    }
    vector<string> strVec;
    utils::split_string(line, ' ', &strVec);
    if(strVec.size() != 4)
    {
        return false;
    }
    muBias = ftrl_model_unit<T>::create_instance(0, strVec);  // 先读bias。在栈上分配内存，建bias的unit
    while(getline(in, line))
    {
        strVec.clear();
        utils::split_string(line, ' ', &strVec);
        if(strVec.size() != 3 * factor_num + 4)
        {
            return false;
        }
        string& index = strVec[0];
        char* pKey = create_fea_c_str(index.c_str());
        muMap[pKey].instance_init(factor_num, strVec);        // 用str初始化每个特征unit
    }
    return true;
}

// 直接读到内存
template<typename T>
bool ftrl_model<T>::load_bin_model(const string& modelPath)
{
    model_bin_file mbf;
    if(!mbf.open_file_for_read(modelPath)) return false;
    if(mbf.get_num_byte_len() != sizeof(T)) return false;
    if(mbf.get_factor_num() != (size_t)factor_num) return false;
    if(mbf.get_unit_len() != get_unit_mem_size()) return false;
    char* buffer = new char[64*1024];
    unsigned short feaLen;
    if(!mbf.read_one_fea(buffer, feaLen)) return false;
    buffer[feaLen] = 0;
    if(get_bias_fea_name() != string(buffer)) return false;
    muBias = ftrl_model_unit<T>::create_instance(0, 0.0, 0.0);
    if(!mbf.read_one_unit(muBias)) return false;
    for(size_t i = 1; i < mbf.get_fea_num(); ++i)
    {
        if(!mbf.read_one_fea(buffer, feaLen)) return false;
        buffer[feaLen] = 0;
        char* pKey = create_fea_c_str(buffer);
        if(!mbf.read_one_unit(&(muMap[pKey]))) return false;
    }
    if(!mbf.close_file()) return false;
    delete[] buffer;
    return true;
}


template<typename T>
const string& ftrl_model<T>::get_bias_fea_name()
{
    return biasFeaName;
}


template<typename T>
inline bool ftrl_model<T>::convert_one_line_of_txt_model_to_vec(ifstream& in, vector<string>& strVec,  bool& dataFmtErr, bool isBias)
{
    static string line;
    dataFmtErr = false;
    if(!getline(in, line))
    {
        return false;
    }
    int fn = isBias ? 0 : factor_num;
    strVec.clear();
    utils::split_string(line, ' ', &strVec);
    if(strVec.size() != 3 * fn + 4)
    {
        dataFmtErr = true;
        return false;
    }
    return true;
}


template<typename T>
size_t ftrl_model<T>::get_unit_mem_size()
{
    return ftrl_model_unit<T>::get_mem_size();
}



#endif /*FTRL_MODEL_H_*/
