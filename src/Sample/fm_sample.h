#ifndef FM_SAMPLE_H_
#define FM_SAMPLE_H_

#include <string>
#include <vector>
#include <iostream>

using namespace std;

// 样本类。单个样本。对应输入的一行string
class fm_sample
{
public:
    int y;                               // label,不论输入如何，被转化为-1,1
    vector<pair<string, double> > x;     // x:[(特征:取值),(特征:取值)]
    fm_sample(const string& line);
private:
    static const string spliter;
    static const string innerSpliter;
};

const string fm_sample::spliter = " ";
const string fm_sample::innerSpliter = ":";

// 用line初始化成一个样本
// label f1:v1 f2:v2 f3:v3
// 1 sex:1 age:0.3
fm_sample::fm_sample(const string& line)    // 从每行输入中提取数据:
{
    // label
    this->x.clear();
    size_t posb = line.find_first_not_of(spliter, 0); // 首个非“ ”的位置：label位置
    size_t pose = line.find_first_of(spliter, posb);      //  之后的首个“ ”位置
    int label = atoi(line.substr(posb, pose-posb).c_str());  // label:0/1/-1
    this->y = label > 0 ? 1 : -1;

    string key;    // 每个特征名 str
    double value;  // 每个特征值 double
    while(pose < line.size()) // 特征们
    {
        posb = line.find_first_not_of(spliter, pose); // 每个特征名的开头位置：下一个非“ ”的位置
        if(posb == string::npos)
        {
            break;
        }
        pose = line.find_first_of(innerSpliter, posb); // 特证名结束位置
        if(pose == string::npos)
        {
            cerr << "wrong line of sample input\n" << line << endl;
            exit(1);
        }
        key = line.substr(posb, pose-posb);         // 特征名 str
        posb = pose + 1;
        if(posb >= line.size())
        {
            cerr << "wrong line of sample input\n" << line << endl;
            exit(1);
        }
        pose = line.find_first_of(spliter, posb);     // 该特征的结尾
        value = stod(line.substr(posb, pose-posb));  //  特征值。此时pose在特征结尾，posb在特征名结尾。转为double

        // 这里样本只用了非0特征（0值特征作为稀疏的特征，没有加进来）
        if(value != 0)
        {
            this->x.push_back(make_pair(key, value));
        }
    }
}


#endif /*FM_SAMPLE_H_*/
