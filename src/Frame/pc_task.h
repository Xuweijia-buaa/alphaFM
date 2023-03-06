#ifndef PC_TASK_H
#define PC_TASK_H

#include <string>
#include <vector>

using std::string;
using std::vector;

class pc_task
{
public:
    pc_task(){}                                             // 构造函数
    virtual void run_task(vector<string>& dataBuffer) = 0;  // 虚函数。子类可覆盖。传入数据dataBuffer是vector<string>&，包含训练样本
};


#endif //PC_TASK_H
