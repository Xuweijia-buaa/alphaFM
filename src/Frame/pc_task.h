#ifndef PC_TASK_H
#define PC_TASK_H

#include <string>
#include <vector>

using std::string;
using std::vector;


// 抽象类。每个消费者线程，拿到数据后，要运行的任务本身
class pc_task
{
public:
    pc_task(){}                                             // 构造函数
    virtual void run_task(vector<string>& dataBuffer) = 0;  // 每个消费者线程，拿到数据后，要运行的任务本身。每个数据对应一个样本。（虚函数，子类可定制不同任务）
};


#endif //PC_TASK_H
