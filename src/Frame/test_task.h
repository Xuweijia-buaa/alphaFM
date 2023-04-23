#ifndef TEST_TASK_H
#define TEST_TASK_H

#include <iostream>
#include "pc_task.h"
using namespace std;


// 单纯打印出来。可以用来测试消费者线程送给任务的输入数据是否正确。
class test_task : public pc_task
{
public:
    test_task(){}
    virtual void run_task(vector<string>& dataBuffer)
    {
        cout << "==========\n";
        for(int i = 0; i < dataBuffer.size(); ++i)
        {
            cout << dataBuffer[i] << endl;
        }
        cout << "**********\n";
    }
};


#endif //TEST_TASK_H
