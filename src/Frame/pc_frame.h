#ifndef PC_FRAME_H
#define PC_FRAME_H

#include <string>
#include <vector>
#include <utility>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <semaphore.h>   // 信号量
#include <iostream>
#include "pc_task.h"

using namespace std;

class pc_frame
{
public:
    pc_frame(){}
    bool init(pc_task& task, int t_num, int buf_size = 5000, int log_num = 200000);  // 按下方生产者线程，消费者线程的具体逻辑，初始化所有线程，
                                                                                     // 并按生产者-消费者模式开始运行
                                                                                     
    void run();                                                                      // 线程运行后。阻塞，等待结束。

private:
    //  生产者-消费者模型。用信号量和buffer来控制读取速度。
    //  每批新数据都交给一个线程处理。但每个线程读完，处理前，会释放锁，让其他线程可以继续对buffer读写
    mutex bufMtx;              // 单纯对buffer本身读写上的互斥锁
    sem_t semPro, semCon;      // 2个信号量，用来做生产者-消费者线程间的同步。生产者-消费者模式
    queue<string> buffer;      // 生产者-消费者中的buffer. 大小5000，每个元素对应一行输入(一个样本)。每个消费者每次从buffer中取完，生产者才再次放入。
    vector<thread> threadVec;  // 放生产者消费者线程。std::thread 系统级线程。之后统一join。
    int bufSize;               // buffer大小
    int logNum;
    void pro_thread();         // 生产者线程的具体逻辑，放到threadVec中（声明）
    void con_thread();         // 消费者线程的具体逻辑，放到threadVec中（声明）  主要靠信号量同步
    int threadNum;             // 消费者线程的数目
    pc_task* pTask;            // 每个消费者线程，拿到私有数据后，要运行的任务对象本身。
};


#endif //PC_FRAME_H
