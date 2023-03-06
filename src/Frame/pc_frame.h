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
    // task:trainer,t_num:线程数目
    bool init(pc_task& task, int t_num, int buf_size = 5000, int log_num = 200000);
    void run();

private:
    //  对buffer的读写用锁来控制,防止同一时间既读又写。但每个线程读完，处理前，会释放锁，让其他线程可以继续对buffer读写
    //  用信号量来告诉其他线程可以读新的数据了。但是每批新数据都交给一个线程处理。
    int threadNum;
    pc_task* pTask;        // trainer  训练器本身
    mutex bufMtx;
    sem_t semPro, semCon;  // 2个信号量，用来做线程间的同步。生产者-消费者模式
    queue<string> buffer;      // 每个元素对应输入流的一行，一次处理最多放5000行。处理完的应该是直接排队从尾部出去了？
    vector<thread> threadVec;  // 放读取线程+10个处理线程。读取线程不停的从cin中按批读，直到结束。 每一批数据由一个线程处理（包含cin中的最多5000行）。
    int bufSize; // 默认5000 批大小
    int logNum;
    void pro_thread();   // 读取线程
    void con_thread();   // 处理线程   主要靠信号量同步
};


#endif //PC_FRAME_H
