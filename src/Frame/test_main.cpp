#include "pc_frame.h"
#include "test_task.h"

int main()
{
    test_task task;                     // 假的任务，只是打印样本
    pc_frame frame;
    frame.init(task, 2, 5, 5);          // 2消费者线程；buffer大小5；日志间隔5；
    frame.run();
    return 0;
}

