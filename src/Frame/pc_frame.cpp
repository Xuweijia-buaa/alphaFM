#include "pc_frame.h"

bool pc_frame::init(pc_task& task, int t_num, int buf_size, int log_num)
{
    pTask = &task;      // trainer
    threadNum = t_num;  // 线程数
    bufSize = buf_size; //  5000
    logNum = log_num;
    sem_init(&semPro, 0, 1);   // pshared==0:只能在当前进程的所有线程共享。（1可以在进程间共享）。初始设置为1，不阻塞
    sem_init(&semCon, 0, 0);   // 初始是0.阻塞

    // 每个线程对信号量的操作都是原子操作：
    // sem_wait(sem):  信号量为0的时候，该线程阻塞.  一直等到其他线程让信号量大于0时，解除阻塞，开始执行。  解除阻塞后再将sem的值减一，说明这个信号量被用了。 如果原信号量是2，调用后是1。原来0.就等待
    // sem_post(sem)：  用来增加信号量的值。调用后，会+1， 使得某个依赖该信号量的等待线程解除阻塞

    threadVec.clear();
    threadVec.push_back(thread(&pc_frame::pro_thread, this));  // 新建一个线程，并加入threadVec中，执行pro_thread函数。负责读输入流

    // 开10个线程，每个线程负责处理读入pro_thread读入的5000行。
    // 每个cron线程抢到后，就读入各自input_vec（消耗共同buffer中的数据）。
    // 之后就释放pro的信号量。pro可以继续读cin剩余的输入
    // pro在读。该cron在处理。 其他线程此时信号量为0
    // pro读完，cron信号量+1，所有cron线程又可以再从buffer取数据了。某其他线程可能获得了该信号量，去处理这一批。
    // 所以pro会一直读（几乎），其他所有cron线程，按批处理cin的5000行。
    for(int i = 0; i < threadNum; ++i)
    {
        threadVec.push_back(thread(&pc_frame::con_thread, this));  // 其他所有线程，也加入threadVec中，用来执行con_thread函数。初始信号量都是0。 阻塞
    }
    return true;
}


void pc_frame::run()
{
    for(int i = 0; i < threadVec.size(); ++i)
    {
        threadVec[i].join();    // 阻塞，直到每个线程都执行完毕
    }
}

// 读取输入。每读5000行处理一次。
void pc_frame::pro_thread()
{
    string line;
    size_t line_num = 0;
    int i = 0;
    bool finished_flag = false;
    while(true)
    {
        sem_wait(&semPro);   // 初始就是1，这里不阻塞，直接减成0。函数立即返回，不阻塞。  但是下一次读的时候，需要cron线程帮忙加1，这里就不阻塞。 信号量 P操作-1，直到0.同锁
        bufMtx.lock();       // 上锁。如果是中间某次读取，需要之前的所有cron线程释放锁。

        for(i = 0; i < bufSize; ++i)   // 5000
        {
            if(!getline(cin, line))  // 把每个输入流的整行读到line中，返回剩余的流（如果所有行读完了，结束）
            {
                finished_flag = true;      // 该输入流读完了
                break;
            }
            line_num++;
            buffer.push(line);             // 每个元素对应输入流的一行

            // 打日志
            if(line_num % logNum == 0)
            {
                cout << line_num << " lines finished" << endl;
            }

        }

        // cin没读完，但到了最大bufsize,先释放锁，让cron线程去执行这5000行。
        // 之后等再次能获得锁的时候（需要cron线程释放），继续读刘中剩余的行，每5000行一批处理。
        // 直到读完，交给cron线程,不再管了。
        bufMtx.unlock();
        sem_post(&semCon);//  cron信号量加1，可以处理这一批了

        if(finished_flag)   // 输入流都读完了，跳出这个循环
        {
            break;
        }
    }
}


void pc_frame::con_thread()
{
    bool finished_flag = false;
    vector<string> input_vec;
    input_vec.reserve(bufSize); // 至少5000大小
    while(true)
    {
        input_vec.clear();
        sem_wait(&semCon);               // 初始信号量是0，每个con线程都阻塞。但pro线程读完一部分数据后（5000行），就会让该信号量+1
                                         // 该cron线程得到该信号量,往下走，让信号量-1，其他线程阻塞。（除非pro又一批读完了，其他线程可以接过来处理）
        bufMtx.lock();
        for(int i = 0; i < bufSize; ++i)
        {
            if(buffer.empty())           // buffer不到5000/已经读完了，结束读取
            {
                finished_flag = true;
                break;
            }
            input_vec.push_back(buffer.front());  // 每行数据读到input_vec中。（按读入队列的先后。最早的最先读）
            buffer.pop();                         // 用掉后清空buffer
        }


        // 这一批读完了，释放锁。pro线程信号量+1。又可以让pro线程读新的输入了
        bufMtx.unlock();
        sem_post(&semPro);

        // 处理这一批 （pro在读，这边在处理。）
        pTask->run_task(input_vec);   // trainer处理输入数据。传入数据input_vec是vector<string>&，包含训练样本
                                         // 预测器相同。去处理读入的每个预测样本
        if(finished_flag) break;
    }
    // 如果这一批都处理完了，释放信号量，信号量+1。告诉其他线程可以准备接pro线程读入的下一批数据了。（即使pro没读完，信号量不阻塞。只需要抢锁）
    sem_post(&semCon);
}

