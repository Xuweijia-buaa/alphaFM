#include "pc_frame.h"
#include<fstream>

// pc_frame类:生产者-消费者模型，含信号量，buffer等。用来创建，运行生产/消费线程，运行不同任务(task).
// 这里是4个成员函数的具体实现

// 初始化：创建用于生产者，消费者的多线程。所有消费者处理结束，即结束。
bool pc_frame::init(pc_task& task, int t_num, int buf_size, int log_num)
{
    // 生产者-消费者模型。一个生产者，多个消费者。
    // 生产者线程：读入单一流，每次读n行到一个buffer,唤醒某个消费者线程具体处理。生产者每读取5000行，被消费者A取走，再读取5000行，被另一消费者B取走。
    // 消费者线程：首先读取buffer中数据到各自私有变量中，再开始处理。处理结束后，等待处理新数据。多个消费者线程，数据并行
    //           读完buffer后，就可以让生产者继续读新数据进来了。
    //           而该线程的处理本身，包含读取的这部分样本的训练逻辑。相当于数据并行训练。所有消费者线程处理结束，训练结束。
    // 计算和读取并行：边计算，边读取新数据。只要buffer中数据空了，生产者就会再读新数据进来。消费者线程总保持计算。除非buffe空了需读取。

    // 从读取消费模型来看，对buffer的读取消费是串行的:
    //      生产者每放一份数据到buffer,需要等一个消费者拿走(此时其他线程阻塞)，才能放一份新数据进去。整个流经过该buffer,从生产者，流到n个消费者中。
    //      每个消费者从buffer中读数据时，阻塞别的消费者，读取只属于自己的数据.
    // 但从数据处理模型上看，消费者是数据并行的。消费者们并行处理流中一部分数据：
    //      每个线程拿到各自数据，开始处理后，就可以让生产者再往新数据给其他线程消费了。多个线程，消费不同次放到buffer中的数据。
    //      线程1的处理时，线程2可以读取新数据并处理，数据的计算同时发生，只是对应buffer不同批次的数据。等效于数据并行。
    //  pro   ->  buffer  ->   con1(取走) -> con1(处理)/pro(放新的) -> buffer -> con2(取走) -> con1(处理)/con2(处理)/pro(放新的) -> (没线程可消费，暂放buffer)
    // 且读取和计算重叠：消费者在计算时，生产者可以读取新的数据。如果所有消费者都在计算，读取的新数据暂时放buffer，相当于提前读好
    //                                                 直到某消费者可计算了，拿去后，生产者又可以很快读取更新的数据。新数据buffer在内存中，比再读io节省时间。
    //                边读取，边计算，使得对数据的读取和对数据的计算是重叠的：
    // 10个消费者线程都取走各自数据，并行处理时，生产者同时读取了最新的5000行。这份新数据，只能等最先处理完的消费者处理。否则待在缓冲区(buffer中)。
    // con1(处理)/con2(处理)/pro(放新的) -> (没线程可消费，暂放buffer) -> con1(处理完，读走)/pro(再放新的) ->  con1(计算)/con2(计算)/buffer(等空出消费者)

    // 首先设置该类的属性。 
    pTask = &task;      // 把一个trainer对象，设置成该类的pTask属性。
                        // 该对象含run_task函数。不同消费者线程调用，来处理不同样本。 
                        // 

    threadNum = t_num;  // 线程数
    bufSize = buf_size; // buffer大小：5000
    logNum = log_num;
    sem_init(&semPro, 0, 1);   // 生产者信号量，初始1。  pshared==0:只能在当前进程的所有线程共享。
    sem_init(&semCon, 0, 0);   // 消费者信号量，初始0

    // 根据设好的属性和任务，新建生产者/消费者线程
    threadVec.clear();
    threadVec.push_back(thread(&pc_frame::pro_thread, this));      // 只用一个生产者线程，统一负责读输入流，并写入buffer。 一次输入流结束，本线程结束。（对应一次训练）
    for(int i = 0; i < threadNum; ++i)                             // 用10个消费者线程，每个处理流中部分数据。
    {
        threadVec.push_back(thread(&pc_frame::con_thread, this));  // this是传入该线程函数的首个参数。该消费者函数，也是本类的成员函数。接收参数(this,arg1,..)
    }                                                              // TODO：如果传入类的成员函数，需要指定类名::f。同时取地址,因为：
    return true;
}

void pc_frame::run()
{
    for(int i = 0; i < threadVec.size(); ++i)
    {
        threadVec[i].join();                        // 主线程阻塞，直到每个线程都执行完毕。（生产者读入流结束，且消费者处理逻辑都执行完）
    }
}


// 生产者线程的逻辑：根据标准输入，生产line到buffer。输入流结束，该线程结束。
// 用信号量和buffer,控制数据流的读入速度: 每个迭代，只放5000行到buffer，让消费者线程们先处理，下个迭代阻塞，不读取新数据，直到buffer被某消费者取走。
//                                   直到输入流结束，结束生产者线程。被主线程join.
// 生产者-消费者模式，可以控制读入数据流的速度。防止读入太快，来不及消费：
//                 当所有消费者都在计算时，本次buffer读入后，无等待的消费者。此时消费较慢，生产者不读入新数据，等至少一个消费者空出来，进入等待队列被调度。
//                 且多个消费者时，数据并行，每个消费者处理读入的不同数据。
void pc_frame::pro_thread()
{
    string line;
    size_t line_num = 0;
    int i = 0;
    bool finished_flag = false;

    string model_path="/media/xuweijia/DATA/代码/github代码学习/大规模分布式系统/alphaFM_old/bin/data/a.txt";
    ifstream in(model_path);       // 可以当做cin用。

    while(true)
    {
        // 生产者信号量。P操作，控制每次写入。 初始是1. 首次进来不阻塞，读5000行。 读入下一个5000行之前，不再往里放(阻塞)，让消费者先消费
        sem_wait(&semPro);                             // P操作：-1.  对信号量的操作都是原子操作

        // 一次从输入流中，读5000行，作为5000个元素，放入buffer.
        bufMtx.lock();                                 // 对buffer操作本身上锁
        for(i = 0; i < bufSize; ++i)                   // 一次从输入流中，读5000行，作为5000个元素，放入buffer.
        {
            //if(!getline(cin, line))                    //  从标准输入,读取一行到到line中。可以从pipe中读取数据,作为标准输入，由getline读取:hadoop fs -cat input.txt | ./hello
            if(!getline(in, line))
            {
                finished_flag = true;                  // 该输入流读完了，生产者线程结束
                break;
            }
            line_num++;
            buffer.push(line);                         // 每行，作为一个元素，写入buffer. 
            if(line_num % logNum == 0)                 // 打日志
                {cout << line_num << " lines finished" << endl;}
        }
        bufMtx.unlock();                                // 释放buffer操作本身的锁

        // 有5000个内容放入了buffer.激活消费者信号量。消费者们可以消费了
        sem_post(&semCon);                              // 消费者信号量加1，可以消费了

        if(finished_flag){break;}                       // 输入流都读完了，生产者线程结束
    }
    in.close();
}

// 每个消费者线程的逻辑：
// 每个迭代,如果拿到了消费权（被唤醒），就消费buffer到vec，并对读到的内容进行处理。
//        拿要消费的数据时，阻塞其他消费者线程;拿完数据，开始处理时，可以让生产者再放新数据进来。如果新数据准备好时，自己也处理完了，下迭代也参与对未来新数据的处理。
// 多个消费者线程，相当于数据并行，处理输入流中，自己获取的这些buffer中的样本。所有线程都结束后，训练结束。
void pc_frame::con_thread()
{
    bool finished_flag = false; // 处理的是否是本次输入流的最后一批。是的话，处理完就结束该线程
    vector<string> input_vec;
    input_vec.reserve(bufSize); // 线程局部变量，保持至少buf大小。使用私有的寄存器，栈内存

    while(true)
    {
        input_vec.clear();

        // 每次buffer满了，交给某消费者线程处理。其他消费者线程都阻塞。如果被唤醒的是该线程：
        sem_wait(&semCon);                        // buffer有内容，消费。s=s-1==0。阻塞其他消费者。其他消费者只能被生产者唤醒。                        

        // 消费完当前buffer，读到input_vec中
        bufMtx.lock();
        for(int i = 0; i < bufSize; ++i)
        {
            if(buffer.empty())                    // buffer不到5000个，说明处理的是最后一批。处理完就结束该消费者线程
            {
                finished_flag = true;
                break;
            }
            input_vec.push_back(buffer.front());  // 把当前buffer中数据，都读到input_vec中(并清空buffer)
            buffer.pop();                         
        }
        bufMtx.unlock();

        // 读完buffer，就可以让生产者放新数据进来了。生产者信号量+1，可读新数据到buffer。
        sem_post(&semPro);               

        // 真正处理这一批。
        // pTask是该类的属性，是一个task对象，含run_task()方法。接收本线程的私有数据。创建的局部变量sample也是线程私有的，在线程栈空间。
        pTask->run_task(input_vec);      // 处理本批输入数据。input_vec每个元素一个样本。包含真正的训练逻辑 TODO （预测器相同。处理读入的每个预测样本）
    
        // 如果新数据已经来到buffer了，激活了消费者信号量： 
        // 如果本线程还在处理这批，等待队列中只有其他消费者线程。新数据只能由其他等待好的线程消费。
        // 如果本线程已经处理完，进入了下一个迭代。该线程也阻塞，作为等待线程，有可能分配到消费下一批数据。
        // 如果所有线程都在处理，buffer只能暂放内存中，分配给空出来的消费者。生产者不继续读取，控制读取的速度，等消费者先消费完之前的。

        if(finished_flag) break;         // 如果这批数据是最后一批，该线程结束。
    }

    // 该线程结束。使消费者信号量+1。
    // 可以唤醒某个阻塞中的消费者。该消费者消费时(先-1)，也会发现buffer不到5000了，同样结束。唤醒其他某个阻塞中的消费者
    // 依次类推。
    // 当某个消费者已经消费到最后一批时，其他消费者仍阻塞着。但生产者不再发新的信号了。
    // 因此该线程结束时，通过信号量+1，依次唤醒其他阻塞的消费者线程，让其他线程都结束。最终所有消费者线程都结束。
    sem_post(&semCon);
}

