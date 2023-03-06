#ifndef LOCK_POOL_H_
#define LOCK_POOL_H_

#include <mutex>
#include <string>

using namespace std;

//负责feature mutex的统一分配
class lock_pool
{
public:
    lock_pool() // 构造函数
    {
        pBiasMutex = new mutex;                 // 互斥锁，控制bias的并发
        pMutexArray = new mutex[lockNum];       // 多个锁，控制array级别的并发
    }

    // 多个fea（hash值相同的话）可能对应某一个锁。获取该fea对应的锁
    inline mutex* get_feature_lock(const string& fea)
    {
        size_t index = strHash(fea) % lockNum;  // 将fea映射为0-100008范围内的hash值
        return &(pMutexArray[index]);           // 获取该fea对应的锁
    }

    // 获取bias的锁
    inline mutex* get_bias_lock()
    {
        return pBiasMutex;
    }

private:
    hash<string> strHash;    // 用来生成string的hash值
    mutex* pBiasMutex;       // 2类互斥锁
    mutex* pMutexArray;
    const int lockNum = 10009;
};



#endif /*LOCK_POOL_H_*/
