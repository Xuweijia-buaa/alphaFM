#ifndef MEM_POOL_H_
#define MEM_POOL_H_


class mem_pool
{
public:
    static void* get_mem(size_t size)
    // 如果首次申请/本次申请的size大于剩余的了，malloc,用来放新的特征unit
    // 否则不申请，只从之前申请的内存中，剩下的[pbegin+size -pend]中,再切一块size大小的，给map的内存分配器即可。map用完该size后，在调用。
    {
        if(size > blockSize) return NULL;
        if(NULL == pBegin || (char*)pBegin + size > pEnd) // 首次申请或已有的不够了（调用map[f]时，调用自定义内存分配器，申请一块较大堆内存，放参数单元）
        {
            pBegin = malloc(blockSize);                  // 统一申请64k的堆内存
            pEnd = (char*)pBegin + blockSize;           
        }
        void* res = pBegin;                              // 新申请的首地址/上次申请size后的最后一个地址。本次size够， 返回这段即可。                             
        pBegin = (char*)pBegin + size;                   // 记录，本次就只用pbegin-pbegin+size。
                                                         // 下次再要新的size,如果上次申请的64k够，从pbegin+size -pend中，再取size即可。不必再malloc。返回这段首地址
        return res;                                      // 返回新申请的首地址/ 上次
    }
private:
    mem_pool() {}
    static void* pBegin;
    static void* pEnd;
    static const size_t blockSize = 64 * 1024 * 1024;// 64k
};

void* mem_pool::pBegin = NULL;      // 每次申请的堆内存(64k)，开始和结束位置.  每次一定的size后，下次
void* mem_pool::pEnd = NULL;


#endif /*MEM_POOL_H_*/
