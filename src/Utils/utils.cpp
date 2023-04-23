#include <cmath>
#include <stdlib.h>
#include "utils.h"

const double kPrecision = 0.0000000001;
// 按分隔符分割string
void utils::split_string(string& line, char delimiter, vector<string>* r)
{
    int begin = 0;
    for(int i = 0; i < line.size(); ++i)
    {
        if(line[i] == delimiter)  // 每遇到一个分隔符，记录[begin,此刻]
        {
            (*r).push_back(line.substr(begin, i - begin));  // start from begin ,length==i-begin   *r:该vector
            begin = i + 1;
        }
    }
    // 最后一个substr
    if(begin < line.size())
    {
        (*r).push_back(line.substr(begin, line.size() - begin));
    }
}


int utils::sgn(double x)
{
    if(x > kPrecision) return 1;
    else return -1;
}


double utils::uniform()
{
    return rand()/((double)RAND_MAX + 1.0);
}


double utils::gaussian()
{
    double u,v, x, y, Q;
    do
    {
        do 
        {
            u = uniform();
        } while (u == 0.0); 

        v = 1.7156 * (uniform() - 0.5);
        x = u - 0.449871;
        y = fabs(v) + 0.386595;
        Q = x * x + y * (0.19600 * y - 0.25472 * x);
    } while (Q >= 0.27597 && (Q > 0.27846 || v * v > -4.0 * u * u * log(u)));
    return v / u;
}


double utils::gaussian(double mean, double stdev)
{
    if(0.0 == stdev)
    {
        return mean;
    }
    else
    {
        return mean + stdev * gaussian();
    }
}


// agrc:参数数目
// 
vector<string> utils::argv_to_args(int argc, char* argv[])
{
    vector<string> args;
    for(int i = 1; i < argc; ++i)
    {
        args.push_back(string(argv[i]));
    }
    return args;
}

