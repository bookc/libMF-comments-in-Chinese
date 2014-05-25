#pragma GCC diagnostic ignored "-Wunused-result" //编译指示
#include <cstdlib>
#include <cmath>
#include <csignal> //定义了程序执行时如何处理不同的信号。信号用作进程间通信， 报告异常行为（如除零）、用户的一些按键组合（如同时按下Ctrl与C键，产生信号SIGINT）.
#include <ctime>
#include <cstring>
#include <climits> //定义了整数类型的一些极限值。
#include <cfloat> //定义了浮点类型的一些极限值。
#include <random> //C++11标准，用于产生随机数
#include <numeric> //定义了C++ STL标准中的基础性的数值算法（均为函数模板）,such as std::inner_product()。
#include <algorithm>
#include <thread> //C++11标准，定义了一些表示线程的类 、互斥访问的类与方法等
#include <chrono> //C++11标准，用于处理时间
#include <mutex>  //C++11标准，定义一些互斥访问的类与方法等
#include <vector>
#include <pmmintrin.h> //用于SSE指令的头文件 mmintrin.h for MMX;xmmintrin.h for SSE; emmintrin.h for SSE2; pmmintrin.h for SSE3
#include <sys/stat.h> //linux中的头文件，获取文件状态
#include <unistd.h> //linux中的头文件
#include <time.h>

#define DATAVER 1
#define MODELVER 1

#define EN_SHOW_SCHED false
#define EN_SHOW_GRID false

#define flag fprintf(stderr, "LINE: %d\n", __LINE__)

enum FileType {DATA,MODEL};

void convert(int argc, char **argv); //文件格式转化函数
void train(int argc, char **argv); //训练函数
void predict(int argc, char **argv); //预测函数
void view(int argc, char **argv); //查看模型或数据文件基本参数信息函数

void exit_file_error(char *path); //文件错误退出函数
void exit_file_ver(float ver); //文件形式（如分为DATAVER，MODELVER)错误退出函数

/**
 * 定义结构体Clock，用于操作过程的耗时统计
 */
struct Clock {
    clock_t begin, end; 
    void tic(); //为成员变量begin赋初值
    float toc(); //为成员变量end赋初值，并计算出开始到结束直接相差的时间，时间单位为秒
};


/**
 * 定义结构体Node，用于存储用户uid对项目iid的评分rate，
 * 即用来存储数据文件中的一行数据。
 */
struct Node {
    int uid, iid; 
    float rate;
};


/**
 * 结构体Matrix，主要于存储数据，即评分矩阵
 */
struct Matrix {
    int nr_us, nr_is;  //分别代表user的数目，item的数目
    long nr_rs; //评分的数目
    float avg; //总体评分的平均值
    Node *M;

    Matrix(); //构造函数
    Matrix(int nr_rs, int nr_us, int nr_is, float avg); //nr_rs应该定义为long型
    Matrix(char *path);
    Matrix(char *path, int *map_u, int *map_i); //适用于建立重排数据的Matrix

    void read_meta(FILE *f); //从二进制文件中读取不包括数组M的其他信息
    void read(char *path); //从二进制文件中读取用户数目等信息
    void write(char *path); //把用户数目等信息写入二进制文件
    void sort(); //借用了STL的std::sort函数
    static bool sort_uid_iid(Node lhs, Node rhs); //充当上面成员函数sort()里调用的std::sort()函数的判别式子（一个返回bool类型的函数）

    ~Matrix();
};


/**
 * 结构体Model,主要用于存储模型的各个参数
 * ，其中模型的有关参数说明可以参照train.cpp文件中的
 * TrainOption::TrainOption函数和TrainOption::exit_train函数
 */
struct Model {
	int nr_us, nr_is, dim, dim_off, nr_thrs, iter, nr_gubs, nr_gibs, *map_uf, *map_ub, *map_if, *map_ib; //有些参数与Matrix中类似，dim表示隐含因子的个数，map_*等用于重排数据。
    //nr_gubs代表user被分成的块数，nr_gibs代表item被分成的块数（即，评分矩阵被分成nr_gubs * nr_gibs 块）

    float *P, *Q, *UB, *IB, lp, lq, lub, lib, glp, glq, glub, glib, gamma, avg; //对应论文中附录A中的公式里的各个参数。如gamma代表学习速率

    bool en_rand_shuffle, en_avg, en_ub, en_ib; //用作标记是否需要重排，损失函数中是否需要平均值，用户偏差数据，项目偏差数据。

    Model();
    Model(char *src);

    void initialize(Matrix *Tr); //在mf.cpp文件中做解释
    void read_meta(FILE *f);
    void read(char *path);
    void write(char *path);
    void gen_rand_map();
    void shuffle();
    void inv_shuffle();

    ~Model();
};


/**
 * 根据模型model，计算节点Node的评分rate。
 * @param  model 模型model
 * @param  r     节点r
 * @return       节点r的评分值
 */
float calc_rate(Model *model, Node *r);

/**
 * 根据模型model，计算整个数据集形成的矩阵R的均方根误差rmse
 * @param  model 模型model
 * @param  R     矩阵R
 * @return       均方根误差值
 */
float calc_rmse(Model *model, Matrix *R);
