#include "mf.h"

/**
 * 结构体ConvertOption，主要用于解析convert命令参数，并负责控制训练流程
 */
struct ConvertOption {
    char *src, *dst; //src代表格式转化前的文件，dst代表格式转化后的文件
    ConvertOption(int argc, char **argv);	
    static void exit_convert();
    ~ConvertOption();
};


/**
 * 构造函数
 * 通过命令行参数初始化src和dst
 */
ConvertOption::ConvertOption(int argc, char **argv) {
    if(argc!=3 && argc!=4) exit_convert(); //参数个数不满足转化命令的格式，则退出

    src = argv[2]; //源文件为转化命令中第3个参数
    if(argc==4) { //如果有第4个参数
        dst = new char[strlen(argv[3])+1]; //源文件为转化命令中第4个参数，记得大小+1
        sprintf(dst,"%s",argv[3]); 
    }
    else { //如果没有第4个参数，则自动构建格式转化后的文件dst名称
		char *p = strrchr(argv[2],'/'); //找到src中最后一个'/'
		if(p==NULL) p = argv[2]; //如果src中没有'/'，则表明src中没有包括路径
		else p++; //如果src中有'/'，让p指向'/'后面紧挨的一个字母
        dst = new char[strlen(p)+5]; //这里+5，是要存储下面一行的".bin"和一个'\0'
		sprintf(dst,"%s.bin",p);
    }
}

/** 
 * 转化命令参数格式不对时的退出函数
 */
void ConvertOption::exit_convert() {
    printf(
        "usage: libmf convert text_file binary_file\n"
        "\n"
        "Convert a text file to a binary file\n"
    ); 
    exit(1);
}

/**
 * 析构函数
 */
ConvertOption::~ConvertOption() { 
    delete[] dst; //前面只有dst进行了new操作（src没有），所以只需进行delete[] dst
}

/**
 * 执行文件格式转换过程，把原数据文件，转换成由定义好的Matrix结构形式的2进制文件
 * @param src_path 格式转化前的文件
 * @param dst_path 格式转化后的文件
 */
void convert(char *src_path, char *dst_path) {
    printf("Converting %s... ", src_path); 
    fflush(stdout);

    Clock clock; 
    clock.tic();

    //uid，iid分别用于存储数据文件中一行中的user id，item id
    //nr_us, nr_is, nr_rs用于存储数据文件中所有的用户数目，item数目，评分数目
    int uid, iid, nr_us=0, nr_is=0, nr_rs; 
    float rate; //rate用于存储数据文件中一行中的rate
    double sum = 0; //sum用于存储数据文件中所有的rate之和

    FILE *f = fopen(src_path, "r"); 
    if(!f) exit_file_error(src_path);

    std::vector<Node> rs; //存储数据文件中的所有数据

    while(fscanf(f,"%d %d %f\n",&uid,&iid,&rate) != EOF) { //一行行扫描数据文件
        if(uid+1>nr_us) nr_us = uid+1; //注意这种统计用户数目的方法
        if(iid+1>nr_is) nr_is = iid+1; 
        sum += rate;
        Node r; 
        r.uid=uid, r.iid=iid, r.rate=rate; //
        rs.push_back(r);
    }

    nr_rs = rs.size(); 
    fclose(f);

    Matrix *R = new Matrix(nr_rs, nr_us, nr_is, sum/nr_rs);

    //auto 为C++11中的新特性，auot it 等价于std::vector<Node>::iterator it
    for(auto it=rs.begin(); it!=rs.end(); it++) R->M[it-rs.begin()] = (*it); //把rs中的数据复制到R->M中

    printf("done. %.2f\n", clock.toc()); 
    fflush(stdout);
    
    R->write(dst_path);

    delete R;
}

/**
 * 接受convert转换命令的函数，并解析参数，控制转换操作的流程
 * @param argc 参数个数
 * @param argv 具体参数
 */
void convert(int argc, char **argv) {
    ConvertOption *option = new ConvertOption(argc,argv);

    convert(option->src,option->dst); //传入格式转化前的文件名，格式转化后的文件名

    delete option;
}
