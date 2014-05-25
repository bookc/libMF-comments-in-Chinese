#include "mf.h"

//这个文件的结构与convert.cpp基本一致，有些地方的注释请参考convert.cpp文件
//主要查看转换格式后的数据文件 或者 查看模型文件的基本信息

/**
 * 结构体ViewOption，主要用于解析view命令参数，并负责控制训练流程
 */
struct ViewOption {
    char *src;
	ViewOption(int argc, char **argv);	
    static void exit_view();
};

void ViewOption::exit_view() {
    printf(
    "usage: libmf view file\n"
	"\n"
	"View info in a binary data or model file\n"
    ); 

    exit(1);
}

ViewOption::ViewOption(int argc, char **argv) {
    if(argc!=3) exit_view();

    if(!strcmp(argv[1],"help")) exit_view();

    src = argv[2];

    FILE *f = fopen(src, "rb"); 
    if(!f) exit_file_error(src); //这个函数在mf.cpp文件中

    fclose(f);
}

/**
 * 查看数据文件的信息
 * @param f 文件的名称
 */
void view_data(FILE *f) {
    Matrix *R = new Matrix;

    fseek(f,0,SEEK_SET); //
    R->read_meta(f); //

    printf("number of users = %d\n", R->nr_us);
    printf("number of items = %d\n", R->nr_is);
    printf("number of ratings = %ld\n", R->nr_rs);
    printf("rating average = %f\n", R->avg);
}

void view_model(FILE *f) {
    Model *model = new Model; 

    fseek(f,0,SEEK_SET); //SEEK_SET是定义在<cstdio>中的Macros（宏指令），代表Beginning of file
    model->read_meta(f); //

    printf("dimensions = %d\n", model->dim);
    printf("iterations = %d\n", model->iter);
    printf("lambda p = %f\n", model->lp);
    printf("lambda q = %f\n", model->lq);
    if(model->en_ub) printf("lambda user bias = %f\n", model->lub); //从demo.sh运行结果来看，demo中并没有采用用户（项目）偏差
    if(model->en_ib) printf("lambda item bias = %f\n", model->lib);
    printf("gamma = %f\n", model->gamma);
    printf("random shuffle = %d\n", (int)model->en_rand_shuffle); //在mf.h中Model类中，en_rand_shuffle为bool型
    printf("use average = %d\n", (int)model->en_avg); //在mf.h中Model类中，en_avg为bool型
}

/** 
 * 接受view查看命令的参数，并解析参数，控制查看操作的流程
 * @param argc 参数个数
 * @param argv 具体参数信息
 */
void view(int argc, char **argv) {

    ViewOption option(argc,argv); //这个地方与convert.cpp,predict.cpp格式不一致，后者用的是指针。

    FILE *f = fopen(option.src, "rb"); 

    int type;

    fread(&type,sizeof(int),1,f);

    if(type==DATA) view_data(f);
    else if(type==MODEL) view_model(f);
    else fprintf(stderr,"Invalid file type.\n");

    fclose(f);
}
