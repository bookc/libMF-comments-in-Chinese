#include "mf.h"

//这个文件的结构与convert.cpp基本一致，有些地方的注释请参考convert.cpp文件

/**
 * 结构体PredictOption，主要用于解析predict命令参数，并负责控制训练流程
 */
struct PredictOption {
    char *model_path, *test_path, *dst_path; //分别代表模型文件，测试数据文件，预测结果文件的名称
	PredictOption(int argc, char **argv);	
    static void exit_predict();
	~PredictOption();
};

PredictOption::PredictOption(int argc, char **argv) {
    if(argc!=5 && argc!=4) exit_predict();

    if(!strcmp(argv[1],"help")) exit_predict();

    model_path=argv[3], test_path=argv[2]; //

	if(argc==5) {
		dst_path = new char[strlen(argv[4])];
		sprintf(dst_path,"%s",argv[4]);
	}
	else {
		char *p = strrchr(argv[2],'/');
		if(p==NULL)
			p = argv[2];
		else
			++p;
		dst_path = new char[strlen(p)+5];
		sprintf(dst_path,"%s.out",p);
	}
}


PredictOption::~PredictOption() { 
    delete [] dst_path; 
}

void PredictOption::exit_predict() {
    printf(
    "usage: libmf predict binary_test_file model output\n"
	"\n"
	"Predict a test file from a model\n"
    ); exit(1);
}

/** 
 * 根据模型model，进行预测test_path文件，并把结果保存在dst_path文件中
 * @param model     模型文件
 * @param test_path 测试文件
 * @param dst_path  结果文件
 */
void predict(Model *model, char *test_path, char *dst_path) {

    Matrix *Te = new Matrix(test_path); //

    FILE *f = fopen(dst_path, "w"); 
    double rmse = 0;

    printf("Predicting..."); 
    fflush(stdout); 

    Clock clock; 
    clock.tic();

    for(int rx=0; rx<Te->nr_rs; rx++) {
        float rate = calc_rate(model,&Te->M[rx]); //计算预测评分
        float e = Te->M[rx].rate - rate; //实际评分与预测评分的之差
        fprintf(f,"%f\n",rate); //把预测评分写到结果文件中
        rmse += e*e;
    }

    printf("done. %.2lf\n",clock.toc()); 
    fflush(stdout);

    printf("RMSE: %.4lf\n",sqrt(rmse/Te->nr_rs));

    delete Te;
}

/** 
 * 接受predict命令的参数，并解析参数，控制预测操作的流程
 * @param argc 参数个数
 * @param argv 具体参数信息
 */
void predict(int argc, char **argv) {

    PredictOption *option = new PredictOption(argc,argv); 

    Model *model = new Model(option->model_path); //根据模型model文件名称获取模型的相关信息

    predict(model, option->test_path, option->dst_path); //转到实际预测函数

    delete option; 
    delete model;
}
