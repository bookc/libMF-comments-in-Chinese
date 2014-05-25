#include "mf.h"

/**
 * 退出，并给出相应命令参数信息
 */
void exit_main() {
    printf(
        "usage: mf COMMAND [ARGS]\n"
        "\n"
        "Commands include:\n"
        "    convert    Convert a text file to a binary file\n"
        "    train      Train a model from training data\n"
        "    predict    Predict a test data from a model\n"
        "    view       View model and data info\n"
        "\n"
        "See 'mf COMMAND' for more information on a specific command.\n"
    ); 
    exit(1);
}

/**
 * 主函数，并根据不同的命令参数执行不同的函数
 * @param  argc [description]
 * @param  argv [description]
 * @return      [description]
 */
int main(int argc, char **argv) {

    if(argc<2) exit_main(); //如果参数个数少于2个，则退出，并给出相应命令参数信息

    if(!strcmp(argv[1],"convert")) convert(argc,argv); //执行转化数据文件格式转换函数
    else if(!strcmp(argv[1],"train")) train(argc,argv); //执行训练函数
    else if(!strcmp(argv[1],"predict")) predict(argc,argv); //执行预测函数
    else if(!strcmp(argv[1],"view")) view(argc,argv); //执行查看模型参数或数据统计信息的函数
    else printf("Invalid command: %s\n",argv[1]); //否则，退出

    return 0;
}
