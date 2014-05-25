#include "mf.h"

/**
 * 文件路径名称错误
 * @param path 文件路径名称
 */
void exit_file_error(char *path) { 
    fprintf(stderr,"\nError: Invalid file name %s.\n", path); 
    exit(1); 
}

/**
 * 文件版本错误
 * @param current_ver 当前版本
 * @param file_ver    文件实际的版本
 */
void exit_file_ver(float current_ver, float file_ver) { 
    fprintf(stderr,"\nError: Inconsistent file version.\n");
    fprintf(stderr,"current version:%.2f    file version:%.2f\n",current_ver,file_ver);
    exit(1); 
}

/*----------------以下为结构体Clock部分-----------------------*/

/**
 * 为结构体Clock成员变量begin赋初值
 */
void Clock::tic() { 
    begin = clock(); 
}

/**
 * 为结构体Clock成员变量end赋初值,并计算出开始到结束直接相差的时间，时间单位为秒
 * @return 开始到结束直接相差的时间，时间单位为秒
 */
float Clock::toc() {
    end = clock();
    return (float)(end-begin)/CLOCKS_PER_SEC;
}


/*----------------以下为结构体Matrix部分-----------------------*/

/**
 * 
 */
Matrix::Matrix() {}

Matrix::Matrix(int nr_rs, int nr_us, int nr_is, float avg) : nr_us(nr_us), nr_is(nr_is), nr_rs(nr_rs), avg(avg) { 
    M = new Node[nr_rs]; //用一个一维数组替代矩阵来保存数据
}

Matrix::Matrix(char *path) { 
    read(path); 
}

Matrix::Matrix(char *path, int *map_u, int *map_i) {
    read(path); 
    for(int rx=0; rx<nr_rs; rx++) { 
        M[rx].uid = map_u[M[rx].uid]; //????覆盖问题
        M[rx].iid = map_i[M[rx].iid];
    }
}

/**
 * 从f文件中读取相关Matrix的基本信息
 * @param f 文件
 */
void Matrix::read_meta(FILE *f) {
    int type; 
    float ver;

    fread(&type,sizeof(int),1,f); 

    if(type!=(int)DATA) { 
        fprintf(stderr,"Error: It is not a data file.\n"); 
        exit(1); 
    }

    fread(&ver,sizeof(float),1,f); 

    if(ver!=(float)DATAVER) exit_file_ver(DATAVER,ver);

    fread(this,sizeof(Matrix),1,f); //

    this->M = NULL; //
}

/**
 * 从path文件中读取相关Matrix的信息
 * @param path 文件名称
 */
void Matrix::read(char *path) {
    printf("Reading from %s...",path); 
    fflush(stdout);

    Clock clock; 
    clock.tic();

    FILE *f = fopen(path, "rb"); 
    if(!f) exit_file_error(path);

    read_meta(f);

    M = new Node[nr_rs];

    fread(M,sizeof(Node),nr_rs,f);
    fclose(f);

    printf("done. %.2f\n",clock.toc()); 
    fflush(stdout);
}

/**
 * 把存储数据的Matrix以二进制格式写道ptah代表的文件中
 * @param path 文件名称
 */
void Matrix::write(char *path) {
    printf("Writing %s... ",path); 
    fflush(stdout);

    Clock clock; 
    clock.tic();

    FILE *f = fopen(path,"wb"); 
    if(!f) exit_file_error(path);

    float ver = (float)DATAVER;
    int file_type = DATA;

    fwrite(&file_type,sizeof(int),1,f);
    fwrite(&ver,sizeof(float),1,f);
    fwrite(this,sizeof(Matrix),1,f); 
    fwrite(M,sizeof(Node),nr_rs,f); //注意上面已经写了this，但是这里需要写M
    fclose(f);

    printf("done. %.2f\n",clock.toc()); 
    fflush(stdout);
}

void Matrix::sort() { 
    std::sort(M,M+nr_rs,Matrix::sort_uid_iid); 
}

bool Matrix::sort_uid_iid(Node lhs, Node rhs) { 
    if(lhs.uid!=rhs.uid) 
        return lhs.uid < rhs.uid; 
    else 
        return lhs.iid < rhs.iid; 
}

Matrix::~Matrix() { 
    delete [] M; 
}

/*----------------以下为结构体Model部分-----------------------*/

Model::Model() {}

Model::Model(char *path) { 
    read(path); 
}

/**
 * 初始化论文附录A中公式里相关参数
 * @param Tr 评分矩阵
 */
void Model::initialize(Matrix *Tr) {
    printf("Initializing model..."); 
    fflush(stdout);

    Clock clock; 
    clock.tic();

    nr_us = Tr->nr_us; //nr_us, 用户user个数 (论文附录A中m）。
    nr_is = Tr->nr_is; //nr_is, 项目item个数 (论文附录A中n）。
    glp = 1 - gamma * lp; //glp, 正则化的参数lambda P (论文附录A中）。
    glq = 1 - gamma * lq; //glq, 正则化的参数lambda Q (论文附录A中）。
    glub = 1 - gamma * lub; //glub, 正则化的参数lambda a (论文附录A中）。
    glib = 1 - gamma * lib; //glib, 正则化的参数lambda b (论文附录A中）。
    dim_off = dim%4? (dim/4)*4+4 : dim; //dim_off代表隐含因子，而不是dim??????。
    avg = en_avg? Tr->avg : 0.0; //所有评分数据中的平均值 (论文附录A中）。

    //用随机数初始化隐含因子矩阵P（论文附录A中的P）。
    srand48(0L); //linux中的函数，为drand48设置随机种子。
    P = new float[nr_us*dim_off]; //这里又用一维数组代替nr_us * dim_off维的矩阵，而没有用二维数组。
    float *p = P;

    for(int px=0; px<nr_us; px++) {
        for(int dx=0; dx<dim; dx++) *(p++)=0.1*drand48(); //drand48()借用线性同余法和48位整数运算，产生伪随机数。并且返回双精度的均匀分布在[0.0,1.0)之间的随机数。

        for(int dx=dim; dx<dim_off; dx++) *(p++) = 0;
    }

    //用随机数初始化隐含因子矩阵Q（论文附录A中的Q）。
    srand48(0L);
    Q = new float[nr_is*dim_off]; //同上
    float *q = Q;
    for(int qx=0; qx<nr_is; qx++) {
        for(int dx=0; dx<dim; dx++) *(q++)=0.1*drand48();

        for(int dx=dim; dx<dim_off; dx++) *(q++) = 0;
    }

    //用0初始化用户user偏差向量a（论文附录A中的a）。
    if(en_ub) {
        UB = new float[nr_us];
        for(int ubx=0; ubx<nr_us; ubx++) UB[ubx] = 0;
    }
    
    //用0初始化项目item偏差向量b（论文附录A中的b）。
    if(en_ib) {
        IB = new float[nr_is];
        for(int ibx=0; ibx<nr_is; ibx++) IB[ibx] = 0;
    }

    printf("done. %.2f\n", clock.toc()); 
    fflush(stdout);
}

/**
 * 读取相关模型文件的元参数，如type
 * @param f 模型文件
 */
void Model::read_meta(FILE *f) {
    int type; 
    float ver;

    fread(&type,sizeof(int),1,f);
     if(type!=(int)MODEL) { 
        fprintf(stderr,"Error: It is not a model file.\n"); 
        exit(1); 
    }

    fread(&ver, sizeof(float), 1, f); 
    if(ver!=(float)MODELVER) exit_file_ver(MODELVER,ver);

    fread(this,sizeof(Model),1,f); //this 

    this->P = NULL; this->Q = NULL;
}

/**
 * 读取相关模型文件的信息
 * @param path 模型文件路径
 */
void Model::read(char *path) {

    printf("Reading model..."); 
    fflush(stdout);

    Clock clock; 
    clock.tic();
    FILE *f = fopen(path,"rb"); 
    if(!f) exit_file_error(path);

    read_meta(f); //

    P = new float[nr_us*dim_off]; 
    Q = new float[nr_is*dim_off];
    fread(P, sizeof(float), nr_us*dim_off, f);
    fread(Q, sizeof(float), nr_is*dim_off, f);

	if(en_ub) {
		UB = new float[nr_us];
		fread(UB, sizeof(float), nr_us, f);
	}

	if(en_ib) {
		IB = new float[nr_is];
		fread(IB, sizeof(float), nr_is, f);
	}

	if(en_rand_shuffle) {

        map_uf = new int[nr_us]; 
        map_ub = new int[nr_us]; 
        map_if = new int[nr_is]; 
        map_ib = new int[nr_is]; 

		fread(map_uf, sizeof(int), nr_us, f);
		fread(map_ub, sizeof(int), nr_us, f);
		fread(map_if, sizeof(int), nr_is, f);
		fread(map_ib, sizeof(int), nr_is, f);
	}

    fclose(f);

    printf("done. %.2f\n",clock.toc()); 
    fflush(stdout);
}

/**
 * 写入模型数据
 * @param path 模型文件的存储路径
 */
void Model::write(char *path) {

    printf("Writing model..."); 
    fflush(stdout);

    Clock clock; 
    clock.tic();

    FILE *f = fopen(path, "wb"); 
    if(!f) exit_file_error(path); 

    float ver = (float)MODELVER; 
    int file_type = MODEL;

    //开始写
    fwrite(&file_type,sizeof(int),1,f);
    fwrite(&ver,sizeof(float),1,f);
    fwrite(this,sizeof(Model),1,f);
    fwrite(P,sizeof(float),nr_us*dim_off,f);
    fwrite(Q,sizeof(float),nr_is*dim_off,f);
	if(en_ub) fwrite(UB, sizeof(float), nr_us, f);
	if(en_ib) fwrite(IB, sizeof(float), nr_is, f);
    
	if(en_rand_shuffle) {
		fwrite(map_uf, sizeof(int), nr_us, f);
		fwrite(map_ub, sizeof(int), nr_us, f);
		fwrite(map_if, sizeof(int), nr_is, f);
		fwrite(map_ib, sizeof(int), nr_is, f);
	}
    fclose(f);

    printf("done. %.2f\n", clock.toc()); 
    fflush(stdout);
}

/**
 * 为map_*先按顺序产生数据，然后再重排。
 */
void Model::gen_rand_map() {

    map_uf = new int[nr_us]; 
    map_ub = new int[nr_us]; 
    map_if = new int[nr_is]; 
    map_ib = new int[nr_is];

    for(int ix=0; ix<nr_us; ix++) map_uf[ix] = ix; 
    for(int ix=0; ix<nr_is; ix++) map_if[ix] = ix;

    std::random_shuffle(map_uf,map_uf+nr_us); //随机重排map_uf里面的数据。
    std::random_shuffle(map_if,map_if+nr_is);

    for(int ix=0; ix<nr_us; ix++) map_ub[map_uf[ix]] = ix; 
    for(int ix=0; ix<nr_is; ix++) map_ib[map_if[ix]] = ix;
}


/**
 * 借用上面重排的map_*f数据，重排P，Q，UB,UI。
 */
void Model::shuffle() {
	float *P1 = new float[nr_us*dim_off]; 
    float *Q1 = new float[nr_is*dim_off]; 
    float *UB1 = new float[nr_us]; 
    float *IB1 = new float[nr_is];

	for(int px=0; px<nr_us; px++) 
        std::copy(&P[px*dim_off], &P[px*dim_off+dim_off], &P1[map_uf[px]*dim_off]);

	for(int qx=0; qx<nr_is; qx++) 
        std::copy(&Q[qx*dim_off], &Q[qx*dim_off+dim_off], &Q1[map_if[qx]*dim_off]);

	delete [] P; 
    delete [] Q; 
    P = P1; 
    Q = Q1;

    if(en_ub) {
        for(int px=0; px<nr_us; px++) 
            UB1[map_uf[px]] = UB[px];
        delete [] UB; 
        UB = UB1;
    }

    if(en_ib) {
        for(int qx=0; qx<nr_is; qx++) 
            IB1[map_if[qx]] = IB[qx];
        delete [] IB; 
        IB = IB1;
    }
}


/**
 * 借用上面重排的map_*b数据，反向重排P，Q，UB,UI。
 */
void Model::inv_shuffle() {
	float *P1 = new float[nr_us*dim_off]; 
    float *Q1 = new float[nr_is*dim_off]; 
    float *UB1 = new float[nr_us]; 
    float *IB1 = new float[nr_is];

	for(int px=0; px<nr_us; px++) 
        std::copy(&P[px*dim_off], &P[px*dim_off+dim_off], &P1[map_ub[px]*dim_off]); //注意这里是用map_ub,上面的函数用的是map_uf。

	for(int qx=0; qx<nr_is; qx++) 
        std::copy(&Q[qx*dim_off], &Q[qx*dim_off+dim_off], &Q1[map_ib[qx]*dim_off]);

	delete [] P; 
    delete [] Q; 
    P = P1; 
    Q = Q1;

    if(en_ub) {
        for(int px=0; px<nr_us; px++) UB1[map_ub[px]] = UB[px];
        delete [] UB; 
        UB = UB1;
    }
    if(en_ib) {
        for(int qx=0; qx<nr_is; qx++) IB1[map_ib[qx]] = IB[qx];
        delete [] IB; 
        IB = IB1;
    }
}

/**
 * 析构函数
 */
Model::~Model() { 
    delete [] P; delete [] Q; delete [] map_uf; delete [] map_ub; delete [] map_if; delete [] map_ib; 
    if(en_ub) delete [] UB;
    if(en_ib) delete [] IB;
}


/**
 * 根据模型model，计算节点Node的评分rate。
 * @param  model 模型model
 * @param  r     节点r
 * @return       节点r的评分值
 */
float calc_rate(Model *model, Node *r) { 
    float rate = std::inner_product(&model->P[r->uid*model->dim_off], 
                                    &model->P[r->uid*model->dim_off]+model->dim, 
                                    &model->Q[r->iid*model->dim_off], 
                                    0.0)
                 + model->avg;

    if(model->en_ub) rate += model->UB[r->uid]; //如果en_ub为true，则预测评分加上偏差。
    if(model->en_ib) rate += model->IB[r->iid];

    return rate;
}


/**
 * 根据模型model，计算整个数据集形成的矩阵R的均方根误差rmse
 * @param  model 模型model
 * @param  R     矩阵R
 * @return       均方根误差值
 */
float calc_rmse(Model *model, Matrix *R) {
    double loss=0; 
    float e; 

    for(int rx=0; rx<R->nr_rs; rx++) { 
        e = R->M[rx].rate - calc_rate(model,&R->M[rx]); 
        loss += e*e; 
    }

    return sqrt(loss/R->nr_rs);
}
