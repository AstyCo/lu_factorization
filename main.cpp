#include "test_lu_factorization.h"

#include "magma_types.h" // magma_int_t
#include "magma_s.h" // magma_sgetrf_m
#include "magma_auxiliary.h" // magma_init, magma_finalize, magma_wtime

#include "cuda_runtime_api.h" // cudaGetDeviceCount

#include <list>
#include <iostream>
#include <fstream>

#include <ctime>
#include <unistd.h> // sysconf

CommandLineArgs cmd_args;

typedef std::list<uint> ListUint;

bool two_gpu_available;

std::string fname = "last_size.txt";

#include <dirent.h>

const char env_last_size[] = "LAST_SIZE";
static void write_value(uint N)
{
    std::ofstream myfile;
    myfile.open (fname.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (!myfile.is_open()) {
        std::cout << "CAN'T OPEN out FILE " << fname << std::endl;
        exit(0);
    }
    myfile << N << std::endl;
    myfile.close();
}

static uint read_value()
{
    std::ifstream myfile;
    myfile.open (fname.c_str(), std::ifstream::in);
    if (!myfile.is_open()) {
        std::cout << "CAN'T OPEN in FILE " << fname << std::endl;
        exit(0);
    }
    char buff[64];
    myfile.getline(buff, sizeof(buff));
    long tmp = strtol(buff, NULL, 10);
    return tmp;
}

static ListUint matrix_sizes()
{
    ListUint sizes;
    uint last_size = read_value() + 1;
    for (uint i = last_size; i <50000; i += 256)
        sizes.push_back(i);
    return sizes;
}


static void eval_on_size(uint N)
{
    const magma_int_t m = N;
    const magma_int_t n = N;
    const magma_int_t lda = m;
    magma_int_t ipiv[N];
    magma_int_t info;

    cmd_args.ngpu = 1;
    write_value(N);

    for (Matrix matrix_N_x_N(N);;) {
        bool no_error_flag = true;

        matrix_N_x_N.rndNondegenirate();

        for (int iter = 0; iter < cmd_args.iter_count; ++iter) {
            if (iter >= cmd_args.one_gpu_iter_count) {
//                if (!two_gpu_available)
//                    break;
                cmd_args.ngpu = 2;
            }
            Matrix matrix = matrix_N_x_N;
            if (cmd_args.test)
                matrix.transpose();

            Profiler prf;
            prf.start();
            magma_int_t magma_retcode = magma_sgetrf_m(cmd_args.ngpu,
                                                       m,
                                                       n,
                                                       matrix.array(),
                                                       lda,
                                                       ipiv,
                                                       &info);
            if (magma_retcode != 0) {
                no_error_flag = false;
                break;
            }
            prf.finish();

            char out_str[512];
            snprintf(out_str, sizeof(out_str),
                     "###,%u,%u,%lf", matrix.N(), cmd_args.ngpu, prf.time());

            std::cout << out_str << std::endl;

            Matrix L;
            L.setDataL(matrix.array(), N);

            Matrix U;
            U.setDataU(matrix.array(), N);

            if (cmd_args.test)
                do_test_lu_factorization(L, U, ipiv, matrix_N_x_N);
        }
        if (no_error_flag)
            break;
    }
}


int main(int argc, char *argv[])
{
    char fname[512];
    snprintf(fname, sizeof(fname),"outs/OUT_%lld_%d",
             static_cast<long long>(time(0)),
             cmd_args.matrix_size + cmd_args.ngpu);
    std::ofstream out(fname);
    std::cout.rdbuf(out.rdbuf()); //redirect std::cout

    cmd_args.parse(argc, argv);

    std::cout << "arg test " << cmd_args.test << std::endl;
    std::cout << "arg ngpu " << cmd_args.ngpu << std::endl;
    std::cout << "arg matrix_size " << cmd_args.matrix_size << std::endl;
    std::cout << "arg iter_count " << cmd_args.iter_count << std::endl;

    int num_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    two_gpu_available = (dev_count > 1);

    MY_ASSERT(magma_init() == MAGMA_SUCCESS);
    std::cout << "TESTING " << (cmd_args.test ? "ENABLED" : "DISABLED")
              << std::endl
              << "num_cpu " << num_cpu << std::endl
              << "dev_count " << dev_count << std::endl
              << "magma_num_gpus " << magma_num_gpus() << std::endl;
    cudaGetDeviceCount(&dev_count);
//    std::cout << "float size" << sizeof(float) << std::endl;
//    for (int i = 0; i < dev_count; i++) {
//        cudaDeviceProp prop;
//        cudaGetDeviceProperties(&prop, i);
//        printf("Device Number: %d\n", i);
//        printf("  Device name: %s\n", prop.name);
//        printf("  Memory Clock Rate (KHz): %d\n",
//               prop.memoryClockRate);
//        printf("  Memory Bus Width (bits): %d\n",
//               prop.memoryBusWidth);
//        printf("  Peak Memory Bandwidth (GB/s): %f\n",
//               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
//        printf("  sharedMemPerBlock (bytes): %lu\n",
//               prop.sharedMemPerBlock);
//        printf("  totalGlobalMem (bytes): %lu\n",
//               prop.totalGlobalMem);
//        printf("  l2CacheSize (bytes): %d\n",
//               prop.l2CacheSize);
//        printf("  maxThreadsPerBlock: %d\n",
//               prop.maxThreadsPerBlock);
//        printf("  maxThreadsDim: %d %d %d\n",
//               prop.maxThreadsDim[0], prop.maxThreadsDim[1],
//               prop.maxThreadsDim[2]);
//        printf("  sharedMemPerMultiprocessor (bytes): %lu\n",
//               prop.sharedMemPerMultiprocessor);
//        printf("  regsPerMultiprocessor (32-bit): %d\n",
//               prop.regsPerMultiprocessor);
//        printf("  maxGridSize: %d %d %d\n",
//               prop.maxGridSize[0], prop.maxGridSize[1],
//               prop.maxGridSize[2]);


//        printf("\n");
//    }
//    return 0;


    initialize_seed();

    if (cmd_args.matrix_size > 0) {
        eval_on_size(cmd_args.matrix_size);
    }
    else {
        ListUint msizes = matrix_sizes();
        for (ListUint::const_iterator it = msizes.begin();
             it != msizes.end();
             ++it) {
            eval_on_size(*it);
        }
    }

    magma_finalize();

    return 0;
}
