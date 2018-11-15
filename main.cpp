#include "test_lu_factorization.h"

#include "magma_types.h" // magma_int_t
#include "magma_s.h" // magma_sgetrf_m
#include "magma_auxiliary.h" // magma_init, magma_finalize, magma_wtime

#include "cuda_runtime_api.h" // cudaGetDeviceCount

#include <list>

#include <ctime>
#include <unistd.h> // sysconf

CommandLineArgs cmd_args;

typedef std::list<uint> ListUint;
static ListUint matrix_sizes()
{
    ListUint sizes;
    for (uint i = 200; i <12000; i *= 1.2)
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

    float start = magma_wtime();
    Profiler prfMatrixGen;
    prfMatrixGen.start();
    Matrix matrix_N_x_N(N);
    matrix_N_x_N.rndNondegenirate();
    prfMatrixGen.finish();
    std::cout << "matrixGen:" << std::endl;
    prfMatrixGen.print();

    cmd_args.ngpu = 1;
    
    for (int iter = 0; iter < cmd_args.iter_count; ++iter) {
        if (iter >= cmd_args.one_gpu_iter_count)
            cmd_args.ngpu = 2;
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
        prf.finish();
        MY_ASSERT(magma_retcode == 0);
        std::cout << N << ":" << std::endl;
        prf.print();
//        std::cout << "PERF: " << static_cast<double>(N) * N * N / (static_cast<double>(100000000) * prf.time())
//                  << std::endl;

        Matrix L;
        L.setDataL(matrix.array(), N);

        Matrix U;
        U.setDataU(matrix.array(), N);

        if (cmd_args.test)
            do_test_lu_factorization(L, U, ipiv, matrix_N_x_N);
    }
    float total = magma_wtime() - start;
    std::cout << "TOTAL: " << total << " s." << std::endl;
}

int main(int argc, char *argv[])
{
    cmd_args.parse(argc, argv);

    std::cout << "arg test " << cmd_args.test << std::endl;
    std::cout << "arg ngpu " << cmd_args.ngpu << std::endl;
    std::cout << "arg matrix_size " << cmd_args.matrix_size << std::endl;
    std::cout << "arg iter_count " << cmd_args.iter_count << std::endl;

    int num_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    std::cout << "TESTING " << (cmd_args.test ? "ENABLED" : "DISABLED")
                << std::endl
              << "num_cpu " << num_cpu
                << std::endl
              << "dev_count " << dev_count
                << std::endl;

    MY_ASSERT(magma_init() == MAGMA_SUCCESS);

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
