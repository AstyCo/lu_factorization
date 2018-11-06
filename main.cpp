#include "test_lu_factorization.h"

#include "magma_types.h" // magma_int_t
#include "magma_s.h" // magma_sgetrf_m
#include "magma_auxiliary.h" // magma_init, magma_finalize, magma_wtime

#include "cuda_runtime_api.h" // cudaGetDeviceCount

#include <list>

#include <ctime>
#include <unistd.h> // sysconf

struct CommandLineArgs
{
    bool test;
    int ngpu;

    CommandLineArgs()
    {
        // default
        test = false;
        ngpu = 1;
    }
} cmd_args;

typedef std::list<uint> ListUint;
static ListUint matrix_sizes()
{
    ListUint sizes;
    for (uint i = 1000; i <1010; ++i)
        sizes.push_back(i);
    for (uint i = 200; i <210; ++i)
        sizes.push_back(i);
    for (uint i = 2000; i <2005; ++i)
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

    Matrix matrix_N_x_N(N);
    matrix_N_x_N.rndNondegenirate();

    for (int iter = 0; iter < 10; ++iter) {
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

        Matrix L;
        L.setDataL(matrix.array(), N);

        Matrix U;
        U.setDataU(matrix.array(), N);

        if (cmd_args.test)
            do_test_lu_factorization(L, U, ipiv, matrix_N_x_N);
    }
}


int main(int argc, char *argv[])
{
    if (argc > 1) {
        cmd_args.test = !strcmp(argv[1], "test");
        std::cout << "arg test " << cmd_args.test << std::endl;

    }
    if (argc > 2) {
        int tmp = strtol(argv[2], NULL, 10);
        if (tmp >= 1 && tmp <= 2)
            cmd_args.ngpu = tmp;
        std::cout << "arg ngpu " << cmd_args.ngpu << std::endl;
    }
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
    ListUint msizes = matrix_sizes();

    for (ListUint::const_iterator it = msizes.begin();
         it != msizes.end();
         ++it)
    {
        eval_on_size(*it);
    }

    magma_finalize();

    return 0;
}
