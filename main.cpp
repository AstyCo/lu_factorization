#include "test_lu_factorization.h"

#include "magma_types.h" // magma_int_t
#include "magma_s.h" // magma_sgetrf_m
#include "magma_auxiliary.h" // magma_init, magma_finalize, magma_wtime

#include <list>

#include <ctime>

struct CommandLineArgs
{
    bool test;

    CommandLineArgs()
    {
        // default
        test = false;
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
    double s_wc = magma_wtime();
    const magma_int_t ngpu = 2;
    const magma_int_t m = N;
    const magma_int_t n = N;
    const magma_int_t lda = m;
    magma_int_t ipiv[N];
    magma_int_t info;

    Matrix matrix_N_x_N(N);
    matrix_N_x_N.rndNondegenirate();
//    matrix_N_x_N.print("Random matrix");

    Matrix matrix = matrix_N_x_N;
    matrix.transpose();

    double begin_wc = magma_wtime();

    magma_int_t magma_retcode = magma_sgetrf_m(ngpu,
                                               m,
                                               n,
                                               matrix.array(),
                                               lda,
                                               ipiv,
                                               &info);
    MY_ASSERT(magma_retcode == 0);
    double end_wc = magma_wtime();

    Matrix L;
    L.setDataL(matrix.array(), N);
//    L.print("L-matrix");

    Matrix U;
    U.setDataU(matrix.array(), N);
//    U.print("U-matrix");

    if (cmd_args.test)
        do_test_lu_factorization(L, U, ipiv, matrix_N_x_N);

    std::cout << N << ": LU-factorization time"
              << " : " << end_wc - begin_wc << "s. "
              << '[' << magma_wtime() - s_wc << "s.]"
              << std::endl;
}


int main(int argc, char *argv[])
{
    if (argc > 1) {
        cmd_args.test = !strcmp(argv[1], "test");
    }
    std::cout << "TESTING " << (cmd_args.test ? "ENABLED" : "DISABLED")
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
