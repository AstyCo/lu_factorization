#include "test_lu_factorization.h"

#include <vector>

#define ABS(x) ((x) > 0 ? (x) : (-(x)))

typedef std::vector<uint> VecUint;

static VecUint getPermutations(const int ipiv[], uint N)
{
    VecUint v;
    v.reserve(N);

    for (uint i = 0; i < N; ++i)
        v.push_back(i);

    for (uint i = 0; i < N; ++i)
        std::swap(v[i], v[ipiv[i] - 1]);

    VecUint perms;
    perms.reserve(N);

    for (uint i = 0; i < N; ++i)
        perms[v[i]] = i + 1;

    return perms;
}

void do_test_lu_factorization(const Matrix &L, const Matrix &U,
                              const int ipiv[], const Matrix &M)
{
    MY_ASSERT(L.N() == U.N() && L.N() == M.N());
    uint N = L.N();
    Matrix P;
    VecUint permutations = getPermutations(ipiv, N);
//    for (int i = 0; i < N; ++ i)
//        std::cout << permutations[i] << ',';
//    std::cout << std::endl;

    P.setPermutations(permutations.data(), N);
//    P.print("Permutations");
    Matrix LU = L * U;
//    LU.print("LU");
    Matrix PLU = P * LU;
//    M.print("M");
//    PLU.print("PLU");

    const ValueType *array_PLU = PLU.array();
    const ValueType *array_M = M.array();

    for (uint i = 0; i < N * N; ++i) {
        static double eps = 0.01;
        ValueType delta = ABS(array_PLU[i] - array_M[i]);
        if(delta > eps) {
            std::cout << delta << std::endl;
            MY_ASSERT(false);
        }
    }
}
