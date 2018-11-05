#ifndef TEST_LU_FACTORIZATION_H
#define TEST_LU_FACTORIZATION_H

#include "utils.h"
#include "matrix.h"

// L,U - результаты magma
// M - изначальная матрица
void do_test_lu_factorization(const Matrix &L, const Matrix &U,
                              const int ipiv[], const Matrix &M);

#endif // TEST_LU_FACTORIZATION_H
