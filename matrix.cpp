#include "matrix.h"

#include "magma_auxiliary.h"

#include <cstring> // memset
#include <ctime> // time

double Matrix::_minEugenvalue = 0.3;
double Matrix::_maxEugenvalue = 999.9;
double Matrix::_conditionNumber = 100.0001;
int Matrix::pinned_allocation = 0;

Matrix::Matrix(uint n)
    : _n(0), _data(NULL)
{
    allocate(n);
}

Matrix::~Matrix()
{
    clear();
}

Matrix::Matrix(const Matrix &m)
    : _n(0), _data(NULL)
{
    copy(m);
}

const Matrix &Matrix::operator=(const Matrix &m)
{
    copy(m);
    return *this;
}

alglib::real_2d_array Matrix::alglibArray() const
{
    alglib::real_2d_array r;

    double d_array[size()];
    for (uint i = 0; i < size(); ++i)
        d_array[i] = _data[i];

    r.setcontent(_n, _n, d_array);
    return r;
}

void Matrix::rndNondegenirate()
{
    alglib::hqrndstate state;

    // initialize
    alglib::hqrndrandomize(state);
    alglib::ae_int_t seed1 = time(0);

    // generate N*N random values
    for (uint i = 0; i < size(); ++i) {
        alglib::hqrndseed(seed1, i,
                          state); // install seed
        // get the value
        _data[i] = alglib::hqrnduniformr(state);
    }
}

void Matrix::transpose()
{
    for (unsigned i = 0; i < _n; ++i) {
        for (unsigned j = i + 1; j < _n; ++j) {
            std::swap(_data[i * _n + j], _data[j * _n + i]);
        }
    }
}

void Matrix::print(const std::string &extra) const
{
    if (!extra.empty()) {
        char buff[256];
        snprintf(buff, sizeof(buff),
                 "%s %ux%u", extra.c_str(), _n, _n);
        std::cout << std::string(buff) << std::endl;
    }
    std::cout << '[';
    for (uint i = 0; i + 1 < _n; ++i) {
        printRow(i);
        std::cout << ',' << std::endl;
    }
    printRow(_n - 1);
    std::cout << ']' << std::endl;
}

void Matrix::printRow(uint i) const
{
    if (!_data)
        return;
    std::cout << "[ ";
    for (uint j = 0; j + 1 < _n; ++j) {
        std::cout << _data[i * _n + j] << ", ";
    }
    std::cout << _data[i * _n + _n - 1];
    std::cout << " ]";
}

void Matrix::clear()
{
    delete []_data;
}

void Matrix::setAlglibArray(const alglib::real_2d_array &array)
{
    MY_ASSERT(array.rows() == array.cols());

    allocate(array.rows());

    int offset = 0;
    for (uint i = 0; i < _n; ++i) {
        const double *row = array[i];
        for (uint j = 0; j < _n; ++j)
            _data[offset++] = row[j];
    }
}

void Matrix::setDataL(const ValueType *L, uint n)
{
    allocate(n);
    zero();

    for (uint c = 0; c < n; ++c) {
        for (uint r = c + 1; r < n; ++r) {
            uint index_columnmajor = c * _n + r;
            uint index_rowmajor = r * _n + c;
            _data[index_rowmajor] = L[index_columnmajor];
        }
        _data[c * _n + c] = 1;
    }
}

void Matrix::setDataU(const ValueType *U, uint n)
{
    allocate(n);
    zero();

    for (uint c = 0; c < n; ++c) {
        for (uint r = 0; r <= c; ++r) {
            uint index_columnmajor = c * _n + r;
            uint index_rowmajor = r * _n + c;
            _data[index_rowmajor] = U[index_columnmajor];
        }
    }
}

void Matrix::setPermutations(const uint permutations[], uint n)
{
    allocate(n);
    zero();

    for (uint r = 0; r < _n; ++r) {
        uint c = permutations[r] - 1;
        uint index = r * _n + c;

        _data[index] = 1;
    }
}

//void Matrix::interchangeRows(const uint permutations[])
//{
//    for (uint r = 0; r < _n; ++r) {
//        int interchange_with_row = permutations[r];
//        if (static_cast<int>(r) == interchange_with_row)
//            continue; // не переставлять строчки

//        ValueType tmp[_n];

//        ValueType *first_row = _data + r * _n;
//        ValueType *second_row = _data + interchange_with_row * _n;
//        size_t rowsize = _n * sizeof(ValueType);

//        memcpy(tmp, first_row, rowsize);
//        memcpy(first_row, second_row, rowsize);
//        memcpy(second_row, tmp, rowsize);
//    }
//}

void Matrix::allocate(uint n)
{
    if (_n == n)
        return;

    clear();
    _n = n;
    if (n == 0)
        _data = NULL;
    else {
        if (pinned_allocation) {
            if (magma_smalloc_pinned(&_data, size()) != MAGMA_SUCCESS) {
                std::cout << "ERROR magma_smalloc_pinned " << size() << std::endl;
                exit(0);
            }
        }
        else {
            _data = new ValueType[size()];
        }
    }
}

void Matrix::zero()
{
    if (_data)
        memset(_data, 0, sizeof(ValueType) * size());
}

void Matrix::copy(const Matrix &m)
{
    allocate(m.N());
    if (_data)
        memcpy(_data, m._data, size() * sizeof(ValueType));
}

Matrix operator*(const Matrix &l, const Matrix &r)
{
    MY_ASSERT(l.N() == r.N());
    uint n = l.N();
    alglib::real_2d_array larray = l.alglibArray();
    alglib::real_2d_array rarray = r.alglibArray();
    alglib::real_2d_array mul_matrix;
    mul_matrix.setlength(n, n);

    // перемножаем матрицы с помощью функции из alglib
    alglib::rmatrixgemm(n, n, n, 1,
                        larray, 0, 0, 0,
                        rarray, 0, 0, 0, 0,
                        mul_matrix, 0,0);
    Matrix result(n);
    result.setAlglibArray(mul_matrix);

    return result;
}
