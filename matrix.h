#include "utils.h"
#include "alglib/linalg.h"

#include <string>

class Matrix
{
public:
    Matrix(uint n = 0);
    ~Matrix();

    Matrix(const Matrix &m);
    const Matrix &operator=(const Matrix &m);

    // матрица в виде (вещественного) массива
    const ValueType *array() const { return _data;}
    ValueType *array() { return _data;}

    uint N() const { return _n;}
    uint size() const { return _n * _n;}

    // alglib matrix
    alglib::real_2d_array alglibArray() const;

    // генерация невырожденной матрицы NxN
    void rndNondegenirate();

    void transpose();

    // alglib matrix -> матрица в виде (вещественного) массива
    void setAlglibArray(const alglib::real_2d_array &array);

    // устанавливает значения матрицы по значениям массива L из libmagma
    // (единицы не хранятся)
    void setDataL(const ValueType *L, uint n);

    // устанавливает значения матрицы по значениям массива U из libmagma
    void setDataU(const ValueType *U, uint n);

    // устанавливает значения матрицы перестановок
    void setPermutations(const uint permutations[], uint n);

//    void interchangeRows(const uint permutations[]);

    // DEBUG
    void print(const std::string &extra = std::string()) const;
    void printRow(uint i) const;
    //

    static int pinned_allocation;
private:
    void clear(); // очистка памяти
    void allocate(uint n); // выделение памяти
    void zero(); // обнуляет матрицу
    void copy(const Matrix &m); // копирование матрицы

    uint _n; // число строк/столбцов
    ValueType *_data; // матрица в виде массива ROW-MAJOR (magma работает с COLUMN-MAJOR)

    static double _minEugenvalue;    // мин/макс собственное значение (для генерации)
    static double _maxEugenvalue;    // мин/макс собственное значение (для генерации)
    static double _conditionNumber;  // число обусловленности (точность)

    friend Matrix operator*(const Matrix &l, const Matrix &r);
};

Matrix operator*(const Matrix &l, const Matrix &r);
