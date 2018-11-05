#include "utils.h"

#include <iostream>
#include <limits>

#include <cstdlib>
#include <ctime>

void Asserter(const char *file, int line)
{
    std::cerr << "ASSERT at FILE:" << file << " LINE:"<< line << std::endl;
    exit(1);
}

void initialize_seed()
{
    srand(time(NULL));
}

double randomize(double min, double max)
{
    return min + (static_cast<double>(rand()) / RAND_MAX) * (max - min);
}
