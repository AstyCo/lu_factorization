#ifndef UTILS_H
#define UTILS_H

#define MY_ASSERT(x) if (!(x)) Asserter(__FILE__, __LINE__);

typedef unsigned int uint;

typedef float ValueType;

void Asserter(const char *file, int line);

void initialize_seed();
double randomize(double min, double max);

#endif // UTILS_H
