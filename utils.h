#ifndef UTILS_H
#define UTILS_H

#define MY_ASSERT(x) if (!(x)) Asserter(__FILE__, __LINE__);

typedef unsigned int uint;

typedef float ValueType;

void Asserter(const char *file, int line);

void initialize_seed();
double randomize(double min, double max);

class ProfilerPrivate;
class Profiler
{
public:
    enum Options
    {
        Default = 0x0,
        PrintOnDestructor = 0x1
    };

    explicit Profiler(Options opts = Default);

    void start();
    void finish();

    double time() const;
    void print() const;

private:
    ProfilerPrivate *_impl;
};

struct CommandLineArgs
{
    bool test;
    uint ngpu;
    int matrix_size;
    int iter_count;
    int one_gpu_iter_count;
    long long nrun;

    CommandLineArgs()
    {
        // default
        test = false;
        ngpu = 1;
        matrix_size = -1;
        iter_count = 20;
        one_gpu_iter_count = iter_count / 2;
        nrun = static_cast<long long>(-1);
    }

    void parse(int argc, char *argv[]);
    void parseArg(char arg[]);
};

#endif // UTILS_H
