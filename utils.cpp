#include "utils.h"

#include "cuda_runtime_api.h" // cudaEvent_t, cudaError_t, cudaGetErrorString

#include <iostream>
#include <limits>

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void Asserter(const char *file, int line)
{
    std::cerr << "ASSERT at FILE:" << file << " LINE:"<< line << std::endl;
    exit(1);
}

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

void initialize_seed()
{
    srand(time(NULL));
}

double randomize(double min, double max)
{
    return min + (static_cast<double>(rand()) / RAND_MAX) * (max - min);
}



class ProfilerPrivate
{
public:
    ProfilerPrivate(Profiler::Options opts)
        : _started(false), _cpu_elapsed(0), _gpu_elapsed_ms(0), _opts(opts)
    {

    }

    ~ProfilerPrivate()
    {
        if (_opts & Profiler::PrintOnDestructor) {
            if (_started)
                finish();
            print();
        }
    }

    void start()
    {
        if (_started)
            clear();
        _started = true;

        _cpu_start = clock();

        HANDLE_ERROR(cudaEventCreate(&_cudaStart));
        HANDLE_ERROR(cudaEventCreate(&_cudaStop));

        HANDLE_ERROR(cudaEventRecord(_cudaStart, 0));
    }

    void clear()
    {
        HANDLE_ERROR(cudaEventDestroy(_cudaStart));
        HANDLE_ERROR(cudaEventDestroy(_cudaStop));

        _started = false;
    }

    void finish()
    {
        MY_ASSERT(_started);

        _cpu_elapsed = (clock() - _cpu_start) / CLOCKS_PER_SEC;

        HANDLE_ERROR(cudaEventRecord(_cudaStop, 0));
        HANDLE_ERROR(cudaEventSynchronize (_cudaStop) );

        HANDLE_ERROR(cudaEventElapsedTime(&_gpu_elapsed_ms,
                                          _cudaStart, _cudaStop) );
        clear();
    }

    void print() const
    {
        char buff[256];
        snprintf(buff, sizeof(buff),
                 "Elapsed CPU: %f s.  GPU: %f s.\n",
                 _cpu_elapsed, _gpu_elapsed_ms / 1000);

        std::cout << buff << std::endl;
    }

    double time() const
    {
        return _gpu_elapsed_ms/* - _cpu_elapsed*/;
    }

private:
    bool _started;

    double _cpu_start;
    cudaEvent_t _cudaStart;
    cudaEvent_t _cudaStop;

    float _cpu_elapsed;
    float _gpu_elapsed_ms;

    Profiler::Options _opts;
};

Profiler::Profiler(Profiler::Options opts)
    : _impl(new ProfilerPrivate(opts))
{

}

void Profiler::start()
{
    _impl->start();
}

void Profiler::finish()
{
    _impl->finish();
}

double Profiler::time() const
{
    return _impl->time();
}

void Profiler::print() const
{
    _impl->print();
}
