CC=g++
CFLAGS=-c -Wall -O3
INC=/polusfs/soft/magma-2.4.0_open_blas/include /polusfs/soft/PGI/linuxpower/2018/cuda/9.1/include
INC_PARAMS=$(foreach d, $(INC), -I$d)
LDFLAGS=-L /polusfs/soft/magma-2.4.0_open_blas/lib -lmagma -L /polusfs/soft/PGI/linuxpower/2018/cuda/9.1/lib64 -lcudart -lcublas -L /polusfs/soft/openblas-0.3.4/lib -lopenblas
SOURCES=main.cpp utils.cpp matrix.cpp test_lu_factorization.cpp $(wildcard alglib/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=t1
ARGUMENTS=

all: $(SOURCES) $(EXECUTABLE)
	
clean: 
	rm -rf *.o
	
submit:
	mpisubmit.pl -n 1 -w 00:30 -g $(EXECUTABLE) $(ARGUMENTS)
	
run:
	$(EXECUTABLE) $(ARGUMENTS)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(INC_PARAMS) $(CFLAGS) $< -o $@
	