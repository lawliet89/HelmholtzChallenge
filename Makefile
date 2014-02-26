CXX = g++
SRC = helmholtz.cpp gettimemicroseconds.cpp utils.cpp wrappers_kernels.cpp
FLAGS = -O3 -std=c++11
LIBS = -lrt -lm

all: compare_dat helmholtz

helmholtz:	helmholtz.cpp
			$(CXX) $(FLAGS) -o ACA2-2014 $(SRC) $(LIBS)

compare_dat:    script/compare.cpp
				$(CXX) script/compare.cpp -o compare_dat -lstdc++

clean:
		rm -f compare_dat compare_bin ACA2-2014

.PHONY: all clean
