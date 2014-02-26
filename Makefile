CPP = gcc
SRC = helmholtz.c gettimemicroseconds.c utils.c wrappers_kernels.c
FLAGS = -O3 -std=c99
LIBS = -lrt -lm

all: compare_dat helmholtz

helmholtz:	helmholtz.c
			gcc $(FLAGS) -o ACA2-2014 $(SRC) $(LIBS)

compare_dat:    script/compare.cpp
				$(CPP) script/compare.cpp -o compare_dat -lstdc++

clean:
		rm -f compare_dat compare_bin ACA2-2014

.PHONY: all clean