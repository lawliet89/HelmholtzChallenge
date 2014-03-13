/* Helmholtz Extrusion Challenge Exercise
*
* Advanced Computer Architecture
* 
* Paul Kelly, Gheorghe-Teodor Bercea, Fabio Luporini - Imperial College London - 2014
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <Windows.h>
#include <iostream>

#include "wrappers_kernels.h"
#include "utils.h"

#define LAYERS 101
#define LAYER_HEIGHT 0.1
#define FILE_LHS "lhs_out"
#define FILE_RHS "rhs_out"

#include <cuda_runtime.h>

//#define CHECK_VS_CPU
#define TIME_INDIVIDUAL

void startTimer(LARGE_INTEGER *timer) {
    QueryPerformanceCounter(timer);
}

double getTimer(LARGE_INTEGER StartingTime, LARGE_INTEGER Frequency) {
    LARGE_INTEGER EndingTime, ElapsedMicroseconds;
    QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

    return double(ElapsedMicroseconds.QuadPart);
}

//Cuda prototypes (cause I'm too lazy to make a header for now)
int testcuda();
__global__ void wrap_expression_1_GPU(double* __restrict__ outarr, const double* __restrict__ coordarr);
__global__ void wrap_rhs_1_GPU(double* __restrict__ outarr, 
					double* __restrict__ coordarr,
					double* __restrict__ inarr,
					int* __restrict__ sextet_map, int layers);
__global__ void wrap_rhs_GPU(double* __restrict__ outarr, 
					double* __restrict__ coordarr,
					double* __restrict__ inarr,
					int* __restrict__ sextet_map, int layers);

__global__ void wrap_lhs_GPU(double* __restrict__ outarr, 
					double* __restrict__ coordarr,
					int* __restrict__ sextet_map, int layers);

// GLOBALS
LARGE_INTEGER StartingTime;
LARGE_INTEGER Frequency;

int main (int argc, char *argv[]) 
{ 
    double elapsed;

    QueryPerformanceFrequency(&Frequency);

	int nodes, cells, cell_size;
	cudaError e;

	//testcuda();

	if ( argc != 2 ){
		// Print usage
		printf("usage: %s filename \n", argv[0]);
		printf("The filename must be of the form: /Path/To/Meshes/Folder/meshname_without_extension\n");
		exit(0);
	}

	/* 
	* Read in 2D mesh informations, coordinates of the vertices and the
	* map from trinagles to vertices.
	*/
	std::string node_path = str_cat(argv[1], ".node");
	std::string cell_path = str_cat(argv[1], ".ele");
	double * coords_2D = read_coords_2D(node_path.c_str(), &nodes);
	int * map_2D = read_cell_node_map_2D(cell_path.c_str(), &cells, &cell_size);

	/* 
	* 3D coordinate field.
	* stored in triplets [x1 y1 z1 x2 y2 z2 ... ], stored LAYER major order
	* TODO: Change to struct of arrays instead (or 3 separate arrays) to improve memory access on GPU
	*/
	double * coords_3D = extrude_coords(coords_2D, nodes, LAYERS, LAYER_HEIGHT);
	free(coords_2D);
	/* 
	* 3D map from cells to vertices.
	* Stores triangle pairs as sextets of indicies into coords_3D as [x1A x1B y1A y1B z1A z1B  x2A x2B...] = [hexprism1, hexprsm2, etc] (all on base layer)
	* Incrementing ALL indicies by 1 triplet effectively steps a layer forward as coords_3D is layer major.
	* see "arg1_0_vec[0] += _arg1_0_off0_0[0] * 3;" etc in wrappers_kernels
	* TODO: change to SoA instead of sextets
	*/
	int * map_3D = extrude_map(map_2D, cells, cell_size, LAYERS);
	int off_3D[6] = {1, 1, 1, 1, 1, 1};
	free(map_2D);

	size_t expr_size = sizeof(double) * nodes * LAYERS;

	/*
	*	CPU Stuff
	*
	*/
#ifdef CHECK_VS_CPU
	printf("CPU TIME:\n");
#ifndef TIME_INDIVIDUAL 
	startTimer(&StartingTime); 
#endif
	double *expr1 = (double*)malloc(expr_size);
#ifdef TIME_INDIVIDUAL
	printf(" Evaluating expression... ");
	startTimer(&StartingTime);
#endif
    wrap_expression_1(0, cells,
		expr1, map_3D,
		coords_3D, map_3D,
		off_3D, off_3D, LAYERS);
#ifdef TIME_INDIVIDUAL
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);
#endif

	double *expr2 = (double*)malloc(expr_size);
	wrap_zero_1(0, nodes * LAYERS,
		expr2,
		LAYERS);
	
#ifdef TIME_INDIVIDUAL 
	printf(" Interpolate expression... ");
	startTimer(&StartingTime); 
#endif
	wrap_rhs_1(0, cells,
		expr2, map_3D,
		coords_3D, map_3D,
		expr1, map_3D,
		off_3D, off_3D, off_3D, LAYERS);
#ifdef TIME_INDIVIDUAL
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);
#endif
	double *expr3 = expr2;
	double *expr4 = (double*)malloc(sizeof(double) * nodes * LAYERS);

#ifdef TIME_INDIVIDUAL 
	printf(" Assembling right-hand side... ");
	startTimer(&StartingTime); 
#endif
	wrap_zero_1(0, nodes * LAYERS,
		expr4,
		LAYERS);
	wrap_rhs(0, cells,
		expr4, map_3D,
		coords_3D, map_3D,
		expr2, map_3D,
		expr3, map_3D,
		off_3D, off_3D, off_3D, off_3D, LAYERS);

#ifdef TIME_INDIVIDUAL
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);
#endif

	double *expr5 = (double*)malloc(sizeof(double) * nodes * LAYERS);
#ifdef TIME_INDIVIDUAL 
	printf(" Assembling left-hand side... ");
	startTimer(&StartingTime); 
#endif
	wrap_zero_1(0, nodes * LAYERS,
		expr5,
		LAYERS);
	startTimer(&StartingTime);
	wrap_lhs(0, cells,
		expr5, map_3D, map_3D,
		coords_3D, map_3D,
		off_3D, off_3D, off_3D, LAYERS);
#ifdef TIME_INDIVIDUAL
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);
#endif
#ifndef TIME_INDIVIDUAL 
	elapsed = getTimer(StartingTime, Frequency);
	printf("Time: %g s\n", elapsed/1e6);
#endif
#endif // CHECK_VS_CPU

	/**
		CUDA Computation
	**/

	printf("CUDA:\n");

#ifndef TIME_INDIVIDUAL 
	startTimer(&StartingTime); 
#endif

	// Send coords to GPU
	size_t coord_3D_size = sizeof(double) * nodes * 3 * LAYERS;
	double* coords_3DGPU;
	if(e = cudaMalloc(&coords_3DGPU, coord_3D_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	if(e = cudaMemcpy(coords_3DGPU, coords_3D, coord_3D_size, cudaMemcpyHostToDevice)) printf("Cuda error %d on line %d\n", e, __LINE__);

	// Send map to GPU
	size_t map_3D_size = sizeof(int) * cells * cell_size * 2;
	int* map_3DGPU;
	if(e = cudaMalloc(&map_3DGPU, map_3D_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	if(e = cudaMemcpy(map_3DGPU, map_3D, map_3D_size, cudaMemcpyHostToDevice)) printf("Cuda error %d on line %d\n", e, __LINE__);

	// RHS calculation ping-pong buffer1-2. LHS result in buffer3
	double *buffer1, *buffer2, *buffer3;
	if(e = cudaMalloc(&buffer1, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	if(e = cudaMalloc(&buffer2, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	if(e = cudaMalloc(&buffer3, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);

	// Memset is very fast (milliseconds). Buffer1 inital contents generated on device
	if(e = cudaMemset(buffer2, 0, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	if(e = cudaMemset(buffer3, 0, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);

	// set to max L1 cache and minimum shared memory
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// Evaluating Expression
#ifdef TIME_INDIVIDUAL
	printf(" Evaluating expression... ");
	startTimer(&StartingTime);
#endif

	wrap_expression_1_GPU<<<nodes, LAYERS>>>(buffer1, coords_3DGPU);
	if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);

#ifdef TIME_INDIVIDUAL
	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

	printf(" Interpolate expression... ");
	startTimer(&StartingTime); 
#endif

	wrap_rhs_1_GPU<<<cells, LAYERS>>>(buffer2, coords_3DGPU, buffer1, map_3DGPU, LAYERS);
	if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);
	
#ifdef TIME_INDIVIDUAL
	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

	printf(" Assembling right-hand side... ");
	startTimer(&StartingTime); 
#endif

	if(e = cudaMemset(buffer1, 0, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	wrap_rhs_GPU<<<cells, LAYERS>>>(buffer1, coords_3DGPU, buffer2, map_3DGPU, LAYERS);
	if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);
	
#ifdef TIME_INDIVIDUAL
	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

	printf(" Assembling left-hand side... ");
	startTimer(&StartingTime); 
#endif

	wrap_lhs_GPU<<<cells, LAYERS>>>(buffer3, coords_3DGPU, map_3DGPU, LAYERS);
	if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);

#ifdef TIME_INDIVIDUAL
	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);
#endif


	/*
	* RHS and LHS output
	*/

	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);
	
	double *lhs = (double*)malloc(sizeof(double) * nodes * LAYERS);
	if(e = cudaMemcpy(lhs, buffer3, expr_size, cudaMemcpyDeviceToHost)) 
		printf("Cuda error %d on line %d\n", e, __LINE__);
	double *rhs = (double*)malloc(sizeof(double) * nodes * LAYERS);
	if(e = cudaMemcpy(rhs, buffer1, expr_size, cudaMemcpyDeviceToHost)) 
		printf("Cuda error %d on line %d\n", e, __LINE__);
#ifndef TIME_INDIVIDUAL 
	elapsed = getTimer(StartingTime, Frequency);
	printf("Time: %g s\n", elapsed/1e6);
#endif

#ifdef CHECK_VS_CPU
	for (int i = 0; i < nodes * LAYERS; ++i)
	{
		if (std::abs((expr5[i] / lhs[i]) - 1) > 0.000001)
		{
			printf("LHS differs\n");
			break;
		}
	}
	for (int i = 0; i < nodes * LAYERS; ++i)
	{
		if (std::abs((expr4[i] / rhs[i]) - 1) > 0.000001)
		{
			printf("RHS differs\n");
			break;
		}
	}
#endif

	output(FILE_RHS, lhs, nodes * LAYERS, 1);
	output(FILE_LHS, rhs, nodes * LAYERS, 1);
	printf(" Numerical results written to output files.\n");


	cudaFree(coords_3DGPU);
	cudaFree(map_3DGPU);
	cudaFree(buffer1);
	cudaFree(buffer2);
	cudaFree(buffer3);
	free(lhs);
	free(rhs);
	free(coords_3D);
	free(map_3D);

#ifdef CHECK_VS_CPU
	free(expr1);
	free(expr2);
	//free(expr3);
	free(expr4);
	free(expr5);
#endif // CHECK_VS_CPU

	return 0;
}
