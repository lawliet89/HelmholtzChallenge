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

#define CHECK_VS_CPU

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

int main (int argc, char *argv[]) 
{ 
    LARGE_INTEGER StartingTime;
    LARGE_INTEGER Frequency;
    double elapsed;

    QueryPerformanceFrequency(&Frequency);

	int nodes, cells, cell_size;
	cudaError e;

	testcuda();

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

	/*
	* Helmholtz Assembly
	*
	* Assembly of the LHS and RHS of a Helmholtz Equation.
	*/

	size_t expr_size = sizeof(double) * nodes * LAYERS;
	double *buffer1, *buffer2, *buffer3;
	if(e = cudaMalloc(&buffer1, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	if(e = cudaMalloc(&buffer2, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	if(e = cudaMalloc(&buffer3, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);
	/* 
	* Evaluate an expression over the mesh.
	*
	*/
	printf(" Evaluating expression... ");

	//CUDA
	startTimer(&StartingTime);
	wrap_expression_1_GPU<<<nodes, LAYERS>>>(buffer1, coords_3DGPU);
	if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);

	//Explicit sync for timing
	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);

	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

#ifdef CHECK_VS_CPU

	//CPU
	double *expr1 = (double*)malloc(expr_size);
	startTimer(&StartingTime);
    wrap_expression_1(0, cells,
		expr1, map_3D,
		coords_3D, map_3D,
		off_3D, off_3D, LAYERS);
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);
	//fprint(expr1, 150, 1);

	//Check results
	double *expr1check = (double*)malloc(expr_size);
	if(e = cudaMemcpy(expr1check, buffer1, expr_size, cudaMemcpyDeviceToHost)) printf("Cuda error %d on line %d\n", e, __LINE__);

	for (int i = 0; i < nodes * LAYERS; ++i)
	{
		if (std::abs((expr1[i] / expr1check[i]) - 1) > 0.000001)
		{
			printf("Expr1 differs\n");
		}
	}

	free(expr1check);

	/*
	* Zero an array
	*/
	double *expr2 = (double*)malloc(expr_size);
	printf(" Set array to zero... ");
	wrap_zero_1(0, nodes * LAYERS,
		expr2,
		LAYERS);
	elapsed = getTimer(StartingTime, Frequency);

#endif // CHECK_VS_CPU

	//CUDA
	if(e = cudaMemset(buffer2, 0, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);

	/*
	* Interpolation operation.
	*/
	printf(" Interpolate expression... ");

	//CUDA
	startTimer(&StartingTime);

	wrap_rhs_1_GPU<<<cells, LAYERS>>>(buffer2, coords_3DGPU, buffer1, map_3DGPU, LAYERS);
	if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);
	
	//Explicit sync for timing
	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);

	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

#ifdef CHECK_VS_CPU

	startTimer(&StartingTime);
	wrap_rhs_1(0, cells,
		expr2, map_3D,
		coords_3D, map_3D,
		expr1, map_3D,
		off_3D, off_3D, off_3D, LAYERS);
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

	//Check results
	double *expr2check = (double*)malloc(expr_size);
	if(e = cudaMemcpy(expr2check, buffer2, expr_size, cudaMemcpyDeviceToHost)) printf("Cuda error %d on line %d\n", e, __LINE__);

	for (int i = 0; i < nodes * LAYERS; ++i)
	{
		if (std::abs((expr2[i] / expr2check[i]) - 1) > 0.000001)
		{
			printf("Expr2 differs\n");
		}
	}

	free(expr2check);

#endif // CHECK_VS_CPU

	/*
	* Another expression kernel
	*/

	// this kernel basically says expr3 = expr2
#ifdef CHECK_VS_CPU
	double *expr3 = expr2;
//	double *expr3 = (double*)malloc(sizeof(double) * nodes * LAYERS);
//	printf(" Evaluating expression... ");
//	startTimer(&StartingTime);
//	wrap_expression_2(0, nodes * LAYERS,
//		expr2,
//		expr3,
//		LAYERS);
//	elapsed = getTimer(StartingTime, Frequency);
//	printf("%g s\n", elapsed/1e6);
#endif // CHECK_VS_CPU

	/*
	* RHS assembly loop
	*/
	printf(" Assembling right-hand side... ");
	startTimer(&StartingTime);

	// ? Not sure!
	if(e = cudaMemset(buffer1, 0, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);

	wrap_rhs_GPU<<<cells, LAYERS>>>(buffer1, coords_3DGPU, buffer2, map_3DGPU, LAYERS);
	if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);
	
	//Explicit sync for timing
	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);

	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

#ifdef CHECK_VS_CPU
	double *expr4 = (double*)malloc(sizeof(double) * nodes * LAYERS);
	startTimer(&StartingTime);
	// is this necessary?? In the original code when run on linux the malloc is zero
	// but not necessary. accumulate to indeterminate initial value?
	wrap_zero_1(0, nodes * LAYERS,
		expr4,
		LAYERS);
	wrap_rhs(0, cells,
		expr4, map_3D,
		coords_3D, map_3D,
		expr2, map_3D,
		expr3, map_3D,
		off_3D, off_3D, off_3D, off_3D, LAYERS);
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

	//Check results
	double *expr4check = (double*)malloc(expr_size);
	if(e = cudaMemcpy(expr4check, buffer1, expr_size, cudaMemcpyDeviceToHost)) printf("Cuda error %d on line %d\n", e, __LINE__);

	for (int i = 0; i < nodes * LAYERS; ++i)
	{
		if (std::abs((expr4[i] / expr4check[i]) - 1) > 0.000001)
		{
			printf("Expr4 differs\n");
		}
	}

	free(expr4check);
#endif // CHECK_VS_CPU

	/*
	* Matrix assembly loop
	*/

	printf(" Assembling left-hand side... ");

	startTimer(&StartingTime);

	if(e = cudaMemset(buffer3, 0, expr_size)) printf("Cuda error %d on line %d\n", e, __LINE__);

	wrap_lhs_GPU<<<cells, LAYERS>>>(buffer3, coords_3DGPU, map_3DGPU, LAYERS);
	if(e = cudaGetLastError()) printf("Cuda error %d on line %d\n", e, __LINE__);
	
	//Explicit sync for timing
	if(e = cudaDeviceSynchronize()) printf("Cuda error %d on line %d\n", e, __LINE__);

	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);


#ifdef CHECK_VS_CPU
	double *expr5 = (double*)malloc(sizeof(double) * nodes * LAYERS);
	wrap_zero_1(0, nodes * LAYERS,
		expr5,
		LAYERS);
	startTimer(&StartingTime);
	wrap_lhs(0, cells,
		expr5, map_3D, map_3D,
		coords_3D, map_3D,
		off_3D, off_3D, off_3D, LAYERS);
	elapsed = getTimer(StartingTime, Frequency);
	printf("%g s\n", elapsed/1e6);

	//Check results
	double *expr5check = (double*)malloc(expr_size);
	if(e = cudaMemcpy(expr5check, buffer3, expr_size, cudaMemcpyDeviceToHost)) printf("Cuda error %d on line %d\n", e, __LINE__);

	for (int i = 0; i < nodes * LAYERS; ++i)
	{
		if (std::abs((expr5[i] / expr5check[i]) - 1) > 0.000001)
		{
			printf("Expr5 differs\n");
		}
	}

	free(expr5check);
#endif // CHECK_VS_CPU

	/*
	* RHS and LHS output
	*/
	double *lhs = (double*)malloc(sizeof(double) * nodes * LAYERS);
	if(e = cudaMemcpy(lhs, buffer3, expr_size, cudaMemcpyDeviceToHost)) 
		printf("Cuda error %d on line %d\n", e, __LINE__);
	double *rhs = (double*)malloc(sizeof(double) * nodes * LAYERS);
	if(e = cudaMemcpy(rhs, buffer2, expr_size, cudaMemcpyDeviceToHost)) 
		printf("Cuda error %d on line %d\n", e, __LINE__);
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
#ifdef CHECK_VS_CPU
	free(coords_3D);
	free(map_3D);
	free(expr1);
	free(expr2);
	//free(expr3);
	free(expr4);
	free(expr5);
#endif // CHECK_VS_CPU

	return 0;
}
