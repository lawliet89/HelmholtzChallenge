/*
* Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/* Template project which demonstrates the basics on how to setup a project
* example application, doesn't use cutil library.
*/

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;

#ifdef _WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif

#define ASSERT(x, msg, retcode) \
	if (!(x)) \
{ \
	cout << msg << " " << __FILE__ << ":" << __LINE__ << endl; \
	return retcode; \
}

__global__ void sequence_gpu(int *d_ptr, int length)
{
	int elemID = blockIdx.x * blockDim.x + threadIdx.x;

	if (elemID < length)
	{
		d_ptr[elemID] = elemID;
	}
}


void sequence_cpu(int *h_ptr, int length)
{
	for (int elemID=0; elemID<length; elemID++)
	{
		h_ptr[elemID] = elemID;
	}
}

int testcuda()
{
	cout << "CUDA Runtime API template" << endl;
	cout << "=========================" << endl;
	cout << "Self-test started" << endl;

	const int N = 100;

	int *d_ptr;
	ASSERT(cudaSuccess == cudaMalloc(&d_ptr, N * sizeof(int)), "Device allocation of " << N << " ints failed", -1);

	int *h_ptr;
	ASSERT(cudaSuccess == cudaMallocHost(&h_ptr, N * sizeof(int)), "Host allocation of "   << N << " ints failed", -1);

	cout << "Memory allocated successfully" << endl;

	dim3 cudaBlockSize(32,1,1);
	dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
	sequence_gpu<<<cudaGridSize, cudaBlockSize>>>(d_ptr, N);
	ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
	ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

	sequence_cpu(h_ptr, N);

	cout << "CUDA and CPU algorithm implementations finished" << endl;

	int *h_d_ptr;
	ASSERT(cudaSuccess == cudaMallocHost(&h_d_ptr, N *sizeof(int)), "Host allocation of " << N << " ints failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(h_d_ptr, d_ptr, N *sizeof(int), cudaMemcpyDeviceToHost), "Copy of " << N << " ints from device to host failed", -1);
	bool bValid = true;

	for (int i=0; i<N && bValid; i++)
	{
		if (h_ptr[i] != h_d_ptr[i])
		{
			bValid = false;
		}
	}

	ASSERT(cudaSuccess == cudaFree(d_ptr),       "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFreeHost(h_ptr),   "Host deallocation failed",   -1);
	ASSERT(cudaSuccess == cudaFreeHost(h_d_ptr), "Host deallocation failed",   -1);

	cout << "Memory deallocated successfully" << endl;
	cout << "TEST Results " << endl;

	cout << (bValid ? EXIT_SUCCESS : EXIT_FAILURE) << endl;

	return bValid;
}


__constant__ double W8[8] = {0.0795103454359941, 0.0795103454359941, 0.0454896545640056, 0.0454896545640056, 0.0795103454359941, 0.0795103454359941, 0.0454896545640056, 0.0454896545640056};
__constant__ double FE0_D100[8][6] = {{-0.788675134594813, -0.211324865405187, 0.788675134594813, 0.211324865405187, 0.0, 0.0},
{-0.211324865405187, -0.788675134594813, 0.211324865405187, 0.788675134594813, 0.0, 0.0},
{-0.788675134594813, -0.211324865405187, 0.788675134594813, 0.211324865405187, 0.0, 0.0},
{-0.211324865405187, -0.788675134594813, 0.211324865405187, 0.788675134594813, 0.0, 0.0},
{-0.788675134594813, -0.211324865405187, 0.788675134594813, 0.211324865405187, 0.0, 0.0},
{-0.211324865405187, -0.788675134594813, 0.211324865405187, 0.788675134594813, 0.0, 0.0},
{-0.788675134594813, -0.211324865405187, 0.788675134594813, 0.211324865405187, 0.0, 0.0},
{-0.211324865405187, -0.788675134594813, 0.211324865405187, 0.788675134594813, 0.0, 0.0}};
__constant__ double FE0_D001[8][6] = {{-0.666390246014701, 0.666390246014701, -0.178558728263616, 0.178558728263616, -0.155051025721682, 0.155051025721682},
{-0.666390246014701, 0.666390246014701, -0.178558728263616, 0.178558728263616, -0.155051025721682, 0.155051025721682},
{-0.280019915499074, 0.280019915499074, -0.0750311102226081, 0.0750311102226081, -0.644948974278318, 0.644948974278318},
{-0.280019915499074, 0.280019915499074, -0.0750311102226081, 0.0750311102226081, -0.644948974278318, 0.644948974278318},
{-0.178558728263616, 0.178558728263616, -0.666390246014701, 0.666390246014701, -0.155051025721682, 0.155051025721682},
{-0.178558728263616, 0.178558728263616, -0.666390246014701, 0.666390246014701, -0.155051025721682, 0.155051025721682},
{-0.0750311102226081, 0.0750311102226081, -0.280019915499074, 0.280019915499074, -0.644948974278318, 0.644948974278318},
{-0.0750311102226081, 0.0750311102226081, -0.280019915499074, 0.280019915499074, -0.644948974278318, 0.644948974278318}};
__constant__ double FE0[8][6] = {{0.525565416968315, 0.140824829046386, 0.140824829046386, 0.0377338992172301, 0.122284888580111, 0.0327661371415707},
{0.140824829046386, 0.525565416968315, 0.0377338992172301, 0.140824829046386, 0.0327661371415707, 0.122284888580111},
{0.22084474454546, 0.0591751709536137, 0.0591751709536137, 0.0158559392689944, 0.508655219095739, 0.136293755182579},
{0.0591751709536137, 0.22084474454546, 0.0158559392689944, 0.0591751709536137, 0.136293755182579, 0.508655219095739},
{0.140824829046386, 0.0377338992172301, 0.525565416968315, 0.140824829046386, 0.122284888580111, 0.0327661371415707},
{0.0377338992172301, 0.140824829046386, 0.140824829046386, 0.525565416968315, 0.0327661371415707, 0.122284888580111},
{0.0591751709536137, 0.0158559392689944, 0.22084474454546, 0.0591751709536137, 0.508655219095739, 0.136293755182579},
{0.0158559392689944, 0.0591751709536137, 0.0591751709536137, 0.22084474454546, 0.136293755182579, 0.508655219095739}};
__constant__ double FE0_D010[8][6] = {{-0.788675134594813, -0.211324865405187, 0.0, 0.0, 0.788675134594813, 0.211324865405187},
{-0.211324865405187, -0.788675134594813, 0.0, 0.0, 0.211324865405187, 0.788675134594813},
{-0.788675134594813, -0.211324865405187, 0.0, 0.0, 0.788675134594813, 0.211324865405187},
{-0.211324865405187, -0.788675134594813, 0.0, 0.0, 0.211324865405187, 0.788675134594813},
{-0.788675134594813, -0.211324865405187, 0.0, 0.0, 0.788675134594813, 0.211324865405187},
{-0.211324865405187, -0.788675134594813, 0.0, 0.0, 0.211324865405187, 0.788675134594813},
{-0.788675134594813, -0.211324865405187, 0.0, 0.0, 0.788675134594813, 0.211324865405187},
{-0.211324865405187, -0.788675134594813, 0.0, 0.0, 0.211324865405187, 0.788675134594813}};


//This used to be the shittiest "for each node" loop ever. Used to be for each cell, overcompute the nodes. Now rewritten to be for each node
__global__ void wrap_expression_1_GPU(double* __restrict__ outarr, const double* __restrict__ coordarr) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	const double pi = 3.141592653589793;
	outarr[i] = (1+12*pi*pi)*cos(coordarr[i*3 + 0]*pi*2)*cos(coordarr[i*3 + 1]*pi*2)*cos(coordarr[i*3 + 2]*pi*2);
}

__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
			__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void wrap_rhs_1_GPU(double* __restrict__ outarr, 
							   double* __restrict__ coordarr,
							   double* __restrict__ inarr,
							   int* __restrict__ sextet_map, int layers)
{

	int cellID = blockIdx.x * blockDim.x + threadIdx.x;
	int cellID_2D = cellID/layers;
	int layerheight = cellID - cellID_2D*layers;

	if (layerheight == layers-1)
		return;

	int curr_verts[6];
	for (int i = 0; i < 6; ++i)
	{
		curr_verts[i] = sextet_map[cellID_2D * 6 + i] + layerheight;
	}

	double vertex_coordinates[18];
	for (int d = 0, int i = 0; d < 3; ++d)
	{
		for (int v = 0; v < 6; ++v)
		{
			vertex_coordinates[i++] = coordarr[curr_verts[v] * 3 + d];
		}
	}

	double in_vec[6];
	for (int i = 0; i < 6; ++i)
	{
		in_vec[i] = inarr[curr_verts[i]];
	}

	double J[9];
	J[0] = vertex_coordinates[2] - vertex_coordinates[0]; J[1] = vertex_coordinates[4] - vertex_coordinates[0]; J[2] = vertex_coordinates[1] - vertex_coordinates[0]; J[3] = vertex_coordinates[8] - vertex_coordinates[6]; J[4] = vertex_coordinates[10] - vertex_coordinates[6]; J[5] = vertex_coordinates[7] - vertex_coordinates[6]; J[6] = vertex_coordinates[14] - vertex_coordinates[12]; J[7] = vertex_coordinates[16] - vertex_coordinates[12]; J[8] = vertex_coordinates[13] - vertex_coordinates[12];
	double K[9];
	double detJ;
	{ const double d_00 = J[4]*J[8] - J[5]*J[7]; const double d_01 = J[5]*J[6] - J[3]*J[8]; const double d_02 = J[3]*J[7] - J[4]*J[6]; const double d_10 = J[2]*J[7] - J[1]*J[8]; const double d_11 = J[0]*J[8] - J[2]*J[6]; const double d_12 = J[1]*J[6] - J[0]*J[7]; const double d_20 = J[1]*J[5] - J[2]*J[4]; const double d_21 = J[2]*J[3] - J[0]*J[5]; const double d_22 = J[0]*J[4] - J[1]*J[3]; detJ = J[0]*d_00 + J[3]*d_10 + J[6]*d_20; K[0] = d_00 / detJ; K[1] = d_10 / detJ; K[2] = d_20 / detJ; K[3] = d_01 / detJ; K[4] = d_11 / detJ; K[5] = d_21 / detJ; K[6] = d_02 / detJ; K[7] = d_12 / detJ; K[8] = d_22 / detJ; }
	const double det = fabs(detJ);

	double A[6] = {0};
	for (int ip = 0; ip<8; ip++)
	{
		double F0 = 0.0;
		for (int r = 0; r<6; r++)
		{
			F0 += (in_vec[r]*FE0[ip][r]);
		}
		for (int j = 0; j<6; j++)
		{
			A[j] += (det*W8[ip]*FE0[ip][j]*F0);
		}
	}

	for (int i = 0; i < 6; ++i)
	{
		atomicAdd(outarr + curr_verts[i], A[i]);
		//*(outarr + curr_verts[i]) += A[i];
	}
}

__global__ void wrap_rhs_GPU(double* __restrict__ outarr, 
							 double* __restrict__ coordarr,
							 double* __restrict__ inarr,
							 int* __restrict__ sextet_map, int layers)
{

	int cellID = blockIdx.x * blockDim.x + threadIdx.x;
	int cellID_2D = cellID/layers;
	int layerheight = cellID - cellID_2D*layers;

	if (layerheight == layers-1)
		return;

	int curr_verts[6];
	for (int i = 0; i < 6; ++i)
	{
		curr_verts[i] = sextet_map[cellID_2D * 6 + i] + layerheight;
	}

	double vertex_coordinates[18];
	for (int d = 0, int i = 0; d < 3; ++d)
	{
		for (int v = 0; v < 6; ++v)
		{
			vertex_coordinates[i++] = coordarr[curr_verts[v] * 3 + d];
		}
	}

	double in_vec[6];
	for (int i = 0; i < 6; ++i)
	{
		in_vec[i] = inarr[curr_verts[i]];
	}

	double J[9];
	J[0] = vertex_coordinates[2] - vertex_coordinates[0]; 
	J[1] = vertex_coordinates[4] - vertex_coordinates[0]; 
	J[2] = vertex_coordinates[1] - vertex_coordinates[0]; 
	J[3] = vertex_coordinates[8] - vertex_coordinates[6]; 
	J[4] = vertex_coordinates[10] - vertex_coordinates[6];
	J[5] = vertex_coordinates[7] - vertex_coordinates[6]; 
	J[6] = vertex_coordinates[14] - vertex_coordinates[12];
	J[7] = vertex_coordinates[16] - vertex_coordinates[12]; 
	J[8] = vertex_coordinates[13] - vertex_coordinates[12];;

	double K[9];
	double detJ;
	do { 
		const double d_00 = J[4]*J[8] - J[5]*J[7];
		const double d_01 = J[5]*J[6] - J[3]*J[8];
		const double d_02 = J[3]*J[7] - J[4]*J[6];
		const double d_10 = J[2]*J[7] - J[1]*J[8];
		const double d_11 = J[0]*J[8] - J[2]*J[6]; 
		const double d_12 = J[1]*J[6] - J[0]*J[7];
		const double d_20 = J[1]*J[5] - J[2]*J[4];
		const double d_21 = J[2]*J[3] - J[0]*J[5]; 
		const double d_22 = J[0]*J[4] - J[1]*J[3];

		detJ = J[0]*d_00 + J[3]*d_10 + J[6]*d_20; 
		K[0] = d_00 / detJ; 
		K[1] = d_10 / detJ; 
		K[2] = d_20 / detJ; 
		K[3] = d_01 / detJ; 
		K[4] = d_11 / detJ; 
		K[5] = d_21 / detJ; 
		K[6] = d_02 / detJ; 
		K[7] = d_12 / detJ; 
		K[8] = d_22 / detJ; } while (0);

		const double det = fabs(detJ);

		double A[6] = {0};
		for (int ip = 0; ip<8; ip++)
		{
			double F0 = 0.0;
			double F1 = 0.0;
			double F2 = 0.0;
			double F3 = 0.0;
			double F4 = 0.0;
			for (int r = 0; r<6; r++)
			{
				F0 += (in_vec[r]*FE0[ip][r]);
				F1 += (in_vec[r]*FE0_D100[ip][r]);
				F2 += (in_vec[r]*FE0_D010[ip][r]);
				F3 += (in_vec[r]*FE0_D001[ip][r]);
				F4 += (in_vec[r]*FE0[ip][r]);
			}
			for (int j = 0; j<6; j++)
			{
				A[j] += (((FE0[ip][j]*F4)+(((K[2]*FE0_D100[ip][j])+(K[5]*FE0_D010[ip][j])+(K[8]*FE0_D001[ip][j]))*((K[8]*F3)+(K[5]*F2)+(K[2]*F1)))+(((K[1]*FE0_D100[ip][j])+(K[4]*FE0_D010[ip][j])+(K[7]*FE0_D001[ip][j]))*((K[7]*F3)+(K[4]*F2)+(K[1]*F1)))+(((K[0]*FE0_D100[ip][j])+(K[3]*FE0_D010[ip][j])+(K[6]*FE0_D001[ip][j]))*((K[6]*F3)+(K[3]*F2)+(K[0]*F1)))+(FE0[ip][j]*F0*-1.0))*det*W8[ip]);
			}
		}

		for (int i = 0; i < 6; ++i)
		{
			atomicAdd(outarr + curr_verts[i], A[i]);
			//*(outarr + curr_verts[i]) += A[i];
		}
}

// MATRIX ASSEMBLY KERNEL
// WARNING: Do not modify this function
__device__ void addto_vector(double *arg0_0,
							 double buffer_arg0_0[6][6],
							 int map_size_1, int *xtr_arg0_0_map0_0,
							 int map_size_2, int *xtr_arg0_0_map1_0,
							 int position)
{

	for(int i = 0; i < map_size_1; i++){
		for(int j = 0; j < map_size_2; j++){
			if (buffer_arg0_0[i][j] > buffer_arg0_0[j][i]){
				//arg0_0[xtr_arg0_0_map0_0[i]] += buffer_arg0_0[i][j];
				atomicAdd(arg0_0 + xtr_arg0_0_map0_0[i], buffer_arg0_0[i][j]);
			}else{
				//arg0_0[xtr_arg0_0_map1_0[i]] += buffer_arg0_0[i][j];
				atomicAdd(arg0_0 + xtr_arg0_0_map1_0[i], buffer_arg0_0[i][j]);
			}
		}
	}

	return;

}

__global__ void wrap_lhs_GPU(double* __restrict__ outarr, 
							 double* __restrict__ coordarr,
							 int* __restrict__ sextet_map, int layers)
{
	int cellID = blockIdx.x * blockDim.x + threadIdx.x;
	int cellID_2D = cellID/layers;
	int layerheight = cellID - cellID_2D*layers;

	if (layerheight == layers-1)
		return;

	int curr_verts[6];
	for (int i = 0; i < 6; ++i)
	{
		curr_verts[i] = sextet_map[cellID_2D * 6 + i] + layerheight;
	}

	double vertex_coordinates[18];
	for (int d = 0, int i = 0; d < 3; ++d)
	{
		for (int v = 0; v < 6; ++v)
		{
			vertex_coordinates[i++] = coordarr[curr_verts[v] * 3 + d];
		}
	}
	double J[9];
	J[0] = vertex_coordinates[2] - vertex_coordinates[0]; 
	J[1] = vertex_coordinates[4] - vertex_coordinates[0]; 
	J[2] = vertex_coordinates[1] - vertex_coordinates[0]; 
	J[3] = vertex_coordinates[8] - vertex_coordinates[6]; 
	J[4] = vertex_coordinates[10] - vertex_coordinates[6]; 
	J[5] = vertex_coordinates[7] - vertex_coordinates[6]; 
	J[6] = vertex_coordinates[14] - vertex_coordinates[12]; 
	J[7] = vertex_coordinates[16] - vertex_coordinates[12]; 
	J[8] = vertex_coordinates[13] - vertex_coordinates[12];
	double K[9];
	double detJ;
	do { const double d_00 = J[4]*J[8] - J[5]*J[7]; const double d_01 = J[5]*J[6] - J[3]*J[8]; const double d_02 = J[3]*J[7] - J[4]*J[6]; const double d_10 = J[2]*J[7] - J[1]*J[8]; const double d_11 = J[0]*J[8] - J[2]*J[6]; const double d_12 = J[1]*J[6] - J[0]*J[7]; const double d_20 = J[1]*J[5] - J[2]*J[4]; const double d_21 = J[2]*J[3] - J[0]*J[5]; const double d_22 = J[0]*J[4] - J[1]*J[3]; detJ = J[0]*d_00 + J[3]*d_10 + J[6]*d_20; K[0] = d_00 / detJ; K[1] = d_10 / detJ; K[2] = d_20 / detJ; K[3] = d_01 / detJ; K[4] = d_11 / detJ; K[5] = d_21 / detJ; K[6] = d_02 / detJ; K[7] = d_12 / detJ; K[8] = d_22 / detJ; } while (0);
	const double det = fabs(detJ);

	double accum[6][6] = {{0}};
	for (int ip = 0; ip<8; ip++)
	{
		for (int j = 0; j<6; j++)
		{
			for (int k = 0; k<6; k++)
			{
				accum[j][k] += (
					(
						(FE0[ip][k]*FE0[ip][j]) +
						(
							(
								(K[2]*FE0_D100[ip][k]) +
								(K[5]*FE0_D010[ip][k]) +
								(K[8]*FE0_D001[ip][k])
							)*(
								(K[2]*FE0_D100[ip][j]) +
								(K[5]*FE0_D010[ip][j]) +
								(K[8]*FE0_D001[ip][j])
							)
						) + (
							(
								(K[1]*FE0_D100[ip][k]) +
								(K[4]*FE0_D010[ip][k]) +
								(K[7]*FE0_D001[ip][k])
							)*(
								(K[1]*FE0_D100[ip][j]) +
								(K[4]*FE0_D010[ip][j]) +
								(K[7]*FE0_D001[ip][j])
							)
						) + (
							(
								(K[0]*FE0_D100[ip][k]) +
								(K[3]*FE0_D010[ip][k]) +
								(K[6]*FE0_D001[ip][k])
							)*(
								(K[0]*FE0_D100[ip][j]) +
								(K[3]*FE0_D010[ip][j]) +
								(K[6]*FE0_D001[ip][j])
							)
						)
					)*det*W8[ip]
				);
			}
		}
	}
	for (int j = 0; j<6; j++)
	{
		for (int k = 0; k<6; k++)
		{
			atomicAdd(outarr + curr_verts[j], accum[j][k]);
		}
	}
}