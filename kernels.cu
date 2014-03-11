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

	cout << (bValid ? EXIT_SUCCESS : EXIT_FAILURE);

	return bValid;
}




//This used to be the shittiest "for each node" loop ever. Used to be for each cell, overcompute the nodes. Now rewritten to be for each node
__global__ void wrap_expression_1_GPU(double* outarr, double* coordarr) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	const double pi = 3.141592653589793;
	outarr[i] = (1+12*pi*pi)*cos(coordarr[i*3 + 0]*pi*2)*cos(coordarr[i*3 + 1]*pi*2)*cos(coordarr[i*3 + 2]*pi*2);
}