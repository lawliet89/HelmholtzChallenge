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

// #include "gettimemicroseconds.h"
#include "wrappers_kernels.h"
#include "utils.h"

#define LAYERS 101
#define LAYER_HEIGHT 0.1
#define FILE_LHS "lhs_out"
#define FILE_RHS "rhs_out"

#include <cuda_runtime.h>

//Cuda prototypes (cause I'm too lazy to make a header for now)
int testcuda();

int main (int argc, char *argv[]) 
{ 
	long s1 = 0, s2 = 0;

	int nodes, cells, cell_size;

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
	*/
	double * coords_3D = extrude_coords(coords_2D, nodes, LAYERS, LAYER_HEIGHT);
	free(coords_2D);
	/* 
	* 3D map from cells to vertices.
	* Stores triangle pairs as sextets of indicies into coords_3D as [x1A x1B y1A y1B z1A z1B  x2A x2B...] = [hexprism1, hexprsm2, etc] (all on base layer)
	* Incrementing ALL indicies by 1 triplet effectively steps a layer forward as coords_3D is layer major.
	* see "arg1_0_vec[0] += _arg1_0_off0_0[0] * 3;" etc in wrappers_kernels
	*/
	int * map_3D = extrude_map(map_2D, cells, cell_size, LAYERS);
	int off_3D[6] = {1, 1, 1, 1, 1, 1};
	free(map_2D);


	// Send coords to GPU
	size_t coord_3D_size = sizeof(double) * nodes * 3 * LAYERS;
	double* coords_3DGPU;
	cudaMalloc(&coords_3DGPU, coord_3D_size);
	cudaFree(coords_3DGPU);

	/*
	* Helmholtz Assembly
	*
	* Assembly of the LHS and RHS of a Helmholtz Equation.
	*/

	/* 
	* Evaluate an expression over the mesh.
	*
	*/
	double *expr1 = (double*)malloc(sizeof(double) * nodes * LAYERS);
	printf(" Evaluating expression... ");
	// s1 = stamp();
	wrap_expression_1(0, cells,
		expr1, map_3D,
		coords_3D, map_3D,
		off_3D, off_3D, LAYERS);
	// s2 = stamp();
	printf("%g s\n", (s2 - s1)/1e9);
	//fprint(expr1, 150, 1);

	/*
	* Zero an array
	*/
	double *expr2 = (double*)malloc(sizeof(double) * nodes * LAYERS);
	printf(" Set array to zero... ");
	// s1 = stamp();
	wrap_zero_1(0, nodes * LAYERS,
		expr2,
		LAYERS);
	// s2 = stamp();
	printf("%g s\n", (s2 - s1)/1e9);

	/*
	* Interpolation operation.
	*/
	printf(" Interpolate expression... ");
	// s1 = stamp();
	wrap_rhs_1(0, cells,
		expr2, map_3D,
		coords_3D, map_3D,
		expr1, map_3D,
		off_3D, off_3D, off_3D, LAYERS);
	// s2 = stamp();
	printf("%g s\n", (s2 - s1)/1e9);

	/*
	* Another expression kernel
	*/
	double *expr3 = (double*)malloc(sizeof(double) * nodes * LAYERS);
	printf(" Evaluating expression... ");
	// s1 = stamp();
	wrap_expression_2(0, nodes * LAYERS,
		expr2,
		expr3,
		LAYERS);
	// s2 = stamp();
	printf("%g s\n", (s2 - s1)/1e9);

	/*
	* RHS assembly loop
	*/
	double *expr4 = (double*)malloc(sizeof(double) * nodes * LAYERS);
	printf(" Assembling right-hand side... ");
	// s1 = stamp();
	wrap_rhs(0, cells,
		expr4, map_3D,
		coords_3D, map_3D,
		expr2, map_3D,
		expr3, map_3D,
		off_3D, off_3D, off_3D, off_3D, LAYERS);
	// s2 = stamp();
	printf("%g s\n", (s2 - s1)/1e9);


	/*
	* Matrix assembly loop
	*/
	double *expr5 = (double*)malloc(sizeof(double) * nodes * LAYERS);
	printf(" Assembling left-hand side... ");
	// s1 = stamp();
	wrap_lhs(0, cells,
		expr5, map_3D, map_3D,
		coords_3D, map_3D,
		off_3D, off_3D, off_3D, LAYERS);
	// s2 = stamp();
	printf("%g s\n", (s2 - s1)/1e9);

	/*
	* RHS and LHS output
	*/
	output(FILE_RHS, expr4, nodes * LAYERS, 1);
	output(FILE_LHS, expr5, nodes * LAYERS, 1);
	printf(" Numerical results written to output files.\n");

	free(coords_3D);
	free(map_3D);
	free(expr1);
	free(expr2);
	free(expr3);
	free(expr4);
	free(expr5);

	return 0;
}
