/* 
 * Utility functions
 *
 * Paul Kelly, Gheorghe-Teodor Bercea, Fabio Luporini - Imperial College London - 2014
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RANGE 1000.0

/*
 * Some utility functions.
 *
 */
void iprint(int * v, int n, int dim){
  for(int i = 0; i< n; i++){
    for(int j = 0; j< dim; j++)
      printf(" %d ", v[dim*i + j]);
    printf("\n");
  }
}

void fprint(double * v, int n, int dim){
  for(int i = 0; i< n; i++){
    for(int j = 0; j< dim; j++)
      printf(" %13.12f ", v[dim*i + j]);
    printf("\n");
  }
}

char * str_cat(char *path, char *ext){
  char *res = malloc(200);
  strcat(res, path);
  strcat(res, ext);
  return res;
}

void check_range(double *v, int n, int dim){
  for(int i = 0; i < n; i++){
    for(int j = 0; j< dim; j++){
      if (!(v[dim*i + j] <  RANGE && v[dim*i + j] > -RANGE)){
        printf("Error: The array contains out of range values: %f\n", v[dim*i + j]);
        exit(0);
      }
    }
  }
  printf("PASS\n");
}

void check_zero(double *v, int n, int dim){
  for(int i = 0; i < n; i++){
    for(int j = 0; j< dim; j++){
      if (!(abs(v[dim*i + j] - 0.0) < 1.0e-14)){
        printf("Error: The array contains non-zero values: %f\n", v[dim*i + j]);
        exit(0);
      }
    }
  }
  printf("PASS\n");
}

void output(char * filename, double* v, int n, int dim){
  FILE *f = fopen(filename, "wr");
  fprintf(f, "%d %d\n", n, dim);
  for(int i = 0; i < n; i++){
    fprintf(f,"%13.12f\n",v[i]);
  }
  fclose(f);
}

/*
 * Mesh handling functions.
 *
 */
double * read_coords_2D(char * filename, int * nodes){
  double * res;
  int len, size, a, i;
  FILE *f;
  f = fopen(filename, "r");
  if (fscanf(f, "%d %d %d %d\n", &len, &size, &a, &a) == -1)
    printf("Error reading file %s \n", filename);
  *nodes = len;
  res = (double *)malloc(sizeof(double) * len * size);
  for(i = 0; i < len * size; i+=2){
    if (fscanf(f, "%d %lf %lf\n", &a, &res[i], &res[i+1]) == -1)
      printf("Error reading file %s \n", filename);
  }
  fclose(f);
  return res;
}

int * read_cell_node_map_2D(char * filename, int * cells, int * cell_size){
  int * res;
  int len, size, a, i;
  FILE *f;
  f = fopen(filename, "r");
  if (fscanf(f, "%d %d %d\n", &len, &size, &a) == -1)
    printf("Error reading file %s \n", filename);
  *cells = len;
  *cell_size = size;
  res = (int *)malloc(sizeof(int) * len * size);
  for(i = 0; i < len*size; i+=size){
    if (fscanf(f, "%d %d %d %d\n", &a, &res[i], &res[i+1], &res[i+2]) == -1)
      printf("Error reading file %s \n", filename);
  }
  fclose(f);
  //Switch to C numbering
  for(i = 0; i < len*size; i++)
    res[i]--;
  return res;
}

double * extrude_coords(double * coords, int nodes, int layers, double layer_height){
  double * res = (double *)malloc(sizeof(double) * nodes * 3 * layers);
  for(int i = 0; i < nodes; i++){
    for(int j = 0; j < layers; j++){
      res[3*(i*layers + j)] = coords[2*i];
      res[3*(i*layers + j) + 1] = coords[2*i + 1];
      res[3*(i*layers + j) + 2] = layer_height * j;
    }
  }
  return res;
}

int * extrude_map(int * map, int cells, int cell_size, int layers){
  int * res = (int *)malloc(sizeof(int) * cells * cell_size * 2);

  for(int i = 0; i < cells; i++){
    for(int j = 0; j < cell_size; j++){
      res[2 * (i * cell_size + j)] = layers * map[i * cell_size + j];
      res[2 * (i * cell_size + j) + 1] = res[2 * (i * cell_size + j)] + 1;
    }
  }
  return res;
}

double * create_constant_field(int size, int dim, double val){
  double * res = (double *)malloc(sizeof(double) * size * dim);
  for(int i = 0; i < size * dim; i++){
    res[i] = val;
  }
  return res;
}

int * create_cell_map(int cells, int dim, int layers){
  // dim is number of dofs per 3D cell
  int * res = (int *)malloc(sizeof(int) * cells * dim);
  for(int i = 0; i < cells; i++){
    for(int j = 0; j < dim; j++){
      res[dim * i + j] = (layers - 1) * dim * i + j;
    }
  }
  return res;
}