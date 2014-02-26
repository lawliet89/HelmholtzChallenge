/* 
 * Utility functions
 *
 * Paul Kelly, Gheorghe-Teodor Bercea, Fabio Luporini - Imperial College London - 2014
 */

extern void iprint(int * v, int n, int dim);

extern void fprint(double * v, int n, int dim);

extern char * str_cat(char *path, char *ext);

extern void check_range(double *v, int n, int dim);

extern void check_zero(double *v, int n, int dim);

extern void output(char * filename, double* v, int n, int dim);

/*
 * Mesh handling functions.
 *
 */
extern double * read_coords_2D(char * filename, int * nodes);

extern int * read_cell_node_map_2D(char * filename, int * cells, int * cell_size);

extern double * extrude_coords(double * coords, int nodes, int layers, double layer_height);

extern int * extrude_map(int * map, int cells, int cell_size, int layers);

extern double * create_constant_field(int size, int dim, double val);

extern int * create_cell_map(int cells, int dim, int layers);
