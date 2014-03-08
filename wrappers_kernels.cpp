/*
 * Kernels and wrappers for the Helmholtz RHS and LHS assembly.
 *
 * Paul Kelly, Gheorghe-Teodor Bercea, Fabio Luporini - Imperial College London
 *- 2014
 */

#include <math.h>

#define PI 3.14159265

/*
 * WRAPPERS AND KERNELS
 */

/*
 * EXPRESSION KERNEL
   Probably parallerise over outermost loop - independent
  Second loop is also independent
  Third loop works via accumulator.
 */
void expression_kernel_1(double A[6], double const **x_) {
  const double X[6][6] = { { 1.0, 0.0, 2.77555756156e-17, 0.0, 0.0, 0.0 },
                           { 0.0, 1.0, 0.0, 2.77555756156e-17, 0.0, 0.0 },
                           { -2.77555756156e-17, -0, 1.0, 0.0, 0.0, 0.0 },
                           { -0, -2.77555756156e-17, 0.0, 1.0, 0.0, 0.0 },
                           { 0.0, 0.0, 5.55111512313e-17, 0.0, 1.0, 0.0 },
                           { 0.0, 0.0, 0.0, 5.55111512313e-17, 0.0, 1.0 } };
  double x[3];
  const double pi = 3.141592653589793;
  {
    for (int k = 0; k < 6; k++) {
      for (unsigned int d = 0; d < 3; d++) {
        x[d] = 0;
        for (unsigned int i = 0; i < 6; i++) {
          x[d] += X[k][i] * x_[i][d];
        };
      };
      A[k] = (1 + 12 * PI * PI) * cos(x[0] * PI * 2) * cos(x[1] * PI * 2) *
             cos(x[2] * PI * 2);
    }
  }
}
void wrap_expression_1(int start, int end, double *arg0_0, int *arg0_0_map0_0,
                       double *arg1_0, int *arg1_0_map0_0, int *_arg0_0_off0_0,
                       int *_arg1_0_off0_0, int layer) {
  double *arg1_0_vec[6];
  int xtr_arg0_0_map0_0[6];
  for (int n = start; n < end; n++) {
    int i = n;
    arg1_0_vec[0] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3;
    arg1_0_vec[1] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3;
    arg1_0_vec[2] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3;
    arg1_0_vec[3] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3;
    arg1_0_vec[4] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3;
    arg1_0_vec[5] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3;
    xtr_arg0_0_map0_0[0] = *(arg0_0_map0_0 + i * 6 + 0);
    xtr_arg0_0_map0_0[1] = *(arg0_0_map0_0 + i * 6 + 1);
    xtr_arg0_0_map0_0[2] = *(arg0_0_map0_0 + i * 6 + 2);
    xtr_arg0_0_map0_0[3] = *(arg0_0_map0_0 + i * 6 + 3);
    xtr_arg0_0_map0_0[4] = *(arg0_0_map0_0 + i * 6 + 4);
    xtr_arg0_0_map0_0[5] = *(arg0_0_map0_0 + i * 6 + 5);
    for (int j_0 = 0; j_0 < layer - 1; ++j_0) {
      double buffer_arg0_0[6] = { 0 };
      expression_kernel_1(buffer_arg0_0,
                          const_cast<double const **>(arg1_0_vec));
      for (int i_0 = 0; i_0 < 6; ++i_0) {
        *(arg0_0 + (xtr_arg0_0_map0_0[i_0]) * 1) = buffer_arg0_0[i_0 * 1 + 0];
      }
      xtr_arg0_0_map0_0[0] += _arg0_0_off0_0[0];
      xtr_arg0_0_map0_0[1] += _arg0_0_off0_0[1];
      xtr_arg0_0_map0_0[2] += _arg0_0_off0_0[2];
      xtr_arg0_0_map0_0[3] += _arg0_0_off0_0[3];
      xtr_arg0_0_map0_0[4] += _arg0_0_off0_0[4];
      xtr_arg0_0_map0_0[5] += _arg0_0_off0_0[5];
      arg1_0_vec[0] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[1] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[2] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[3] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[4] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[5] += _arg1_0_off0_0[5] * 3;
    }
  }
}

// ZERO KERNEL
void zero_1(double *dat) {
  for (int n = 0; n < 1; ++n) {
    dat[n] = (double)0;
  }
}
void wrap_zero_1(int start, int end, double *arg0_0, int layer) {
  for (int n = start; n < end; n++) {
    int i = n;
    { zero_1(arg0_0 + i * 1); }
  }
}

// RHS ASSEMBLY KERNEL
static void kernel_rhs_1(double A[6], double **vertex_coordinates,
                         double **w0) {
  double J[9];
  J[0] = vertex_coordinates[2][0] - vertex_coordinates[0][0];
  J[1] = vertex_coordinates[4][0] - vertex_coordinates[0][0];
  J[2] = vertex_coordinates[1][0] - vertex_coordinates[0][0];
  J[3] = vertex_coordinates[8][0] - vertex_coordinates[6][0];
  J[4] = vertex_coordinates[10][0] - vertex_coordinates[6][0];
  J[5] = vertex_coordinates[7][0] - vertex_coordinates[6][0];
  J[6] = vertex_coordinates[14][0] - vertex_coordinates[12][0];
  J[7] = vertex_coordinates[16][0] - vertex_coordinates[12][0];
  J[8] = vertex_coordinates[13][0] - vertex_coordinates[12][0];
  ;
  double K[9];
  double detJ;
  do {
    const double d_00 = J[4] * J[8] - J[5] * J[7];
    const double d_01 = J[5] * J[6] - J[3] * J[8];
    const double d_02 = J[3] * J[7] - J[4] * J[6];
    const double d_10 = J[2] * J[7] - J[1] * J[8];
    const double d_11 = J[0] * J[8] - J[2] * J[6];
    const double d_12 = J[1] * J[6] - J[0] * J[7];
    const double d_20 = J[1] * J[5] - J[2] * J[4];
    const double d_21 = J[2] * J[3] - J[0] * J[5];
    const double d_22 = J[0] * J[4] - J[1] * J[3];
    detJ = J[0] * d_00 + J[3] * d_10 + J[6] * d_20;
    K[0] = d_00 / detJ;
    K[1] = d_10 / detJ;
    K[2] = d_20 / detJ;
    K[3] = d_01 / detJ;
    K[4] = d_11 / detJ;
    K[5] = d_21 / detJ;
    K[6] = d_02 / detJ;
    K[7] = d_12 / detJ;
    K[8] = d_22 / detJ;
  } while (0);
  const double det = fabs(detJ);
  double W8[8] = { 0.0795103454359941, 0.0795103454359941, 0.0454896545640056,
                   0.0454896545640056, 0.0795103454359941, 0.0795103454359941,
                   0.0454896545640056, 0.0454896545640056 };
  double FE0[8][6] = {
    { 0.525565416968315,  0.140824829046386, 0.140824829046386,
      0.0377338992172301, 0.122284888580111, 0.0327661371415707 },
    { 0.140824829046386, 0.525565416968315,  0.0377338992172301,
      0.140824829046386, 0.0327661371415707, 0.122284888580111 },
    { 0.22084474454546,   0.0591751709536137, 0.0591751709536137,
      0.0158559392689944, 0.508655219095739,  0.136293755182579 },
    { 0.0591751709536137, 0.22084474454546,  0.0158559392689944,
      0.0591751709536137, 0.136293755182579, 0.508655219095739 },
    { 0.140824829046386, 0.0377338992172301, 0.525565416968315,
      0.140824829046386, 0.122284888580111,  0.0327661371415707 },
    { 0.0377338992172301, 0.140824829046386,  0.140824829046386,
      0.525565416968315,  0.0327661371415707, 0.122284888580111 },
    { 0.0591751709536137, 0.0158559392689944, 0.22084474454546,
      0.0591751709536137, 0.508655219095739,  0.136293755182579 },
    { 0.0158559392689944, 0.0591751709536137, 0.0591751709536137,
      0.22084474454546,   0.136293755182579,  0.508655219095739 }
  };
  for (int ip = 0; ip < 8; ip++) {
    double F0 = 0.0;
    for (int r = 0; r < 6; r++) {
      F0 += (w0[r][0] * FE0[ip][r]);
    }
    for (int j = 0; j < 6; j++) {
      A[j] += (det * W8[ip] * FE0[ip][j] * F0);
    }
  }
}
void wrap_rhs_1(int start, int end, double *arg0_0, int *arg0_0_map0_0,
                double *arg1_0, int *arg1_0_map0_0, double *arg2_0,
                int *arg2_0_map0_0, int *_arg0_0_off0_0, int *_arg1_0_off0_0,
                int *_arg2_0_off0_0, int layer) {
  double *arg1_0_vec[18];
  double *arg2_0_vec[6];
  int xtr_arg0_0_map0_0[6];
  for (int n = start; n < end; n++) {
    int i = n;
    arg1_0_vec[0] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3;
    arg1_0_vec[1] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3;
    arg1_0_vec[2] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3;
    arg1_0_vec[3] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3;
    arg1_0_vec[4] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3;
    arg1_0_vec[5] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3;
    arg1_0_vec[6] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3 + 1;
    arg1_0_vec[7] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3 + 1;
    arg1_0_vec[8] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3 + 1;
    arg1_0_vec[9] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3 + 1;
    arg1_0_vec[10] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3 + 1;
    arg1_0_vec[11] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3 + 1;
    arg1_0_vec[12] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3 + 2;
    arg1_0_vec[13] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3 + 2;
    arg1_0_vec[14] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3 + 2;
    arg1_0_vec[15] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3 + 2;
    arg1_0_vec[16] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3 + 2;
    arg1_0_vec[17] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3 + 2;
    arg2_0_vec[0] = arg2_0 + (arg2_0_map0_0[i * 6 + 0]) * 1;
    arg2_0_vec[1] = arg2_0 + (arg2_0_map0_0[i * 6 + 1]) * 1;
    arg2_0_vec[2] = arg2_0 + (arg2_0_map0_0[i * 6 + 2]) * 1;
    arg2_0_vec[3] = arg2_0 + (arg2_0_map0_0[i * 6 + 3]) * 1;
    arg2_0_vec[4] = arg2_0 + (arg2_0_map0_0[i * 6 + 4]) * 1;
    arg2_0_vec[5] = arg2_0 + (arg2_0_map0_0[i * 6 + 5]) * 1;
    xtr_arg0_0_map0_0[0] = *(arg0_0_map0_0 + i * 6 + 0);
    xtr_arg0_0_map0_0[1] = *(arg0_0_map0_0 + i * 6 + 1);
    xtr_arg0_0_map0_0[2] = *(arg0_0_map0_0 + i * 6 + 2);
    xtr_arg0_0_map0_0[3] = *(arg0_0_map0_0 + i * 6 + 3);
    xtr_arg0_0_map0_0[4] = *(arg0_0_map0_0 + i * 6 + 4);
    xtr_arg0_0_map0_0[5] = *(arg0_0_map0_0 + i * 6 + 5);
    for (int j_0 = 0; j_0 < layer - 1; ++j_0) {
      double buffer_arg0_0[6] = { 0 };
      kernel_rhs_1(buffer_arg0_0, arg1_0_vec, arg2_0_vec);
      for (int i_0 = 0; i_0 < 6; ++i_0) {
        *(arg0_0 + (xtr_arg0_0_map0_0[i_0]) * 1) += buffer_arg0_0[i_0 * 1 + 0];
      }
      xtr_arg0_0_map0_0[0] += _arg0_0_off0_0[0];
      xtr_arg0_0_map0_0[1] += _arg0_0_off0_0[1];
      xtr_arg0_0_map0_0[2] += _arg0_0_off0_0[2];
      xtr_arg0_0_map0_0[3] += _arg0_0_off0_0[3];
      xtr_arg0_0_map0_0[4] += _arg0_0_off0_0[4];
      xtr_arg0_0_map0_0[5] += _arg0_0_off0_0[5];
      arg1_0_vec[0] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[1] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[2] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[3] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[4] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[5] += _arg1_0_off0_0[5] * 3;
      arg1_0_vec[6] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[7] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[8] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[9] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[10] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[11] += _arg1_0_off0_0[5] * 3;
      arg1_0_vec[12] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[13] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[14] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[15] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[16] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[17] += _arg1_0_off0_0[5] * 3;
      arg2_0_vec[0] += _arg2_0_off0_0[0] * 1;
      arg2_0_vec[1] += _arg2_0_off0_0[1] * 1;
      arg2_0_vec[2] += _arg2_0_off0_0[2] * 1;
      arg2_0_vec[3] += _arg2_0_off0_0[3] * 1;
      arg2_0_vec[4] += _arg2_0_off0_0[4] * 1;
      arg2_0_vec[5] += _arg2_0_off0_0[5] * 1;
    }
  }
}

// ANOTHER EXPRESSION COMPUTATION
void expression_2(double *fn_1, double *fn_0) { fn_0[0] = fn_1[0]; }
void wrap_expression_2(int start, int end, double *arg0_0, double *arg1_0,
                       int layers) {
  for (int n = start; n < end; n++) {
    int i = n;
    expression_2(arg0_0 + i * 1, arg1_0 + i * 1);
    // for (int j = 0; j < layers - 1; j++){
    // expression_2(arg0_0 + layers * i * 1 + j, arg1_0 + layers * i * 1 + j);
    //}
  }
}

// RHS ASSEMBLY
void kernel_rhs(double A[6], double **vertex_coordinates, double **w0,
                double **w1) {
  double J[9];
  J[0] = vertex_coordinates[2][0] - vertex_coordinates[0][0];
  J[1] = vertex_coordinates[4][0] - vertex_coordinates[0][0];
  J[2] = vertex_coordinates[1][0] - vertex_coordinates[0][0];
  J[3] = vertex_coordinates[8][0] - vertex_coordinates[6][0];
  J[4] = vertex_coordinates[10][0] - vertex_coordinates[6][0];
  J[5] = vertex_coordinates[7][0] - vertex_coordinates[6][0];
  J[6] = vertex_coordinates[14][0] - vertex_coordinates[12][0];
  J[7] = vertex_coordinates[16][0] - vertex_coordinates[12][0];
  J[8] = vertex_coordinates[13][0] - vertex_coordinates[12][0];
  ;

  double K[9];
  double detJ;
  do {
    const double d_00 = J[4] * J[8] - J[5] * J[7];
    const double d_01 = J[5] * J[6] - J[3] * J[8];
    const double d_02 = J[3] * J[7] - J[4] * J[6];
    const double d_10 = J[2] * J[7] - J[1] * J[8];
    const double d_11 = J[0] * J[8] - J[2] * J[6];
    const double d_12 = J[1] * J[6] - J[0] * J[7];
    const double d_20 = J[1] * J[5] - J[2] * J[4];
    const double d_21 = J[2] * J[3] - J[0] * J[5];
    const double d_22 = J[0] * J[4] - J[1] * J[3];

    detJ = J[0] * d_00 + J[3] * d_10 + J[6] * d_20;
    K[0] = d_00 / detJ;
    K[1] = d_10 / detJ;
    K[2] = d_20 / detJ;
    K[3] = d_01 / detJ;
    K[4] = d_11 / detJ;
    K[5] = d_21 / detJ;
    K[6] = d_02 / detJ;
    K[7] = d_12 / detJ;
    K[8] = d_22 / detJ;
  } while (0);

  const double det = fabs(detJ);
  double W8[8] = { 0.0795103454359941, 0.0795103454359941, 0.0454896545640056,
                   0.0454896545640056, 0.0795103454359941, 0.0795103454359941,
                   0.0454896545640056, 0.0454896545640056 };
  double FE0_D100[8][6] = {
    { -0.788675134594813, -0.211324865405187, 0.788675134594813,
      0.211324865405187,  0.0,                0.0 },
    { -0.211324865405187, -0.788675134594813, 0.211324865405187,
      0.788675134594813,  0.0,                0.0 },
    { -0.788675134594813, -0.211324865405187, 0.788675134594813,
      0.211324865405187,  0.0,                0.0 },
    { -0.211324865405187, -0.788675134594813, 0.211324865405187,
      0.788675134594813,  0.0,                0.0 },
    { -0.788675134594813, -0.211324865405187, 0.788675134594813,
      0.211324865405187,  0.0,                0.0 },
    { -0.211324865405187, -0.788675134594813, 0.211324865405187,
      0.788675134594813,  0.0,                0.0 },
    { -0.788675134594813, -0.211324865405187, 0.788675134594813,
      0.211324865405187,  0.0,                0.0 },
    { -0.211324865405187, -0.788675134594813, 0.211324865405187,
      0.788675134594813,  0.0,                0.0 }
  };
  double FE0_D001[8][6] = {
    { -0.666390246014701, 0.666390246014701,  -0.178558728263616,
      0.178558728263616,  -0.155051025721682, 0.155051025721682 },
    { -0.666390246014701, 0.666390246014701,  -0.178558728263616,
      0.178558728263616,  -0.155051025721682, 0.155051025721682 },
    { -0.280019915499074, 0.280019915499074,  -0.0750311102226081,
      0.0750311102226081, -0.644948974278318, 0.644948974278318 },
    { -0.280019915499074, 0.280019915499074,  -0.0750311102226081,
      0.0750311102226081, -0.644948974278318, 0.644948974278318 },
    { -0.178558728263616, 0.178558728263616,  -0.666390246014701,
      0.666390246014701,  -0.155051025721682, 0.155051025721682 },
    { -0.178558728263616, 0.178558728263616,  -0.666390246014701,
      0.666390246014701,  -0.155051025721682, 0.155051025721682 },
    { -0.0750311102226081, 0.0750311102226081, -0.280019915499074,
      0.280019915499074,   -0.644948974278318, 0.644948974278318 },
    { -0.0750311102226081, 0.0750311102226081, -0.280019915499074,
      0.280019915499074,   -0.644948974278318, 0.644948974278318 }
  };
  double FE0[8][6] = {
    { 0.525565416968315,  0.140824829046386, 0.140824829046386,
      0.0377338992172301, 0.122284888580111, 0.0327661371415707 },
    { 0.140824829046386, 0.525565416968315,  0.0377338992172301,
      0.140824829046386, 0.0327661371415707, 0.122284888580111 },
    { 0.22084474454546,   0.0591751709536137, 0.0591751709536137,
      0.0158559392689944, 0.508655219095739,  0.136293755182579 },
    { 0.0591751709536137, 0.22084474454546,  0.0158559392689944,
      0.0591751709536137, 0.136293755182579, 0.508655219095739 },
    { 0.140824829046386, 0.0377338992172301, 0.525565416968315,
      0.140824829046386, 0.122284888580111,  0.0327661371415707 },
    { 0.0377338992172301, 0.140824829046386,  0.140824829046386,
      0.525565416968315,  0.0327661371415707, 0.122284888580111 },
    { 0.0591751709536137, 0.0158559392689944, 0.22084474454546,
      0.0591751709536137, 0.508655219095739,  0.136293755182579 },
    { 0.0158559392689944, 0.0591751709536137, 0.0591751709536137,
      0.22084474454546,   0.136293755182579,  0.508655219095739 }
  };
  double FE0_D010[8][6] = { { -0.788675134594813, -0.211324865405187, 0.0, 0.0,
                              0.788675134594813,  0.211324865405187 },
                            { -0.211324865405187, -0.788675134594813, 0.0, 0.0,
                              0.211324865405187,  0.788675134594813 },
                            { -0.788675134594813, -0.211324865405187, 0.0, 0.0,
                              0.788675134594813,  0.211324865405187 },
                            { -0.211324865405187, -0.788675134594813, 0.0, 0.0,
                              0.211324865405187,  0.788675134594813 },
                            { -0.788675134594813, -0.211324865405187, 0.0, 0.0,
                              0.788675134594813,  0.211324865405187 },
                            { -0.211324865405187, -0.788675134594813, 0.0, 0.0,
                              0.211324865405187,  0.788675134594813 },
                            { -0.788675134594813, -0.211324865405187, 0.0, 0.0,
                              0.788675134594813,  0.211324865405187 },
                            { -0.211324865405187, -0.788675134594813, 0.0, 0.0,
                              0.211324865405187,  0.788675134594813 } };
  for (int ip = 0; ip < 8; ip++) {
    double F0 = 0.0;
    double F1 = 0.0;
    double F2 = 0.0;
    double F3 = 0.0;
    double F4 = 0.0;
    for (int r = 0; r < 6; r++) {
      F0 += (w0[r][0] * FE0[ip][r]);
      F1 += (w1[r][0] * FE0_D100[ip][r]);
      F2 += (w1[r][0] * FE0_D010[ip][r]);
      F3 += (w1[r][0] * FE0_D001[ip][r]);
      F4 += (w1[r][0] * FE0[ip][r]);
    }
    for (int j = 0; j < 6; j++) {
      A[j] += (((FE0[ip][j] * F4) +
                (((K[2] * FE0_D100[ip][j]) + (K[5] * FE0_D010[ip][j]) +
                  (K[8] * FE0_D001[ip][j])) *
                 ((K[8] * F3) + (K[5] * F2) + (K[2] * F1))) +
                (((K[1] * FE0_D100[ip][j]) + (K[4] * FE0_D010[ip][j]) +
                  (K[7] * FE0_D001[ip][j])) *
                 ((K[7] * F3) + (K[4] * F2) + (K[1] * F1))) +
                (((K[0] * FE0_D100[ip][j]) + (K[3] * FE0_D010[ip][j]) +
                  (K[6] * FE0_D001[ip][j])) *
                 ((K[6] * F3) + (K[3] * F2) + (K[0] * F1))) +
                (FE0[ip][j] * F0 * -1.0)) *
               det * W8[ip]);
    }
  }
}

void wrap_rhs(int start, int end, double *arg0_0, int *arg0_0_map0_0,
              double *arg1_0, int *arg1_0_map0_0, double *arg2_0,
              int *arg2_0_map0_0, double *arg3_0, int *arg3_0_map0_0,
              int *_arg0_0_off0_0, int *_arg1_0_off0_0, int *_arg2_0_off0_0,
              int *_arg3_0_off0_0, int layer) {
  double *arg1_0_vec[18];
  double *arg2_0_vec[6];
  double *arg3_0_vec[6];
  int xtr_arg0_0_map0_0[6];
  for (int n = start; n < end; n++) {
    int i = n;
    arg1_0_vec[0] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3;
    arg1_0_vec[1] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3;
    arg1_0_vec[2] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3;
    arg1_0_vec[3] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3;
    arg1_0_vec[4] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3;
    arg1_0_vec[5] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3;
    arg1_0_vec[6] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3 + 1;
    arg1_0_vec[7] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3 + 1;
    arg1_0_vec[8] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3 + 1;
    arg1_0_vec[9] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3 + 1;
    arg1_0_vec[10] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3 + 1;
    arg1_0_vec[11] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3 + 1;
    arg1_0_vec[12] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3 + 2;
    arg1_0_vec[13] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3 + 2;
    arg1_0_vec[14] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3 + 2;
    arg1_0_vec[15] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3 + 2;
    arg1_0_vec[16] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3 + 2;
    arg1_0_vec[17] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3 + 2;
    arg2_0_vec[0] = arg2_0 + (arg2_0_map0_0[i * 6 + 0]) * 1;
    arg2_0_vec[1] = arg2_0 + (arg2_0_map0_0[i * 6 + 1]) * 1;
    arg2_0_vec[2] = arg2_0 + (arg2_0_map0_0[i * 6 + 2]) * 1;
    arg2_0_vec[3] = arg2_0 + (arg2_0_map0_0[i * 6 + 3]) * 1;
    arg2_0_vec[4] = arg2_0 + (arg2_0_map0_0[i * 6 + 4]) * 1;
    arg2_0_vec[5] = arg2_0 + (arg2_0_map0_0[i * 6 + 5]) * 1;
    arg3_0_vec[0] = arg3_0 + (arg3_0_map0_0[i * 6 + 0]) * 1;
    arg3_0_vec[1] = arg3_0 + (arg3_0_map0_0[i * 6 + 1]) * 1;
    arg3_0_vec[2] = arg3_0 + (arg3_0_map0_0[i * 6 + 2]) * 1;
    arg3_0_vec[3] = arg3_0 + (arg3_0_map0_0[i * 6 + 3]) * 1;
    arg3_0_vec[4] = arg3_0 + (arg3_0_map0_0[i * 6 + 4]) * 1;
    arg3_0_vec[5] = arg3_0 + (arg3_0_map0_0[i * 6 + 5]) * 1;
    xtr_arg0_0_map0_0[0] = *(arg0_0_map0_0 + i * 6 + 0);
    xtr_arg0_0_map0_0[1] = *(arg0_0_map0_0 + i * 6 + 1);
    xtr_arg0_0_map0_0[2] = *(arg0_0_map0_0 + i * 6 + 2);
    xtr_arg0_0_map0_0[3] = *(arg0_0_map0_0 + i * 6 + 3);
    xtr_arg0_0_map0_0[4] = *(arg0_0_map0_0 + i * 6 + 4);
    xtr_arg0_0_map0_0[5] = *(arg0_0_map0_0 + i * 6 + 5);
    for (int j_0 = 0; j_0 < layer - 1; ++j_0) {
      double buffer_arg0_0[6] = { 0 };
      kernel_rhs(buffer_arg0_0, arg1_0_vec, arg2_0_vec, arg3_0_vec);
      for (int i_0 = 0; i_0 < 6; ++i_0) {
        *(arg0_0 + (xtr_arg0_0_map0_0[i_0]) * 1) += buffer_arg0_0[i_0 * 1 + 0];
      }
      xtr_arg0_0_map0_0[0] += _arg0_0_off0_0[0];
      xtr_arg0_0_map0_0[1] += _arg0_0_off0_0[1];
      xtr_arg0_0_map0_0[2] += _arg0_0_off0_0[2];
      xtr_arg0_0_map0_0[3] += _arg0_0_off0_0[3];
      xtr_arg0_0_map0_0[4] += _arg0_0_off0_0[4];
      xtr_arg0_0_map0_0[5] += _arg0_0_off0_0[5];
      arg1_0_vec[0] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[1] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[2] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[3] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[4] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[5] += _arg1_0_off0_0[5] * 3;
      arg1_0_vec[6] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[7] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[8] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[9] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[10] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[11] += _arg1_0_off0_0[5] * 3;
      arg1_0_vec[12] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[13] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[14] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[15] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[16] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[17] += _arg1_0_off0_0[5] * 3;
      arg2_0_vec[0] += _arg2_0_off0_0[0] * 1;
      arg2_0_vec[1] += _arg2_0_off0_0[1] * 1;
      arg2_0_vec[2] += _arg2_0_off0_0[2] * 1;
      arg2_0_vec[3] += _arg2_0_off0_0[3] * 1;
      arg2_0_vec[4] += _arg2_0_off0_0[4] * 1;
      arg2_0_vec[5] += _arg2_0_off0_0[5] * 1;
      arg3_0_vec[0] += _arg3_0_off0_0[0] * 1;
      arg3_0_vec[1] += _arg3_0_off0_0[1] * 1;
      arg3_0_vec[2] += _arg3_0_off0_0[2] * 1;
      arg3_0_vec[3] += _arg3_0_off0_0[3] * 1;
      arg3_0_vec[4] += _arg3_0_off0_0[4] * 1;
      arg3_0_vec[5] += _arg3_0_off0_0[5] * 1;
    }
  }
}

// MATRIX ASSEMBLY KERNEL
// WARNING: Do not modify this function
static void addto_vector(double *arg0_0, double buffer_arg0_0[6][6],
                         int map_size_1, int *xtr_arg0_0_map0_0, int map_size_2,
                         int *xtr_arg0_0_map1_0, int position) {

  for (int i = 0; i < map_size_1; i++) {
    for (int j = 0; j < map_size_2; j++) {
      if (buffer_arg0_0[i][j] > buffer_arg0_0[j][i]) {
        arg0_0[xtr_arg0_0_map0_0[i]] += buffer_arg0_0[i][j];
      } else {
        arg0_0[xtr_arg0_0_map1_0[i]] += buffer_arg0_0[i][j];
      }
    }
  }

  return;
}

static void kernel_lhs(double A[6][6], double **vertex_coordinates) {
  double J[9];
  J[0] = vertex_coordinates[2][0] - vertex_coordinates[0][0];
  J[1] = vertex_coordinates[4][0] - vertex_coordinates[0][0];
  J[2] = vertex_coordinates[1][0] - vertex_coordinates[0][0];
  J[3] = vertex_coordinates[8][0] - vertex_coordinates[6][0];
  J[4] = vertex_coordinates[10][0] - vertex_coordinates[6][0];
  J[5] = vertex_coordinates[7][0] - vertex_coordinates[6][0];
  J[6] = vertex_coordinates[14][0] - vertex_coordinates[12][0];
  J[7] = vertex_coordinates[16][0] - vertex_coordinates[12][0];
  J[8] = vertex_coordinates[13][0] - vertex_coordinates[12][0];
  ;
  double K[9];
  double detJ;
  do {
    const double d_00 = J[4] * J[8] - J[5] * J[7];
    const double d_01 = J[5] * J[6] - J[3] * J[8];
    const double d_02 = J[3] * J[7] - J[4] * J[6];
    const double d_10 = J[2] * J[7] - J[1] * J[8];
    const double d_11 = J[0] * J[8] - J[2] * J[6];
    const double d_12 = J[1] * J[6] - J[0] * J[7];
    const double d_20 = J[1] * J[5] - J[2] * J[4];
    const double d_21 = J[2] * J[3] - J[0] * J[5];
    const double d_22 = J[0] * J[4] - J[1] * J[3];
    detJ = J[0] * d_00 + J[3] * d_10 + J[6] * d_20;
    K[0] = d_00 / detJ;
    K[1] = d_10 / detJ;
    K[2] = d_20 / detJ;
    K[3] = d_01 / detJ;
    K[4] = d_11 / detJ;
    K[5] = d_21 / detJ;
    K[6] = d_02 / detJ;
    K[7] = d_12 / detJ;
    K[8] = d_22 / detJ;
  } while (0);
  const double det = fabs(detJ);
  double W8[8] = { 0.0795103454359941, 0.0795103454359941, 0.0454896545640056,
                   0.0454896545640056, 0.0795103454359941, 0.0795103454359941,
                   0.0454896545640056, 0.0454896545640056 };
  double FE0_D100[8][6] = {
    { -0.788675134594813, -0.211324865405187, 0.788675134594813,
      0.211324865405187,  0.0,                0.0 },
    { -0.211324865405187, -0.788675134594813, 0.211324865405187,
      0.788675134594813,  0.0,                0.0 },
    { -0.788675134594813, -0.211324865405187, 0.788675134594813,
      0.211324865405187,  0.0,                0.0 },
    { -0.211324865405187, -0.788675134594813, 0.211324865405187,
      0.788675134594813,  0.0,                0.0 },
    { -0.788675134594813, -0.211324865405187, 0.788675134594813,
      0.211324865405187,  0.0,                0.0 },
    { -0.211324865405187, -0.788675134594813, 0.211324865405187,
      0.788675134594813,  0.0,                0.0 },
    { -0.788675134594813, -0.211324865405187, 0.788675134594813,
      0.211324865405187,  0.0,                0.0 },
    { -0.211324865405187, -0.788675134594813, 0.211324865405187,
      0.788675134594813,  0.0,                0.0 }
  };
  double FE0_D001[8][6] = {
    { -0.666390246014701, 0.666390246014701,  -0.178558728263616,
      0.178558728263616,  -0.155051025721682, 0.155051025721682 },
    { -0.666390246014701, 0.666390246014701,  -0.178558728263616,
      0.178558728263616,  -0.155051025721682, 0.155051025721682 },
    { -0.280019915499074, 0.280019915499074,  -0.0750311102226081,
      0.0750311102226081, -0.644948974278318, 0.644948974278318 },
    { -0.280019915499074, 0.280019915499074,  -0.0750311102226081,
      0.0750311102226081, -0.644948974278318, 0.644948974278318 },
    { -0.178558728263616, 0.178558728263616,  -0.666390246014701,
      0.666390246014701,  -0.155051025721682, 0.155051025721682 },
    { -0.178558728263616, 0.178558728263616,  -0.666390246014701,
      0.666390246014701,  -0.155051025721682, 0.155051025721682 },
    { -0.0750311102226081, 0.0750311102226081, -0.280019915499074,
      0.280019915499074,   -0.644948974278318, 0.644948974278318 },
    { -0.0750311102226081, 0.0750311102226081, -0.280019915499074,
      0.280019915499074,   -0.644948974278318, 0.644948974278318 }
  };
  double FE0[8][6] = {
    { 0.525565416968315,  0.140824829046386, 0.140824829046386,
      0.0377338992172301, 0.122284888580111, 0.0327661371415707 },
    { 0.140824829046386, 0.525565416968315,  0.0377338992172301,
      0.140824829046386, 0.0327661371415707, 0.122284888580111 },
    { 0.22084474454546,   0.0591751709536137, 0.0591751709536137,
      0.0158559392689944, 0.508655219095739,  0.136293755182579 },
    { 0.0591751709536137, 0.22084474454546,  0.0158559392689944,
      0.0591751709536137, 0.136293755182579, 0.508655219095739 },
    { 0.140824829046386, 0.0377338992172301, 0.525565416968315,
      0.140824829046386, 0.122284888580111,  0.0327661371415707 },
    { 0.0377338992172301, 0.140824829046386,  0.140824829046386,
      0.525565416968315,  0.0327661371415707, 0.122284888580111 },
    { 0.0591751709536137, 0.0158559392689944, 0.22084474454546,
      0.0591751709536137, 0.508655219095739,  0.136293755182579 },
    { 0.0158559392689944, 0.0591751709536137, 0.0591751709536137,
      0.22084474454546,   0.136293755182579,  0.508655219095739 }
  };
  double FE0_D010[8][6] = { { -0.788675134594813, -0.211324865405187, 0.0, 0.0,
                              0.788675134594813,  0.211324865405187 },
                            { -0.211324865405187, -0.788675134594813, 0.0, 0.0,
                              0.211324865405187,  0.788675134594813 },
                            { -0.788675134594813, -0.211324865405187, 0.0, 0.0,
                              0.788675134594813,  0.211324865405187 },
                            { -0.211324865405187, -0.788675134594813, 0.0, 0.0,
                              0.211324865405187,  0.788675134594813 },
                            { -0.788675134594813, -0.211324865405187, 0.0, 0.0,
                              0.788675134594813,  0.211324865405187 },
                            { -0.211324865405187, -0.788675134594813, 0.0, 0.0,
                              0.211324865405187,  0.788675134594813 },
                            { -0.788675134594813, -0.211324865405187, 0.0, 0.0,
                              0.788675134594813,  0.211324865405187 },
                            { -0.211324865405187, -0.788675134594813, 0.0, 0.0,
                              0.211324865405187,  0.788675134594813 } };
  for (int ip = 0; ip < 8; ip++) {
    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 6; k++) {
        A[j][k] += (((FE0[ip][k] * FE0[ip][j]) +
                     (((K[2] * FE0_D100[ip][k]) + (K[5] * FE0_D010[ip][k]) +
                       (K[8] * FE0_D001[ip][k])) *
                      ((K[2] * FE0_D100[ip][j]) + (K[5] * FE0_D010[ip][j]) +
                       (K[8] * FE0_D001[ip][j]))) +
                     (((K[1] * FE0_D100[ip][k]) + (K[4] * FE0_D010[ip][k]) +
                       (K[7] * FE0_D001[ip][k])) *
                      ((K[1] * FE0_D100[ip][j]) + (K[4] * FE0_D010[ip][j]) +
                       (K[7] * FE0_D001[ip][j]))) +
                     (((K[0] * FE0_D100[ip][k]) + (K[3] * FE0_D010[ip][k]) +
                       (K[6] * FE0_D001[ip][k])) *
                      ((K[0] * FE0_D100[ip][j]) + (K[3] * FE0_D010[ip][j]) +
                       (K[6] * FE0_D001[ip][j])))) *
                    det * W8[ip]);
      }
    }
  }
}
void wrap_lhs(int start, int end, double *arg0_0, int *arg0_0_map0_0,
              int *arg0_0_map1_0, double *arg1_0, int *arg1_0_map0_0,
              int *_arg0_0_off0_0, int *_arg0_0_off1_0, int *_arg1_0_off0_0,
              int layer) {
  double *arg1_0_vec[18];
  int xtr_arg0_0_map0_0[6];
  int xtr_arg0_0_map1_0[6];
  for (int n = start; n < end; n++) {
    int i = n;
    arg1_0_vec[0] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3;
    arg1_0_vec[1] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3;
    arg1_0_vec[2] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3;
    arg1_0_vec[3] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3;
    arg1_0_vec[4] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3;
    arg1_0_vec[5] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3;
    arg1_0_vec[6] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3 + 1;
    arg1_0_vec[7] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3 + 1;
    arg1_0_vec[8] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3 + 1;
    arg1_0_vec[9] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3 + 1;
    arg1_0_vec[10] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3 + 1;
    arg1_0_vec[11] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3 + 1;
    arg1_0_vec[12] = arg1_0 + (arg1_0_map0_0[i * 6 + 0]) * 3 + 2;
    arg1_0_vec[13] = arg1_0 + (arg1_0_map0_0[i * 6 + 1]) * 3 + 2;
    arg1_0_vec[14] = arg1_0 + (arg1_0_map0_0[i * 6 + 2]) * 3 + 2;
    arg1_0_vec[15] = arg1_0 + (arg1_0_map0_0[i * 6 + 3]) * 3 + 2;
    arg1_0_vec[16] = arg1_0 + (arg1_0_map0_0[i * 6 + 4]) * 3 + 2;
    arg1_0_vec[17] = arg1_0 + (arg1_0_map0_0[i * 6 + 5]) * 3 + 2;
    xtr_arg0_0_map0_0[0] = *(arg0_0_map0_0 + i * 6 + 0);
    xtr_arg0_0_map0_0[1] = *(arg0_0_map0_0 + i * 6 + 1);
    xtr_arg0_0_map0_0[2] = *(arg0_0_map0_0 + i * 6 + 2);
    xtr_arg0_0_map0_0[3] = *(arg0_0_map0_0 + i * 6 + 3);
    xtr_arg0_0_map0_0[4] = *(arg0_0_map0_0 + i * 6 + 4);
    xtr_arg0_0_map0_0[5] = *(arg0_0_map0_0 + i * 6 + 5);
    xtr_arg0_0_map1_0[0] = *(arg0_0_map1_0 + i * 6 + 0);
    xtr_arg0_0_map1_0[1] = *(arg0_0_map1_0 + i * 6 + 1);
    xtr_arg0_0_map1_0[2] = *(arg0_0_map1_0 + i * 6 + 2);
    xtr_arg0_0_map1_0[3] = *(arg0_0_map1_0 + i * 6 + 3);
    xtr_arg0_0_map1_0[4] = *(arg0_0_map1_0 + i * 6 + 4);
    xtr_arg0_0_map1_0[5] = *(arg0_0_map1_0 + i * 6 + 5);
    for (int j_0 = 0; j_0 < layer - 1; ++j_0) {
      double buffer_arg0_0[6][6] = { { 0 } };
      kernel_lhs(buffer_arg0_0, arg1_0_vec);
      addto_vector(arg0_0, buffer_arg0_0, 6, xtr_arg0_0_map0_0, 6,
                   xtr_arg0_0_map1_0, 0);
      xtr_arg0_0_map0_0[0] += _arg0_0_off0_0[0];
      xtr_arg0_0_map0_0[1] += _arg0_0_off0_0[1];
      xtr_arg0_0_map0_0[2] += _arg0_0_off0_0[2];
      xtr_arg0_0_map0_0[3] += _arg0_0_off0_0[3];
      xtr_arg0_0_map0_0[4] += _arg0_0_off0_0[4];
      xtr_arg0_0_map0_0[5] += _arg0_0_off0_0[5];
      xtr_arg0_0_map1_0[0] += _arg0_0_off1_0[0];
      xtr_arg0_0_map1_0[1] += _arg0_0_off1_0[1];
      xtr_arg0_0_map1_0[2] += _arg0_0_off1_0[2];
      xtr_arg0_0_map1_0[3] += _arg0_0_off1_0[3];
      xtr_arg0_0_map1_0[4] += _arg0_0_off1_0[4];
      xtr_arg0_0_map1_0[5] += _arg0_0_off1_0[5];
      arg1_0_vec[0] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[1] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[2] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[3] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[4] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[5] += _arg1_0_off0_0[5] * 3;
      arg1_0_vec[6] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[7] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[8] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[9] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[10] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[11] += _arg1_0_off0_0[5] * 3;
      arg1_0_vec[12] += _arg1_0_off0_0[0] * 3;
      arg1_0_vec[13] += _arg1_0_off0_0[1] * 3;
      arg1_0_vec[14] += _arg1_0_off0_0[2] * 3;
      arg1_0_vec[15] += _arg1_0_off0_0[3] * 3;
      arg1_0_vec[16] += _arg1_0_off0_0[4] * 3;
      arg1_0_vec[17] += _arg1_0_off0_0[5] * 3;
    }
  }
}
