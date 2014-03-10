/* Get current time of day in microseconds.
 *
 * Paul Kelly, Gheorghe-Teodor Bercea, Fabio Luporini - Imperial College London - 2014
 */

extern void wrap_expression_1(int start, int end,
                              double * arg0_0, int *arg0_0_map0_0, double *arg1_0, int *arg1_0_map0_0,
                              int *_arg0_0_off0_0, int *_arg1_0_off0_0 , int layer);

extern void wrap_zero_1(int start, int end, double *arg0_0, int layer);

extern void wrap_rhs_1(int start, int end,
		               double *arg0_0, int *arg0_0_map0_0,
		               double *arg1_0, int *arg1_0_map0_0,
		               double *arg2_0, int *arg2_0_map0_0,
		               int *_arg0_0_off0_0, int *_arg1_0_off0_0,
		               int *_arg2_0_off0_0 , int layer);

extern void wrap_expression_2(int start, int end, double *arg0_0, double *arg1_0, int layers);

extern void wrap_rhs(int start, int end,
			         double *arg0_0, int *arg0_0_map0_0,
			         double *arg1_0, int *arg1_0_map0_0,
			         double *arg2_0, int *arg2_0_map0_0,
			         double *arg3_0, int *arg3_0_map0_0,
			         int *_arg0_0_off0_0, int *_arg1_0_off0_0,
			         int *_arg2_0_off0_0, int *_arg3_0_off0_0,
			         int layer);

extern void wrap_lhs(int start, int end,
                     double *arg0_0, int *arg0_0_map0_0, int *arg0_0_map1_0,
                     double *arg1_0, int *arg1_0_map0_0,
                     int *_arg0_0_off0_0, int *_arg0_0_off1_0, int *_arg1_0_off0_0, int layer);
