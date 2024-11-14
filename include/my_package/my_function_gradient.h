/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real float
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

int gradient(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int gradient_alloc_mem(void);
int gradient_init_mem(int mem);
void gradient_free_mem(int mem);
int gradient_checkout(void);
void gradient_release(int mem);
void gradient_incref(void);
void gradient_decref(void);
casadi_int gradient_n_in(void);
casadi_int gradient_n_out(void);
casadi_real gradient_default_in(casadi_int i);
const char* gradient_name_in(casadi_int i);
const char* gradient_name_out(casadi_int i);
const casadi_int* gradient_sparsity_in(casadi_int i);
const casadi_int* gradient_sparsity_out(casadi_int i);
int gradient_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int gradient_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define gradient_SZ_ARG 7
#define gradient_SZ_RES 2
#define gradient_SZ_IW 0
#define gradient_SZ_W 18
#ifdef __cplusplus
} /* extern "C" */
#endif