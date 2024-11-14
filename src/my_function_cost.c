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

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) my_function_cost_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real float
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

static const casadi_int casadi_s0[9] = {2, 2, 0, 2, 4, 0, 1, 0, 1};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};

/* cost:(Q[2x2],c[2],x[2],xd[2])->(cost) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, w1, *w2=w+3, *w3=w+5, *w4=w+7, *w5=w+9;
  /* #0: @0 = 0.5 */
  w0 = 5.0000000000000000e-01;
  /* #1: @1 = 0 */
  w1 = 0.;
  /* #2: @2 = zeros(1x2) */
  casadi_clear(w2, 2);
  /* #3: @3 = input[2][0] */
  casadi_copy(arg[2], 2, w3);
  /* #4: @4 = input[3][0] */
  casadi_copy(arg[3], 2, w4);
  /* #5: @3 = (@3-@4) */
  for (i=0, rr=w3, cs=w4; i<2; ++i) (*rr++) -= (*cs++);
  /* #6: @4 = @3' */
  casadi_copy(w3, 2, w4);
  /* #7: @5 = input[0][0] */
  casadi_copy(arg[0], 4, w5);
  /* #8: @2 = mac(@4,@5,@2) */
  for (i=0, rr=w2; i<2; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w4+j, tt=w5+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #9: @1 = mac(@2,@3,@1) */
  for (i=0, rr=(&w1); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w2+j, tt=w3+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #10: @0 = (@0*@1) */
  w0 *= w1;
  /* #11: @1 = 0 */
  w1 = 0.;
  /* #12: @2 = input[1][0] */
  casadi_copy(arg[1], 2, w2);
  /* #13: @2 = @2' */
  /* #14: @1 = mac(@2,@3,@1) */
  for (i=0, rr=(&w1); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w2+j, tt=w3+i*2; k<2; ++k) *rr += ss[k*1]**tt++;
  /* #15: @0 = (@0+@1) */
  w0 += w1;
  /* #16: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int cost(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int cost_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int cost_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void cost_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int cost_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void cost_release(int mem) {
}

CASADI_SYMBOL_EXPORT void cost_incref(void) {
}

CASADI_SYMBOL_EXPORT void cost_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int cost_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int cost_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real cost_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* cost_name_in(casadi_int i) {
  switch (i) {
    case 0: return "Q";
    case 1: return "c";
    case 2: return "x";
    case 3: return "xd";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* cost_name_out(casadi_int i) {
  switch (i) {
    case 0: return "cost";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* cost_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* cost_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int cost_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 13;
  return 0;
}

CASADI_SYMBOL_EXPORT int cost_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 13*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif