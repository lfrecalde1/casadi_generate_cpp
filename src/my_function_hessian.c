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
  #define CASADI_PREFIX(ID) my_function_hessian_ ## ID
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
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)

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

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

static const casadi_int casadi_s0[7] = {1, 2, 0, 1, 2, 0, 0};
static const casadi_int casadi_s1[9] = {2, 2, 0, 2, 4, 0, 1, 0, 1};
static const casadi_int casadi_s2[6] = {1, 2, 0, 1, 1, 0};
static const casadi_int casadi_s3[6] = {1, 2, 0, 0, 1, 0};
static const casadi_int casadi_s4[6] = {2, 1, 0, 2, 0, 1};

/* hessian:(Q[2x2],c[2],x[2],xd[2])->(hessian[2x2]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j;
  casadi_real *rr, *ss;
  const casadi_real *cs;
  casadi_real *w0=w+1, w1, *w2=w+6, w3, *w4=w+9, *w5=w+13, *w6=w+15;
  /* #0: @0 = zeros(2x2) */
  casadi_clear(w0, 4);
  /* #1: @1 = 0.5 */
  w1 = 5.0000000000000000e-01;
  /* #2: @2 = zeros(1x2) */
  casadi_clear(w2, 2);
  /* #3: @3 = ones(1x2,1nz) */
  w3 = 1.;
  /* #4: @4 = input[0][0] */
  casadi_copy(arg[0], 4, w4);
  /* #5: @2 = mac(@3,@4,@2) */
  casadi_mtimes((&w3), casadi_s2, w4, casadi_s1, w2, casadi_s0, w, 0);
  /* #6: @2 = @2' */
  /* #7: @2 = (@1*@2) */
  for (i=0, rr=w2, cs=w2; i<2; ++i) (*rr++)  = (w1*(*cs++));
  /* #8: @5 = zeros(1x2) */
  casadi_clear(w5, 2);
  /* #9: @3 = all_0.5(1x2,1nz) */
  w3 = 5.0000000000000000e-01;
  /* #10: @6 = @4' */
  for (i=0, rr=w6, cs=w4; i<2; ++i) for (j=0; j<2; ++j) rr[i+j*2] = *cs++;
  /* #11: @5 = mac(@3,@6,@5) */
  casadi_mtimes((&w3), casadi_s2, w6, casadi_s1, w5, casadi_s0, w, 0);
  /* #12: @5 = @5' */
  /* #13: @2 = (@2+@5) */
  for (i=0, rr=w2, cs=w5; i<2; ++i) (*rr++) += (*cs++);
  /* #14: (@0[:4:2] = @2) */
  for (rr=w0+0, ss=w2; rr!=w0+4; rr+=2) *rr = *ss++;
  /* #15: @2 = zeros(1x2) */
  casadi_clear(w2, 2);
  /* #16: @3 = ones(1x2,1nz) */
  w3 = 1.;
  /* #17: @2 = mac(@3,@4,@2) */
  casadi_mtimes((&w3), casadi_s3, w4, casadi_s1, w2, casadi_s0, w, 0);
  /* #18: @2 = @2' */
  /* #19: @2 = (@1*@2) */
  for (i=0, rr=w2, cs=w2; i<2; ++i) (*rr++)  = (w1*(*cs++));
  /* #20: @5 = zeros(1x2) */
  casadi_clear(w5, 2);
  /* #21: @1 = all_0.5(1x2,1nz) */
  w1 = 5.0000000000000000e-01;
  /* #22: @5 = mac(@1,@6,@5) */
  casadi_mtimes((&w1), casadi_s3, w6, casadi_s1, w5, casadi_s0, w, 0);
  /* #23: @5 = @5' */
  /* #24: @2 = (@2+@5) */
  for (i=0, rr=w2, cs=w5; i<2; ++i) (*rr++) += (*cs++);
  /* #25: (@0[1:5:2] = @2) */
  for (rr=w0+1, ss=w2; rr!=w0+5; rr+=2) *rr = *ss++;
  /* #26: @6 = @0' */
  for (i=0, rr=w6, cs=w0; i<2; ++i) for (j=0; j<2; ++j) rr[i+j*2] = *cs++;
  /* #27: output[0][0] = @6 */
  casadi_copy(w6, 4, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int hessian(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int hessian_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int hessian_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void hessian_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int hessian_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void hessian_release(int mem) {
}

CASADI_SYMBOL_EXPORT void hessian_incref(void) {
}

CASADI_SYMBOL_EXPORT void hessian_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int hessian_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int hessian_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real hessian_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* hessian_name_in(casadi_int i) {
  switch (i) {
    case 0: return "Q";
    case 1: return "c";
    case 2: return "x";
    case 3: return "xd";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* hessian_name_out(casadi_int i) {
  switch (i) {
    case 0: return "hessian";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* hessian_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s4;
    case 2: return casadi_s4;
    case 3: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* hessian_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int hessian_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 19;
  return 0;
}

CASADI_SYMBOL_EXPORT int hessian_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 19*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
