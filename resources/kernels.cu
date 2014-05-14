// See comment inside, but efill_mvf is broken

// Produces functions in the form
//  kind is
//   s scalar
//   v vector
//   m matrix
//  element by element functions
//   e<name>_<kind>f

#include </opt/cuda/include/math_constants.h>

#define FN2(a1, op, a2)                         \
  op(a1,a2)

#define OP2(a1, op, a2)                         \
  (a1 op a2)

#define FN_SECOND(a1, op, a2)                   \
  (op(a2))

#define FN_FIRST(a1, op, a2)                    \
  (op(a1))

#define FN_ELEMENT_A(NAME, PRE, POST)                                   \
  extern "C" __global__                                                 \
  void e ## NAME ## _vf(size_t n, float *x, int lx,                     \
                        float *result, int lr) {                        \
    int id = threadIdx.x + blockIdx.x * blockDim.x;                     \
    if(id < n) {                                                        \
      result[id*lr] = PRE(x[id*lx])POST;                                \
    }                                                                   \
  }                                                                     \
  extern "C" __global__                                                 \
  void e ## NAME ## _mf(int rs, int cs,                                 \
                        float *A, int lda,                              \
                        float *B, int ldb) {                            \
    int r = blockIdx.y * blockDim.y + threadIdx.y;                      \
    int c = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if(r < rs && c < cs) {                                              \
      B[r*ldb+c] = PRE(A[r*lda+c])POST;                                 \
    }                                                                   \
  }

#define FN_ELEMENT_AS(NAME, PRE, OP, POST)                      \
  extern "C" __global__                                         \
  void e ## NAME ## _vsf(size_t n, float *x, int lx,            \
                         float y, float *result, int lr) {      \
    int id = threadIdx.x + blockIdx.x * blockDim.x;             \
    if(id < n) {                                                \
      result[id*lr] = PRE(x[id*lx], OP, y)POST;                 \
    }                                                           \
  }                                                             \
  extern "C" __global__                                         \
  void e ## NAME ## _msf(int rs, int cs,                        \
                         float *A, int lda,                     \
                         float B,                               \
                         float *C, int ldc) {                   \
    int r = blockIdx.y * blockDim.y + threadIdx.y;              \
    int c = blockIdx.x * blockDim.x + threadIdx.x;              \
    if(r < rs && c < cs) {                                      \
      C[r*ldc+c] = PRE(A[r*lda+c], OP, B)POST;                  \
    }                                                           \
  }

#define FN_ELEMENT_SA(NAME, PRE, OP, POST)                             \
  extern "C" __global__                                                \
  void e ## NAME ## _svf(size_t n, float x,                            \
                         float *y, int ly, float *result, int lr) {    \
    int id = threadIdx.x + blockIdx.x * blockDim.x;                    \
    if(id < n) {                                                       \
      result[id*lr] = PRE(x, OP, y[id*ly])POST;                        \
    }                                                                  \
  }                                                                    \
  extern "C" __global__                                                \
  void e ## NAME ## _smf(int rs, int cs,                               \
                         float A,                                      \
                         float *B, int ldb,                            \
                         float *C, int ldc) {                          \
    int r = blockIdx.y * blockDim.y + threadIdx.y;                     \
    int c = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if(r < rs && c < cs) {                                             \
      C[r*ldc+c] = PRE(A, OP, B[r*ldb+c])POST;                         \
    }                                                                  \
  }

#define FN_ELEMENT_AA(NAME, PRE, OP, POST)                              \
  extern "C" __global__                                                 \
    void e ## NAME ## _vvf(size_t n, float *x, int lx,                  \
                           float *y, int ly, float *result, int lr) {   \
    int id = threadIdx.x + blockIdx.x * blockDim.x;                     \
    if(id < n) {                                                        \
      result[id*lr] = PRE(x[id*lx], OP, y[id*ly])POST;                  \
    }                                                                   \
  }                                                                     \
  extern "C" __global__                                                 \
  void e ## NAME ## _vmf(int rs, int cs,                                \
                         float *x, int lx,                              \
                         float *B, int ldb,                             \
                         float *C, int ldc) {                           \
    int r = blockIdx.y * blockDim.y + threadIdx.y;                      \
    int c = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if(r < rs && c < cs) {                                              \
      C[r*ldc+c] = PRE(x[c*lx], OP, B[r*ldb+c])POST;                    \
    }                                                                   \
  }                                                                     \
  extern "C" __global__                                                 \
  void e ## NAME ## _mvf(int rs, int cs,                                \
                         float *A, int lda,                             \
                         float *y, int ly,                              \
                         float *C, int ldc) {                           \
    int r = blockIdx.y * blockDim.y + threadIdx.y;                      \
    int c = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if(r < rs && c < cs) {                                              \
      C[r*ldc+c] = PRE(A[r*lda+c], OP, y[c*ly])POST;                    \
    }                                                                   \
  }                                                                     \
  extern "C" __global__                                                 \
  void e ## NAME ## _mmf(int rs, int cs,                                \
                         float *A, int lda,                             \
                         float *B, int ldb,                             \
                         float *C, int ldc) {                           \
    int r = blockIdx.y * blockDim.y + threadIdx.y;                      \
    int c = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if(r < rs && c < cs) {                                              \
      C[r*ldc+c] = PRE(A[r*lda+c], OP, B[r*ldb+c])POST;                 \
    }                                                                   \
  }

#define FN_ELEMENT_SIMPLE_A(NAME) \
  FN_ELEMENT_A(NAME, NAME ## f, )

#define FN_ELEMENT_AS_SA_AA(NAME, PRE, OP, POST)        \
  FN_ELEMENT_AS(NAME, PRE, OP, POST);                   \
  FN_ELEMENT_SA(NAME, PRE, OP, POST);                   \
  FN_ELEMENT_AA(NAME, PRE, OP, POST)

// These are less efficient than they could be, they take an extra argument useless argument
// They also generate an incorrect efill_mvf
FN_ELEMENT_AS(fill, FN_SECOND, , )
FN_ELEMENT_SA(fill, FN_FIRST, , )
FN_ELEMENT_AA(fill, FN_FIRST, , )

FN_ELEMENT_AS_SA_AA(add, OP2, +, )
FN_ELEMENT_AS_SA_AA(sub, OP2, -, )
FN_ELEMENT_AS_SA_AA(mul, OP2, *, )
FN_ELEMENT_AS_SA_AA(div, OP2, /, )

FN_ELEMENT_AS_SA_AA(pow, FN2, powf, )
FN_ELEMENT_AS_SA_AA(dim, FN2, fdimf, )
FN_ELEMENT_AS_SA_AA(max, FN2, fmaxf, )
FN_ELEMENT_AS_SA_AA(min, FN2, fminf, )
FN_ELEMENT_AS_SA_AA(mod, FN2, fmodf, )

FN_ELEMENT_AA(lt,  OP2, <,  ?1.0f:0.0f)
FN_ELEMENT_AA(lte, OP2, <=, ?1.0f:0.0f)
FN_ELEMENT_AA(eq,  OP2, ==, ?1.0f:0.0f)
FN_ELEMENT_AA(gte, OP2, >=, ?1.0f:0.0f)
FN_ELEMENT_AA(gt,  OP2, >,  ?1.0f:0.0f)
FN_ELEMENT_AA(ne,  OP2, !=, ?1.0f:0.0f)

FN_ELEMENT_AS(lt,  OP2, <,  ?1.0f:0.0f)
FN_ELEMENT_AS(lte, OP2, <=, ?1.0f:0.0f)
FN_ELEMENT_AS(eq,  OP2, ==, ?1.0f:0.0f)
FN_ELEMENT_AS(gte, OP2, >=, ?1.0f:0.0f)
FN_ELEMENT_AS(gt,  OP2, >,  ?1.0f:0.0f)
FN_ELEMENT_AS(ne,  OP2, !=, ?1.0f:0.0f)

FN_ELEMENT_A(negate, -, )

FN_ELEMENT_SIMPLE_A(cos)
FN_ELEMENT_SIMPLE_A(cosh)
FN_ELEMENT_SIMPLE_A(sin)
FN_ELEMENT_SIMPLE_A(sinh)
FN_ELEMENT_SIMPLE_A(tan)
FN_ELEMENT_SIMPLE_A(tanh)
FN_ELEMENT_SIMPLE_A(acos)
FN_ELEMENT_SIMPLE_A(acosh)
FN_ELEMENT_SIMPLE_A(asin)
FN_ELEMENT_SIMPLE_A(asinh)
FN_ELEMENT_SIMPLE_A(atan)
FN_ELEMENT_SIMPLE_A(atanh)
FN_ELEMENT_AA(atan2, FN2, atan2f, )
FN_ELEMENT_SIMPLE_A(cospi)
FN_ELEMENT_SIMPLE_A(sinpi)

FN_ELEMENT_SIMPLE_A(erfc)
FN_ELEMENT_SIMPLE_A(erfcinv)
FN_ELEMENT_SIMPLE_A(erfcx)
FN_ELEMENT_SIMPLE_A(erf)
FN_ELEMENT_SIMPLE_A(erfinv)

FN_ELEMENT_SIMPLE_A(ceil)
FN_ELEMENT_SIMPLE_A(floor)
FN_ELEMENT_SIMPLE_A(round)
FN_ELEMENT_SIMPLE_A(rint)
FN_ELEMENT_SIMPLE_A(trunc)

FN_ELEMENT_SIMPLE_A(exp10)
FN_ELEMENT_SIMPLE_A(exp2)
FN_ELEMENT_SIMPLE_A(exp)
FN_ELEMENT_SIMPLE_A(expm1)

FN_ELEMENT_SIMPLE_A(log10)
FN_ELEMENT_SIMPLE_A(log2)
FN_ELEMENT_SIMPLE_A(log)
FN_ELEMENT_SIMPLE_A(log1p)

FN_ELEMENT_SIMPLE_A(cbrt)
FN_ELEMENT_SIMPLE_A(sqrt)
FN_ELEMENT_SIMPLE_A(fabs)
FN_ELEMENT_SIMPLE_A(j0)
FN_ELEMENT_SIMPLE_A(j1)
FN_ELEMENT_SIMPLE_A(lgamma)
FN_ELEMENT_SIMPLE_A(tgamma)

FN_ELEMENT_A(signum, (float)signbit, )

FN_ELEMENT_A(deg_to_rad, 2.0f*CUDART_PI_F/360.0f*, )
FN_ELEMENT_A(rad_to_deg, 360.0f/(2.0f*CUDART_PI_F)*, )
