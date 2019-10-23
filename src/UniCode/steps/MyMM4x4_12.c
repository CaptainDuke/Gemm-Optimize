/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */

#define mc 256
#define kc 128
#define min(i, j) ( (i) < (j) ? (i) : (j))



void AddDot( int, float *, int, float *, float * );
void AddDot4x4(int, float *, int, float *, int, float *, int);

void MY_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;

  int p, pb, ib;

  for ( p=0; p<k; p+=kc ){        
    pb = min(k - p, kc);      // pb = p_block = 256
    for ( i=0; i<m; i+=mc ){
      ib = min(m - i, mc);        // ib = i_block = 128


      InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
   
    }
  }
}

void InnerKernel(int m, int n, int k, float *a, int lda,
                                      float *b, int ldb,
                                      float *c, int ldc)
{
  int i, j;
  float packedA[ m * k];
  for(i = 0; i < m; i+=4)
    for(j = 0; j < n; j+=4){
      PackMatrixA(k, &A(i, 0), lda, &packedA[i*k]);
      AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
}    

void PackMatrixA(int k, float *a, int lda, float *a_to){

}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

typedef union
{
  __m128 v;
  float d[4];
} v2f_t;




void AddDot4x4(int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
 


  /*
  register float
    c_00_reg, c_01_reg, c_02_reg, c_03_reg, 
    c_10_reg, c_11_reg, c_12_reg, c_13_reg, 
    c_20_reg, c_21_reg, c_22_reg, c_23_reg, 
    c_30_reg, c_31_reg, c_32_reg, c_33_reg,

    // store B(p, 0-3)
    b_p0_reg,  b_p1_reg,  b_p2_reg,  b_p3_reg,
    
    a_0p_reg,
    a_1p_reg,
    a_2p_reg,
    a_3p_reg;
  */

  v2f_t
    c00_c01_c02_c03_vreg, c10_c11_c12_c13_vreg, c20_c21_c22_c23_vreg, c30_c31_c32_c33_vreg,
    
    bp0_bp1_bp2_bp3_vreg,

    a_0p_vreg,
    a_1p_vreg,
    a_2p_vreg,
    a_3p_vreg;



    float *a_0p_ptr, 
          *a_1p_ptr, 
          *a_2p_ptr, 
          *a_3p_ptr;

    a_0p_ptr = &A(0, 0);
    a_1p_ptr = &A(1, 0);
    a_2p_ptr = &A(2, 0);
    a_3p_ptr = &A(3, 0);

    c00_c01_c02_c03_vreg.v = _mm_setzero_ps();
    c10_c11_c12_c13_vreg.v = _mm_setzero_ps();
    c20_c21_c22_c23_vreg.v = _mm_setzero_ps();
    c30_c31_c32_c33_vreg.v = _mm_setzero_ps();

    
    // c_00_reg = 0.0; c_01_reg = 0.0; c_02_reg = 0.0; c_03_reg = 0.0; 
    // c_10_reg = 0.0; c_11_reg = 0.0; c_12_reg = 0.0; c_13_reg = 0.0; 
    // c_20_reg = 0.0; c_21_reg = 0.0; c_22_reg = 0.0; c_23_reg = 0.0; 
    // c_30_reg = 0.0; c_31_reg = 0.0; c_32_reg = 0.0; c_33_reg = 0.0;

    for(int p = 0; p < k; p++){

      bp0_bp1_bp2_bp3_vreg.v = _mm_loadu_ps((float*) &B(p, 0));

      a_0p_vreg.v = _mm_load_ps1((float*) a_0p_ptr++);
      a_1p_vreg.v = _mm_load_ps1((float*) a_1p_ptr++);
      a_2p_vreg.v = _mm_load_ps1((float*) a_2p_ptr++);
      a_3p_vreg.v = _mm_load_ps1((float*) a_3p_ptr++);

      //first row and second row
      //c00_c01_c02_c03_vreg.v += a_0p_vreg.v * bp0_bp1_bp2_bp3_vreg.v;
      //c10_c11_c12_c13_vreg.v += a_1p_vreg.v * bp0_bp1_bp2_bp3_vreg.v;

      c00_c01_c02_c03_vreg.v += _mm_mul_ps(a_0p_vreg.v , bp0_bp1_bp2_bp3_vreg.v);
      c10_c11_c12_c13_vreg.v += _mm_mul_ps(a_1p_vreg.v , bp0_bp1_bp2_bp3_vreg.v);
      // Third and fourth row
      //c20_c21_c22_c23_vreg.v += a_2p_vreg.v * bp0_bp1_bp2_bp3_vreg.v;
      //c30_c31_c32_c33_vreg.v += a_3p_vreg.v * bp0_bp1_bp2_bp3_vreg.v;
      c20_c21_c22_c23_vreg.v += _mm_mul_ps(a_2p_vreg.v , bp0_bp1_bp2_bp3_vreg.v);
      c30_c31_c32_c33_vreg.v += _mm_mul_ps(a_3p_vreg.v , bp0_bp1_bp2_bp3_vreg.v);



    }

    C(0, 0) += c00_c01_c02_c03_vreg.d[0]; C(0, 1) += c00_c01_c02_c03_vreg.d[1]; C(0, 2) += c00_c01_c02_c03_vreg.d[2]; C(0, 3) += c00_c01_c02_c03_vreg.d[3];
    C(1, 0) += c10_c11_c12_c13_vreg.d[0]; C(1, 1) += c10_c11_c12_c13_vreg.d[1]; C(1, 2) += c10_c11_c12_c13_vreg.d[2]; C(1, 3) += c10_c11_c12_c13_vreg.d[3];
    C(2, 0) += c20_c21_c22_c23_vreg.d[0]; C(2, 1) += c20_c21_c22_c23_vreg.d[1]; C(2, 2) += c20_c21_c22_c23_vreg.d[2]; C(2, 3) += c20_c21_c22_c23_vreg.d[3];
    C(3, 0) += c30_c31_c32_c33_vreg.d[0]; C(3, 1) += c30_c31_c32_c33_vreg.d[1]; C(3, 2) += c30_c31_c32_c33_vreg.d[2]; C(3, 3) += c30_c31_c32_c33_vreg.d[3];
  
}

