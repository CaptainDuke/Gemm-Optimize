/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */

#define nc 256
#define kc 128
#define nb 1000
#define min(i, j) ( (i) < (j) ? (i) : (j))


void PackMatrixA( int, float *, int, float * );
void PackMatrixB( int, float *, int, float * );
void InnerKernel( int, int, int, float *, int, float *, int, float *, int , int);

void AddDot4x4(int, float *, int, float *, int, float *, int);

void MY_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;

  int p, pb, jb;

  for ( p=0; p<k; p+=kc ){        
    pb = min(k - p, kc);      // pb = p_block = 256
    for ( j=0; j<n; j+=nc ){
      jb = min(n - j, nc);        // jb = j_block = 128

      InnerKernel(m, jb, pb, &A(0, p), lda, &B(p, j), ldb, &C(0, j), ldc, j==0);
   
    }
  }
}

void InnerKernel(int m, int n, int k, float *a, int lda,
                                      float *b, int ldb,
                                      float *c, int ldc, int first_time)
{
  int i, j;
  float packedB[k * n];
  static float packedA[ kc * nb];

  for(i = 0; i < m; i+=4){
    if(first_time)
      PackMatrixA(k, &A(i, 0), lda, &packedA[i*k]);
    for(j = 0; j < n; j+=4){
      if(i == 0)
        PackMatrixB(k, &B(0, j), ldb, &packedB[k * j]);
      AddDot4x4(k, &packedA[i * k], k, &packedB[k*j], 4, &C(i, j), ldc);
      // AddDot4x4(k, &A(i, 0), lda, &packedB[k*j], 4, &C(i, j), ldc);
    }
  }
}                                      

void PackMatrixA( int k, float *a, int lda, float *a_to)
{
  int j;
  float
    *a_0j_ptr = &A(0, 0),
    *a_1j_ptr = &A(1, 0),
    *a_2j_ptr = &A(2, 0),
    *a_3j_ptr = &A(3, 0);

  for(j = 0; j < k; j++){
    *a_to++ = *a_0j_ptr++;
    *a_to++ = *a_1j_ptr++;
    *a_to++ = *a_2j_ptr++;
    *a_to++ = *a_3j_ptr++;
  }
}

void PackMatrixB( int k, float *b, int ldb, float *b_to)
{
  int i;
  for(i = 0; i < k; i++){
    float *b_ij_ptr = &B(i, 0);

    *b_to     = *b_ij_ptr;
    *(b_to+1) = *(b_ij_ptr+1);
    *(b_to+2) = *(b_ij_ptr+2);
    *(b_to+3) = *(b_ij_ptr+3);

    b_to += 4;
  }
}


#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>  // fma

typedef union
{
  __m128 v;
  float d[4];
} v2f_t;




void AddDot4x4(int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
 

  v2f_t
    c00_c01_c02_c03_vreg, c10_c11_c12_c13_vreg, c20_c21_c22_c23_vreg, c30_c31_c32_c33_vreg,
    
    bp0_bp1_bp2_bp3_vreg,

    a_0p_vreg,
    a_1p_vreg,
    a_2p_vreg,
    a_3p_vreg;



    // float *a_0p_ptr, 
    //       *a_1p_ptr, 
    //       *a_2p_ptr, 
    //       *a_3p_ptr;

    // a_0p_ptr = &A(0, 0);
    // a_1p_ptr = &A(1, 0);
    // a_2p_ptr = &A(2, 0);
    // a_3p_ptr = &A(3, 0);

    c00_c01_c02_c03_vreg.v = _mm_setzero_ps();
    c10_c11_c12_c13_vreg.v = _mm_setzero_ps();
    c20_c21_c22_c23_vreg.v = _mm_setzero_ps();
    c30_c31_c32_c33_vreg.v = _mm_setzero_ps();

    

    for(int p = 0; p < k; p++){

      // bp0_bp1_bp2_bp3_vreg.v = _mm_loadu_ps((float*) &B(p, 0));
      bp0_bp1_bp2_bp3_vreg.v = _mm_loadu_ps((float *) b);
      b+=4;

      a_0p_vreg.v = _mm_load_ps1((float*) a);
      a_1p_vreg.v = _mm_load_ps1((float*) (a+1) );
      a_2p_vreg.v = _mm_load_ps1((float*) (a+2) );
      a_3p_vreg.v = _mm_load_ps1((float*) (a+3) );
      a+=4;

      //first row and second row
      // sse
      // c00_c01_c02_c03_vreg.v += _mm_mul_ps(a_0p_vreg.v , bp0_bp1_bp2_bp3_vreg.v);
      // c10_c11_c12_c13_vreg.v += _mm_mul_ps(a_1p_vreg.v , bp0_bp1_bp2_bp3_vreg.v);
      
      // fma
      c00_c01_c02_c03_vreg.v = _mm_fmadd_ps(a_0p_vreg.v, bp0_bp1_bp2_bp3_vreg.v, c00_c01_c02_c03_vreg.v);
      c10_c11_c12_c13_vreg.v = _mm_fmadd_ps(a_1p_vreg.v, bp0_bp1_bp2_bp3_vreg.v, c10_c11_c12_c13_vreg.v);

      // naive
      // c00_c01_c02_c03_vreg.v += a_0p_vreg.v * bp0_bp1_bp2_bp3_vreg.v;
      // c10_c11_c12_c13_vreg.v += a_1p_vreg.v * bp0_bp1_bp2_bp3_vreg.v;
    

      // Third and fourth row
      // sse
      // c20_c21_c22_c23_vreg.v += _mm_mul_ps(a_2p_vreg.v , bp0_bp1_bp2_bp3_vreg.v);
      // c30_c31_c32_c33_vreg.v += _mm_mul_ps(a_3p_vreg.v , bp0_bp1_bp2_bp3_vreg.v);

      // fma
      c20_c21_c22_c23_vreg.v = _mm_fmadd_ps(a_2p_vreg.v, bp0_bp1_bp2_bp3_vreg.v, c20_c21_c22_c23_vreg.v);
      c30_c31_c32_c33_vreg.v = _mm_fmadd_ps(a_3p_vreg.v, bp0_bp1_bp2_bp3_vreg.v, c30_c31_c32_c33_vreg.v);

      // naive
      // c20_c21_c22_c23_vreg.v += a_2p_vreg.v * bp0_bp1_bp2_bp3_vreg.v;
      // c30_c31_c32_c33_vreg.v += a_3p_vreg.v * bp0_bp1_bp2_bp3_vreg.v;



    }

    // C(0, 0) += c00_c01_c02_c03_vreg.d[0]; C(0, 1) += c00_c01_c02_c03_vreg.d[1]; C(0, 2) += c00_c01_c02_c03_vreg.d[2]; C(0, 3) += c00_c01_c02_c03_vreg.d[3];
    // C(1, 0) += c10_c11_c12_c13_vreg.d[0]; C(1, 1) += c10_c11_c12_c13_vreg.d[1]; C(1, 2) += c10_c11_c12_c13_vreg.d[2]; C(1, 3) += c10_c11_c12_c13_vreg.d[3];
    // C(2, 0) += c20_c21_c22_c23_vreg.d[0]; C(2, 1) += c20_c21_c22_c23_vreg.d[1]; C(2, 2) += c20_c21_c22_c23_vreg.d[2]; C(2, 3) += c20_c21_c22_c23_vreg.d[3];
    // C(3, 0) += c30_c31_c32_c33_vreg.d[0]; C(3, 1) += c30_c31_c32_c33_vreg.d[1]; C(3, 2) += c30_c31_c32_c33_vreg.d[2]; C(3, 3) += c30_c31_c32_c33_vreg.d[3];

    __m128 C00_03 = _mm_loadu_ps(&C(0,0));
    __m128 C10_13 = _mm_loadu_ps(&C(1,0));
    __m128 C20_23 = _mm_loadu_ps(&C(2,0));
    __m128 C30_33 = _mm_loadu_ps(&C(3,0));

    C00_03 = _mm_add_ps(C00_03, c00_c01_c02_c03_vreg.v);
    C10_13 = _mm_add_ps(C10_13, c10_c11_c12_c13_vreg.v);
    C20_23 = _mm_add_ps(C20_23, c20_c21_c22_c23_vreg.v);
    C30_33 = _mm_add_ps(C30_33, c30_c31_c32_c33_vreg.v);

    _mm_storeu_ps(&C(0,0), C00_03);
    _mm_storeu_ps(&C(1,0), C10_13);
    _mm_storeu_ps(&C(2,0), C20_23);
    _mm_storeu_ps(&C(3,0), C30_33);


}

