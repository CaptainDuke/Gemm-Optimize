/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

#define mc 256
#define kc 128

#define min(i, j) ((i) < (j) ? (i): (j))



void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void PackMatrixA( int, double *, int, double *);
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j, p, pb, ib;

  for (p = 0; p < k; p += kc){
    pb = min(k - p, kc);
    for(i = 0; i < m; i += mc){
      ib = min(m - i, mc);
      InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
    }
  }
}

void InnerKernel(int m, int n, int k, double *a, int lda,
                                      double *b, int ldb,
                                      double *c, int ldc)
{
  int i, j;
  double packedA[ m * k];
  for (int j = 0; j < n; j += 4){
    for(int i = 0; i < m; i += 4){
      if(j==0)
        PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]);
      //AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
      AddDot4x4(k, &packedA[ i * k ], 4, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

void PackMatrixA(int k, double *a, int lda, double *a_to)
{
  int j;
  for(j = 0; j< k; j++)
  {
    double * a_ij_pntr = &A(0,j);

    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr+1);
    *a_to++ = *(a_ij_pntr+2);
    *a_to++ = *(a_ij_pntr+3);

  }
}

#include <mmintrin.h>
#include <xmmintrin.h> // SSE
#include <pmmintrin.h> // SSE2
#include <emmintrin.h> // SSE3

///* Create macro to let X( i ) equal the ith element of x */
//
//#define X(i) x[ (i)*incx ]

typedef union
{
  __m128d v;
  double d[2];

} v2df_t;

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
  // First row

  int p = 0;
    /* hold contributions to
       C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) 
       C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ) 
       C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ) 
       C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 )   */
    /* hold 
       A( 0, p ) 
       A( 1, p ) 
       A( 2, p ) 
       A( 3, p ) */
  
  v2df_t 
    c_00_c_10_vreg, c_01_c_11_vreg, c_02_c_12_vreg, c_03_c_13_vreg,
    c_20_c_30_vreg, c_21_c_31_vreg, c_22_c_32_vreg, c_23_c_33_vreg,
    a_0p_a_1p_vreg,
    a_2p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

    

  double
    *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  b_p0_pntr = &B(0,0);
  b_p1_pntr = &B(0,1);
  b_p2_pntr = &B(0,2);
  b_p3_pntr = &B(0,3);

  c_00_c_10_vreg.v = _mm_setzero_pd();
  c_01_c_11_vreg.v = _mm_setzero_pd();
  c_02_c_12_vreg.v = _mm_setzero_pd();
  c_03_c_13_vreg.v = _mm_setzero_pd();
  c_20_c_30_vreg.v = _mm_setzero_pd();
  c_21_c_31_vreg.v = _mm_setzero_pd();
  c_22_c_32_vreg.v = _mm_setzero_pd();
  c_23_c_33_vreg.v = _mm_setzero_pd();




  for(p = 0; p < k; p++)
  {
    /*
    a_0p_reg = A( 0, p );
    a_1p_reg = A( 1, p );
    a_2p_reg = A( 2, p );
    a_3p_reg = A( 3, p ); 
    */

    a_0p_a_1p_vreg.v = _mm_loadu_pd((double *) &A( 0, p));
    a_2p_a_3p_vreg.v = _mm_loadu_pd((double *) &A( 2, p));

    /*
    b_p0_reg = *b_p0_pntr++; 
    b_p1_reg = *b_p1_pntr++; 
    b_p2_reg = *b_p2_pntr++; 
    b_p3_reg = *b_p3_pntr++; 
    */

    // load and duplicate
    b_p0_vreg.v = _mm_loaddup_pd((double*) b_p0_pntr++) ;
    b_p1_vreg.v = _mm_loaddup_pd((double*) b_p1_pntr++) ;
    b_p2_vreg.v = _mm_loaddup_pd((double*) b_p2_pntr++) ;
    b_p3_vreg.v = _mm_loaddup_pd((double*) b_p3_pntr++) ;

    // first and second row
    c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
    c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
    c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
    c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;
    // third and fourth row

    c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
    c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
    c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
    c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;

  }

  C( 0, 0 ) += c_00_c_10_vreg.d[0];   C( 0, 1 ) += c_01_c_11_vreg.d[0];   C( 0, 2 ) += c_02_c_12_vreg.d[0];   C( 0, 3 ) += c_03_c_13_vreg.d[0];
  C( 1, 0 ) += c_00_c_10_vreg.d[1];   C( 1, 1 ) += c_01_c_11_vreg.d[1];   C( 1, 2 ) += c_02_c_12_vreg.d[1];   C( 1, 3 ) += c_03_c_13_vreg.d[1];
  C( 2, 0 ) += c_20_c_30_vreg.d[0];   C( 2, 1 ) += c_21_c_31_vreg.d[0];   C( 2, 2 ) += c_22_c_32_vreg.d[0];   C( 2, 3 ) += c_23_c_33_vreg.d[0];
  C( 3, 0 ) += c_20_c_30_vreg.d[1];   C( 3, 1 ) += c_21_c_31_vreg.d[1];   C( 3, 2 ) += c_22_c_32_vreg.d[1];   C( 3, 3 ) += c_23_c_33_vreg.d[1];
}