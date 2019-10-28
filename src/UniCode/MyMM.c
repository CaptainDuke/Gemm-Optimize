/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */

#define nc 256
#define kc 128
#define nb 1000
#define min(i, j) ( (i) < (j) ? (i) : (j))


void PackMatrixA4( int, float *, int, float * );
void PackMatrixA8( int, float *, int, float * );
void PackMatrixB4( int, float *, int, float * );
void PackMatrixB8( int, float *, int, float * );
void InnerKernel4( int, int, int, float *, int, float *, int, float *, int , int);
void InnerKernel8( int, int, int, float *, int, float *, int, float *, int , int);

void AddDot4x4(int, float *, int, float *, int, float *, int);
void AddDot8x8(int, float *, int, float *, int, float *, int);

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

      // InnerKernel4(m, jb, pb, &A(0, p), lda, &B(p, j), ldb, &C(0, j), ldc, j==0);
      InnerKernel8(m, jb, pb, &A(0, p), lda, &B(p, j), ldb, &C(0, j), ldc, j==0);
   
    }
  }
}

void InnerKernel4(int m, int n, int k, float *a, int lda,
                                      float *b, int ldb,
                                      float *c, int ldc, int first_time)
{
  int i, j;
  float packedB[k * n];
  static float packedA[ kc * nb];

  for(i = 0; i < m; i+=4){
    if(first_time)
      PackMatrixA4(k, &A(i, 0), lda, &packedA[i*k]);
    for(j = 0; j < n; j+=4){
      if(i == 0)
        PackMatrixB4(k, &B(0, j), ldb, &packedB[k * j]);
      AddDot4x4(k, &packedA[i * k], k, &packedB[k*j], 4, &C(i, j), ldc);
      // AddDot4x4(k, &A(i, 0), lda, &packedB[k*j], 4, &C(i, j), ldc);
    }
  }
} 

void InnerKernel8(int m, int n, int k, float *a, int lda,
                                      float *b, int ldb,
                                      float *c, int ldc, int first_time)
{
  int i, j;
  float packedB[k * n];
  static float packedA[ kc * nb];

  for(i = 0; i < m; i+=8){
    if(first_time)
      PackMatrixA8(k, &A(i, 0), lda, &packedA[i*k]);
    for(j = 0; j < n; j+=8){
      if(i == 0)
        PackMatrixB8(k, &B(0, j), ldb, &packedB[k * j]);
      AddDot8x8(k, &packedA[i * k], k, &packedB[k*j], 8, &C(i, j), ldc);
      // AddDot4x4(k, &A(i, 0), lda, &packedB[k*j], 4, &C(i, j), ldc);
    }
  }
}  

void PackMatrixA4( int k, float *a, int lda, float *a_to)
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

void PackMatrixA8( int k, float *a, int lda, float *a_to)
{
  int j;
  float
    *a_0j_ptr = &A(0, 0),
    *a_1j_ptr = &A(1, 0),
    *a_2j_ptr = &A(2, 0),
    *a_3j_ptr = &A(3, 0),

    *a_4j_ptr = &A(4, 0),
    *a_5j_ptr = &A(5, 0),
    *a_6j_ptr = &A(6, 0),
    *a_7j_ptr = &A(7, 0);

  for(j = 0; j < k; j++){

    *a_to = *a_0j_ptr++;
    *(a_to+1) = *a_1j_ptr++;
    *(a_to+2) = *a_2j_ptr++;
    *(a_to+3) = *a_3j_ptr++;

    *(a_to+4) = *a_4j_ptr++;
    *(a_to+5) = *a_5j_ptr++;
    *(a_to+6) = *a_6j_ptr++;
    *(a_to+7) = *a_7j_ptr++;

    a_to+=8;

    

  }
}

void PackMatrixB4( int k, float *b, int ldb, float *b_to)
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

void PackMatrixB8( int k, float *b, int ldb, float *b_to)
{
  int i;
  for(i = 0; i < k; i++){
    float *b_ij_ptr = &B(i, 0);

    *b_to     = *b_ij_ptr;
    *(b_to+1) = *(b_ij_ptr+1);
    *(b_to+2) = *(b_ij_ptr+2);
    *(b_to+3) = *(b_ij_ptr+3);

    *(b_to+4) = *(b_ij_ptr+4);
    *(b_to+5) = *(b_ij_ptr+5);
    *(b_to+6) = *(b_ij_ptr+6);
    *(b_to+7) = *(b_ij_ptr+7);


    b_to += 8;
  }
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>  // fma



void AddDot4x4(int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
 



  __m128
    c00_c01_c02_c03_vreg, c10_c11_c12_c13_vreg, c20_c21_c22_c23_vreg, c30_c31_c32_c33_vreg,
    
    bp0_bp1_bp2_bp3_vreg,

    a_0p_vreg,
    a_1p_vreg,
    a_2p_vreg,
    a_3p_vreg;

  

    c00_c01_c02_c03_vreg = _mm_setzero_ps();
    c10_c11_c12_c13_vreg = _mm_setzero_ps();
    c20_c21_c22_c23_vreg = _mm_setzero_ps();
    c30_c31_c32_c33_vreg = _mm_setzero_ps();

    

    for(int p = 0; p < k; p++){

      // bp0_bp1_bp2_bp3_vreg = _mm_loadu_ps((float*) &B(p, 0));
      bp0_bp1_bp2_bp3_vreg = _mm_loadu_ps((float *) b);
      b+=4;

      a_0p_vreg = _mm_load_ps1((float*) a);
      a_1p_vreg = _mm_load_ps1((float*) (a+1) );
      a_2p_vreg = _mm_load_ps1((float*) (a+2) );
      a_3p_vreg = _mm_load_ps1((float*) (a+3) );
      a+=4;

      //first row and second row
      // sse
      // c00_c01_c02_c03_vreg += _mm_mul_ps(a_0p_vreg , bp0_bp1_bp2_bp3_vreg);
      // c10_c11_c12_c13_vreg += _mm_mul_ps(a_1p_vreg , bp0_bp1_bp2_bp3_vreg);
      
      // fma
      c00_c01_c02_c03_vreg = _mm_fmadd_ps(a_0p_vreg, bp0_bp1_bp2_bp3_vreg, c00_c01_c02_c03_vreg);
      c10_c11_c12_c13_vreg = _mm_fmadd_ps(a_1p_vreg, bp0_bp1_bp2_bp3_vreg, c10_c11_c12_c13_vreg);

      // naive
      // c00_c01_c02_c03_vreg += a_0p_vreg * bp0_bp1_bp2_bp3_vreg;
      // c10_c11_c12_c13_vreg += a_1p_vreg * bp0_bp1_bp2_bp3_vreg;
    

      // Third and fourth row
      // sse
      // c20_c21_c22_c23_vreg += _mm_mul_ps(a_2p_vreg , bp0_bp1_bp2_bp3_vreg);
      // c30_c31_c32_c33_vreg += _mm_mul_ps(a_3p_vreg , bp0_bp1_bp2_bp3_vreg);

      // fma
      c20_c21_c22_c23_vreg = _mm_fmadd_ps(a_2p_vreg, bp0_bp1_bp2_bp3_vreg, c20_c21_c22_c23_vreg);
      c30_c31_c32_c33_vreg = _mm_fmadd_ps(a_3p_vreg, bp0_bp1_bp2_bp3_vreg, c30_c31_c32_c33_vreg);

      // naive
      // c20_c21_c22_c23_vreg += a_2p_vreg * bp0_bp1_bp2_bp3_vreg;
      // c30_c31_c32_c33_vreg += a_3p_vreg * bp0_bp1_bp2_bp3_vreg;



    }

    // C(0, 0) += c00_c01_c02_c03_vreg.d[0]; C(0, 1) += c00_c01_c02_c03_vreg.d[1]; C(0, 2) += c00_c01_c02_c03_vreg.d[2]; C(0, 3) += c00_c01_c02_c03_vreg.d[3];
    // C(1, 0) += c10_c11_c12_c13_vreg.d[0]; C(1, 1) += c10_c11_c12_c13_vreg.d[1]; C(1, 2) += c10_c11_c12_c13_vreg.d[2]; C(1, 3) += c10_c11_c12_c13_vreg.d[3];
    // C(2, 0) += c20_c21_c22_c23_vreg.d[0]; C(2, 1) += c20_c21_c22_c23_vreg.d[1]; C(2, 2) += c20_c21_c22_c23_vreg.d[2]; C(2, 3) += c20_c21_c22_c23_vreg.d[3];
    // C(3, 0) += c30_c31_c32_c33_vreg.d[0]; C(3, 1) += c30_c31_c32_c33_vreg.d[1]; C(3, 2) += c30_c31_c32_c33_vreg.d[2]; C(3, 3) += c30_c31_c32_c33_vreg.d[3];

    __m128 C00_03 = _mm_loadu_ps(&C(0,0));
    __m128 C10_13 = _mm_loadu_ps(&C(1,0));
    __m128 C20_23 = _mm_loadu_ps(&C(2,0));
    __m128 C30_33 = _mm_loadu_ps(&C(3,0));

    C00_03 = _mm_add_ps(C00_03, c00_c01_c02_c03_vreg);
    C10_13 = _mm_add_ps(C10_13, c10_c11_c12_c13_vreg);
    C20_23 = _mm_add_ps(C20_23, c20_c21_c22_c23_vreg);
    C30_33 = _mm_add_ps(C30_33, c30_c31_c32_c33_vreg);

    _mm_storeu_ps(&C(0,0), C00_03);
    _mm_storeu_ps(&C(1,0), C10_13);
    _mm_storeu_ps(&C(2,0), C20_23);
    _mm_storeu_ps(&C(3,0), C30_33);


}

void AddDot8x8(int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
 



  __m256
    c00_to_c07_vreg, c10_to_c17_vreg, c20_to_c27_vreg, c30_to_c37_vreg,
    c40_to_c47_vreg, c50_to_c57_vreg, c60_to_c67_vreg, c70_to_c77_vreg,
    
    bp0_to_bp7_vreg,

    a_0p_vreg,
    a_1p_vreg,
    a_2p_vreg,
    a_3p_vreg,

    a_4p_vreg,
    a_5p_vreg,
    a_6p_vreg,
    a_7p_vreg;

  

    c00_to_c07_vreg = _mm256_setzero_ps();
    c10_to_c17_vreg = _mm256_setzero_ps();
    c20_to_c27_vreg = _mm256_setzero_ps();
    c30_to_c37_vreg = _mm256_setzero_ps();

    c40_to_c47_vreg = _mm256_setzero_ps();
    c50_to_c57_vreg = _mm256_setzero_ps();
    c60_to_c67_vreg = _mm256_setzero_ps();
    c70_to_c77_vreg = _mm256_setzero_ps();

    

    for(int p = 0; p < k; p++){

      // bp0_bp1_bp2_bp3_vreg = _mm_loadu_ps((float*) &B(p, 0));
      bp0_to_bp7_vreg = _mm256_loadu_ps((float *) b);
      b+=8;

      a_0p_vreg = _mm256_broadcast_ss((float*) a);
      a_1p_vreg = _mm256_broadcast_ss((float*) (a+1) );
      a_2p_vreg = _mm256_broadcast_ss((float*) (a+2) );
      a_3p_vreg = _mm256_broadcast_ss((float*) (a+3) );

      a_4p_vreg = _mm256_broadcast_ss((float*) (a+4) );
      a_5p_vreg = _mm256_broadcast_ss((float*) (a+5) );
      a_6p_vreg = _mm256_broadcast_ss((float*) (a+6) );
      a_7p_vreg = _mm256_broadcast_ss((float*) (a+7) );


      a+=8;

      //first row and second row
      // sse
      // c00_c01_c02_c03_vreg += _mm_mul_ps(a_0p_vreg , bp0_bp1_bp2_bp3_vreg);
      // c10_c11_c12_c13_vreg += _mm_mul_ps(a_1p_vreg , bp0_bp1_bp2_bp3_vreg);
      
      // fma
      c00_to_c07_vreg = _mm256_fmadd_ps(a_0p_vreg, bp0_to_bp7_vreg, c00_to_c07_vreg);
      c10_to_c17_vreg = _mm256_fmadd_ps(a_1p_vreg, bp0_to_bp7_vreg, c10_to_c17_vreg);
      c20_to_c27_vreg = _mm256_fmadd_ps(a_2p_vreg, bp0_to_bp7_vreg, c20_to_c27_vreg);
      c30_to_c37_vreg = _mm256_fmadd_ps(a_3p_vreg, bp0_to_bp7_vreg, c30_to_c37_vreg);

      c40_to_c47_vreg = _mm256_fmadd_ps(a_4p_vreg, bp0_to_bp7_vreg, c40_to_c47_vreg);
      c50_to_c57_vreg = _mm256_fmadd_ps(a_5p_vreg, bp0_to_bp7_vreg, c50_to_c57_vreg);
      c60_to_c67_vreg = _mm256_fmadd_ps(a_6p_vreg, bp0_to_bp7_vreg, c60_to_c67_vreg);
      c70_to_c77_vreg = _mm256_fmadd_ps(a_7p_vreg, bp0_to_bp7_vreg, c70_to_c77_vreg);

      // naive
      // c00_c01_c02_c03_vreg += a_0p_vreg * bp0_bp1_bp2_bp3_vreg;
      // c10_c11_c12_c13_vreg += a_1p_vreg * bp0_bp1_bp2_bp3_vreg;
    

      // Third and fourth row
      // sse
      // c20_c21_c22_c23_vreg += _mm_mul_ps(a_2p_vreg , bp0_bp1_bp2_bp3_vreg);
      // c30_c31_c32_c33_vreg += _mm_mul_ps(a_3p_vreg , bp0_bp1_bp2_bp3_vreg);

      // fma
      // c20_to_c27_vreg = _mm_fmadd_ps(a_2p_vreg, bp0_bp1_bp2_bp3_vreg, c20_c21_c22_c23_vreg);
      // c30_c31_c32_c33_vreg = _mm_fmadd_ps(a_3p_vreg, bp0_bp1_bp2_bp3_vreg, c30_c31_c32_c33_vreg);

      // naive
      // c20_c21_c22_c23_vreg += a_2p_vreg * bp0_bp1_bp2_bp3_vreg;
      // c30_c31_c32_c33_vreg += a_3p_vreg * bp0_bp1_bp2_bp3_vreg;



    }

    // C(0, 0) += c00_c01_c02_c03_vreg.d[0]; C(0, 1) += c00_c01_c02_c03_vreg.d[1]; C(0, 2) += c00_c01_c02_c03_vreg.d[2]; C(0, 3) += c00_c01_c02_c03_vreg.d[3];
    // C(1, 0) += c10_c11_c12_c13_vreg.d[0]; C(1, 1) += c10_c11_c12_c13_vreg.d[1]; C(1, 2) += c10_c11_c12_c13_vreg.d[2]; C(1, 3) += c10_c11_c12_c13_vreg.d[3];
    // C(2, 0) += c20_c21_c22_c23_vreg.d[0]; C(2, 1) += c20_c21_c22_c23_vreg.d[1]; C(2, 2) += c20_c21_c22_c23_vreg.d[2]; C(2, 3) += c20_c21_c22_c23_vreg.d[3];
    // C(3, 0) += c30_c31_c32_c33_vreg.d[0]; C(3, 1) += c30_c31_c32_c33_vreg.d[1]; C(3, 2) += c30_c31_c32_c33_vreg.d[2]; C(3, 3) += c30_c31_c32_c33_vreg.d[3];


    // 0-3 rows
    __m256 C00_07 = _mm256_loadu_ps(&C(0,0));
    C00_07 = _mm256_add_ps(C00_07, c00_to_c07_vreg);
    _mm256_storeu_ps(&C(0,0), C00_07);


    __m256 C10_17 = _mm256_loadu_ps(&C(1,0));
    C10_17 = _mm256_add_ps(C10_17, c10_to_c17_vreg);
    _mm256_storeu_ps(&C(1,0), C10_17);

    __m256 C20_27 = _mm256_loadu_ps(&C(2,0));
    C20_27 = _mm256_add_ps(C20_27, c20_to_c27_vreg);
    _mm256_storeu_ps(&C(2,0), C20_27);

    __m256 C30_37 = _mm256_loadu_ps(&C(3,0));
    C30_37 = _mm256_add_ps(C30_37, c30_to_c37_vreg);
    _mm256_storeu_ps(&C(3,0), C30_37);


    // 4 - 7 rows
    __m256 C40_47 = _mm256_loadu_ps(&C(4,0));
    C40_47 = _mm256_add_ps(C40_47, c40_to_c47_vreg);
    _mm256_storeu_ps(&C(4,0), C40_47);

    __m256 C50_57 = _mm256_loadu_ps(&C(5,0));
    C50_57 = _mm256_add_ps(C50_57, c50_to_c57_vreg);
    _mm256_storeu_ps(&C(5,0), C50_57);

    __m256 C60_67 = _mm256_loadu_ps(&C(6,0));
    C60_67 = _mm256_add_ps(C60_67, c60_to_c67_vreg);
    _mm256_storeu_ps(&C(6,0), C60_67);

    __m256 C70_77 = _mm256_loadu_ps(&C(7,0));
    C70_77 = _mm256_add_ps(C70_77, c70_to_c77_vreg);
    _mm256_storeu_ps(&C(7,0), C70_77);

  

}