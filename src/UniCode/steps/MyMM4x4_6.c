/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */

void AddDot( int, float *, int, float *, float * );
void AddDot4x4(int, float *, int, float *, int, float *, int);

void MY_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;
  for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
    for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */
      AddDot4x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
   
    }
  }
}

void AddDot4x4(int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
 



  register float
    c_00_reg, c_01_reg, c_02_reg, c_03_reg, 
    c_10_reg, c_11_reg, c_12_reg, c_13_reg, 
    c_20_reg, c_21_reg, c_22_reg, c_23_reg, 
    c_30_reg, c_31_reg, c_32_reg, c_33_reg,

    // store B(p, 0-3)
    b_p0_reg, b_p1_reg, b_p2_reg, b_p3_reg;

    c_00_reg = 0.0; c_01_reg = 0.0; c_02_reg = 0.0; c_03_reg = 0.0; 
    c_10_reg = 0.0; c_11_reg = 0.0; c_12_reg = 0.0; c_13_reg = 0.0; 
    c_20_reg = 0.0; c_21_reg = 0.0; c_22_reg = 0.0; c_23_reg = 0.0; 
    c_30_reg = 0.0; c_31_reg = 0.0; c_32_reg = 0.0; c_33_reg = 0.0;

    for(int p = 0; p < k; p++){
      b_p0_reg = B(p,0);
      b_p1_reg = B(p,1);
      b_p2_reg = B(p,2);
      b_p3_reg = B(p,3);

      c_00_reg += A(0, p) * b_p0_reg;
      c_01_reg += A(0, p) * b_p1_reg;
      c_02_reg += A(0, p) * b_p2_reg;
      c_03_reg += A(0, p) * b_p3_reg;

      c_10_reg += A(1, p) * b_p0_reg;
      c_11_reg += A(1, p) * b_p1_reg;
      c_12_reg += A(1, p) * b_p2_reg;
      c_13_reg += A(1, p) * b_p3_reg;

      c_20_reg += A(2, p) * b_p0_reg;
      c_21_reg += A(2, p) * b_p1_reg;
      c_22_reg += A(2, p) * b_p2_reg;
      c_23_reg += A(2, p) * b_p3_reg;

      c_30_reg += A(3, p) * b_p0_reg;
      c_31_reg += A(3, p) * b_p1_reg;
      c_32_reg += A(3, p) * b_p2_reg;
      c_33_reg += A(3, p) * b_p3_reg;

    }

    C(0, 0) = c_00_reg; C(0, 1) = c_01_reg; C(0, 2) = c_02_reg; C(0, 3) = c_03_reg;
    C(1, 0) = c_10_reg; C(1, 1) = c_11_reg; C(1, 2) = c_12_reg; C(1, 3) = c_13_reg;
    C(2, 0) = c_20_reg; C(2, 1) = c_21_reg; C(2, 2) = c_22_reg; C(2, 3) = c_23_reg;
    C(3, 0) = c_30_reg; C(3, 1) = c_31_reg; C(3, 2) = c_32_reg; C(3, 3) = c_33_reg;
  
}

