/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

/* Routine for computing C = A * B + C */

void AddDot( int, float *, int, float *, float * );
void AddDot1x4(int, float *, int, float *, int, float *, int);

void MY_MMult( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;
  for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
    for ( j=0; j<n; j+=1 ){        /* Loop over the columns of C */
      /* Update the C( i,j ) with the inner product of the ith row of A
	 and the jth column of B */
      AddDot1x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
   
    }
  }
}

void AddDot1x4(int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
 
  int p;
  // for(p = 0; p < k; p++){
  //   C(0, 0) += A(0, p) * B(p, 0);
  //   C(1, 0) += A(1, p) * B(p, 0);
  //   C(2, 0) += A(2, p) * B(p, 0);
  //   C(3, 0) += A(3, p) * B(p, 0);
  // }

  register float
      c_00_reg, c_10_reg, c_20_reg, c_30_reg,
        // hold contributions to C(0,0), C(1,0), C(2,0), C(3,0)
      b_p0_reg;
        // holds B(p,0)
  
  float 
    *a0p_ptr, *a1p_ptr, *a2p_ptr, *a3p_ptr;
  
  a0p_ptr = &A(0, 0);
  a1p_ptr = &A(1, 0);
  a2p_ptr = &A(2, 0);
  a3p_ptr = &A(3, 0);
  

  c_00_reg = 0.0;
  c_10_reg = 0.0;
  c_20_reg = 0.0;
  c_30_reg = 0.0;
  
  for(p = 0; p < k; p+=4){
    
    b_p0_reg = B(p, 0);

    c_00_reg += *a0p_ptr * b_p0_reg;
    c_10_reg += *a1p_ptr * b_p0_reg;
    c_20_reg += *a2p_ptr * b_p0_reg;
    c_30_reg += *a3p_ptr * b_p0_reg;

    b_p0_reg = B(p+1, 0);

    c_00_reg += *(a0p_ptr+1) * b_p0_reg;
    c_10_reg += *(a1p_ptr+1) * b_p0_reg;
    c_20_reg += *(a2p_ptr+1) * b_p0_reg;
    c_30_reg += *(a3p_ptr+1) * b_p0_reg;

    b_p0_reg = B(p+2, 0);

    c_00_reg += *(a0p_ptr+2) * b_p0_reg;
    c_10_reg += *(a1p_ptr+2) * b_p0_reg;
    c_20_reg += *(a2p_ptr+2) * b_p0_reg;
    c_30_reg += *(a3p_ptr+2) * b_p0_reg;

    b_p0_reg = B(p+3, 0);

    c_00_reg += *(a0p_ptr+3) * b_p0_reg;
    c_10_reg += *(a1p_ptr+3) * b_p0_reg;
    c_20_reg += *(a2p_ptr+3) * b_p0_reg;
    c_30_reg += *(a3p_ptr+3) * b_p0_reg;

    a0p_ptr+=4;
    a1p_ptr+=4;
    a2p_ptr+=4;
    a3p_ptr+=4;
  }

  C(0, 0) += c_00_reg;
  C(1, 0) += c_10_reg;
  C(2, 0) += c_20_reg;
  C(3, 0) += c_30_reg;


  
  
}

