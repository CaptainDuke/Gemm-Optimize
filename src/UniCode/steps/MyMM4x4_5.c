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
 

  for(int p = 0; p < k; p++){
    C(0,0) += A(0, p) * B(p, 0);
    C(0,1) += A(0, p) * B(p, 1);
    C(0,2) += A(0, p) * B(p, 2);
    C(0,3) += A(0, p) * B(p, 3);

    C(1,0) += A(1, p) * B(p, 0);
    C(1,1) += A(1, p) * B(p, 1);
    C(1,2) += A(1, p) * B(p, 2);
    C(1,3) += A(1, p) * B(p, 3);

    C(2,0) += A(2, p) * B(p, 0);
    C(2,1) += A(2, p) * B(p, 1);
    C(2,2) += A(2, p) * B(p, 2);
    C(2,3) += A(2, p) * B(p, 3);

    C(3,0) += A(3, p) * B(p, 0);
    C(3,1) += A(3, p) * B(p, 1);
    C(3,2) += A(3, p) * B(p, 2);
    C(3,3) += A(3, p) * B(p, 3);


  }


  
  
}

